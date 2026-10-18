"""Train the HybridNet baseline: SBERT text -> MLP (hidden 1024, 2 residual blocks) with three heads:
synth parameters (min-max scaled, MSE), alpha over K prototypes (MSE to the NNLS alpha), and 8 modulation slots
(source / destination classification with label smoothing 0.1 + amount regression).
Loss weights: parameters 1.0, alpha 0.3, modulations 0.05. Adam 1e-3 (weight decay 1e-4), cosine annealing, 500 epochs,
batch 256, gradient clipping 1.0, seed 42. The epoch with the lowest validation (parameter + 0.3 alpha) loss is kept.
Same data, split and alpha targets as the CVAE.
Output: <run>/hybridnet.pt, <run>/hybridnet_meta.pkl (scaler, label encoders), <run>/hybridnet_log.json
Usage:
  python scripts/train_hybridnet.py --data data/processed --run runs/main
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2w.data import Dataset
from t2w.models import HYBRID_LAMBDA, VitalHybridNet, hybrid_loss
from t2w.vital_io import MAX_MODS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--run", default="runs/main")
ap.add_argument("--alpha", default=None, help="default: <run>/alpha.npy")
ap.add_argument("--epochs", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

run = Path(args.run)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = Dataset(args.data)
df = ds.df
mod_cols = [f"mod_{i}_{t}" for i in range(MAX_MODS) for t in ["src", "dst", "amt"]]
param_cols = [c for c in df.columns if c not in ["filename", "prompt", "split"] + mod_cols]
scaler = MinMaxScaler()
params = scaler.fit_transform(df[param_cols].apply(lambda s: s.astype(float)).fillna(0).values).astype(np.float32)
src_enc = LabelEncoder().fit(sorted(set(v for i in range(MAX_MODS) for v in df[f"mod_{i}_src"].fillna("None"))))
dst_enc = LabelEncoder().fit(sorted(set(v for i in range(MAX_MODS) for v in df[f"mod_{i}_dst"].fillna("None"))))
n = len(df)
mod_src = np.stack([src_enc.transform(df[f"mod_{i}_src"].fillna("None")) for i in range(MAX_MODS)], 1).astype(np.int64)
mod_dst = np.stack([dst_enc.transform(df[f"mod_{i}_dst"].fillna("None")) for i in range(MAX_MODS)], 1).astype(np.int64)
mod_amt = np.stack([np.clip(df[f"mod_{i}_amt"].fillna(0).values.astype(np.float32), -1, 1) for i in range(MAX_MODS)], 1)
mod_active = np.stack([(df[f"mod_{i}_src"].fillna("None") != "None").values & (df[f"mod_{i}_dst"].fillna("None") != "None").values
                       for i in range(MAX_MODS)], 1).astype(np.float32)
A = np.load(args.alpha or run / "alpha.npy").astype(np.float32)
T = ds.text_emb(recompute=True)
full = TensorDataset(*[torch.from_numpy(x) for x in [T, params, A, np.ones(n, np.float32), mod_src, mod_dst, mod_amt, mod_active]])
train_loader = DataLoader(Subset(full, ds.train.tolist()), batch_size=256, shuffle=True)
val_loader = DataLoader(Subset(full, ds.val.tolist()), batch_size=256, shuffle=False)

model = VitalHybridNet(text_dim=T.shape[1], n_params=len(param_cols), K=A.shape[1],
                       n_src=len(src_enc.classes_), n_dst=len(dst_enc.classes_)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
best, best_ep, log, t0 = float("inf"), 0, [], time.time()
for ep in range(1, args.epochs + 1):
    model.train(); s = np.zeros(4); k = 0
    for b in train_loader:
        b = [x.to(dev) for x in b]; opt.zero_grad()
        loss, p, w, m = hybrid_loss(model(b[0]), b)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        s += [loss.item(), p, w, m]; k += 1
    sch.step()
    model.eval(); v = np.zeros(3); vk = 0
    with torch.no_grad():
        for b in val_loader:
            b = [x.to(dev) for x in b]
            loss, p, w, _ = hybrid_loss(model(b[0]), b)
            v += [loss.item(), p, w]; vk += 1
    val_pw = HYBRID_LAMBDA["param"] * v[1] / vk + HYBRID_LAMBDA["wt"] * v[2] / vk
    log.append({"epoch": ep, "train": s[0] / k, "val": v[0] / vk, "val_param_wt": val_pw})
    if val_pw < best:
        best, best_ep = val_pw, ep
        torch.save(model.state_dict(), run / "hybridnet.pt")
    if ep % 50 == 0 or ep == 1:
        print(f"epoch {ep:3d}  train {s[0] / k:.4f}  val_param_wt {val_pw:.4f}  best@{best_ep}", flush=True)

pickle.dump({"param_cols": param_cols, "scaler": scaler, "src_encoder": src_enc, "dst_encoder": dst_enc,
             "n_params": len(param_cols), "n_src": len(src_enc.classes_), "n_dst": len(dst_enc.classes_), "K": A.shape[1],
             "text_dim": T.shape[1]}, open(run / "hybridnet_meta.pkl", "wb"))
json.dump({"args": vars(args), "best_epoch": best_ep, "best_val_param_wt": best, "seconds": time.time() - t0, "log": log},
          open(run / "hybridnet_log.json", "w"), indent=1)
print("saved", run / "hybridnet.pt", "best epoch", best_ep, "val_param_wt", best)
