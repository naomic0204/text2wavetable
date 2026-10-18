"""Train DWTS-CVAE: text (SBERT) -> alpha over K prototypes -> wavetable alpha @ P.

Loss: waveform MSE + beta * KL (free bits 0.05 per dimension), beta warmed up linearly to 1 over 100 epochs.
AdamW 3e-4 (weight decay 1e-4), ReduceLROnPlateau after warmup, early stopping (patience 60, after warmup), at most 500 epochs,
batch 64, latent 32, seed 42. The checkpoint with the lowest validation loss (after warmup) is kept.
Output: <run>/cvae.pt, <run>/cvae_log.json
Usage:
  python scripts/train_cvae.py --data data/processed --run runs/main
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2w.data import Dataset
from t2w.models import DWTSCVAENet, dwts_cvae_loss, normalize_prototypes

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--run", default="runs/main")
ap.add_argument("--prototypes", default=None, help="default: <run>/prototypes.npy")
ap.add_argument("--alpha", default=None, help="default: <run>/alpha.npy")
ap.add_argument("--latent", type=int, default=32)
ap.add_argument("--epochs", type=int, default=500)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--patience", type=int, default=60)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

run = Path(args.run)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = Dataset(args.data)
P = torch.from_numpy(normalize_prototypes(np.load(args.prototypes or run / "prototypes.npy"))).to(dev)
A = np.load(args.alpha or run / "alpha.npy").astype(np.float32)
K = P.shape[0]
full = TensorDataset(torch.from_numpy(ds.text_emb(recompute=True)), torch.from_numpy(A), torch.from_numpy(ds.waves))
tr_idx, va_idx = ds.split_order()
train_loader = DataLoader(Subset(full, tr_idx), batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(Subset(full, va_idx), batch_size=64, shuffle=False)

model = DWTSCVAENet(K=K, cond_dim=full.tensors[0].shape[1], latent_dim=args.latent).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, min_lr=1e-6)
best, wait, log, t0 = float("inf"), 0, [], time.time()
for ep in range(1, args.epochs + 1):
    beta = min(1.0, ep / max(1, args.warmup))
    model.train(); tl = tw = tk = 0.0
    for c, a, w in train_loader:
        c, a, w = c.to(dev), a.to(dev), w.to(dev); opt.zero_grad()
        ap_, mu, lv = model(a, c)
        loss, wl, kl = dwts_cvae_loss(ap_, mu, lv, P, w, beta=beta)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item() * len(c); tw += wl * len(c); tk += kl * len(c)
    model.eval(); vl = 0.0
    with torch.no_grad():
        for c, a, w in val_loader:
            c, a, w = c.to(dev), a.to(dev), w.to(dev)
            ap_, mu, lv = model(a, c)
            vl += dwts_cvae_loss(ap_, mu, lv, P, w, beta=beta)[0].item() * len(c)
    nt, nv = len(train_loader.dataset), len(ds.val)
    vl /= nv
    log.append({"epoch": ep, "beta": beta, "train": tl / nt, "train_wt": tw / nt, "train_kl": tk / nt, "val": vl})
    if ep >= args.warmup:
        sch.step(vl)
        if vl < best:
            best, wait = vl, 0
            torch.save({"model_state": model.state_dict(), "K": K, "latent_dim": args.latent, "cond_dim": model.encoder.net[0].in_features - K,
                        "epoch": ep, "val_loss": best}, run / "cvae.pt")
        else:
            wait += 1
    if ep % 10 == 0 or ep == 1:
        print(f"epoch {ep:4d}  beta {beta:.2f}  train {tl / nt:.5f}  val {vl:.5f}  best {best:.5f}", flush=True)
    if ep >= args.warmup and wait >= args.patience:
        break
json.dump({"args": vars(args), "last_epoch": ep, "best_val": best, "seconds": time.time() - t0, "log": log},
          open(run / "cvae_log.json", "w"), indent=1)
print("saved", run / "cvae.pt", "best val", best)
