"""Learn K prototype wavetables with a VAE whose decoder is a softmax mixture of K learnable prototypes.

Settings: z_dim 32, loss MSE + 0.5 KL, Adam 1e-3, cosine annealing, 100 epochs, batch 64, gradient clipping 1.0.
By default only the training split is used.
Output: <out>/prototypes.npy (K, 2048) (raw; scripts normalize each row to max |value| = 1 when using it)
Usage:
  python scripts/train_dictionary.py --data data/processed --K 20 --out runs/main
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2w.data import Dataset
from t2w.models import WavetableVAE, dictionary_vae_loss

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--out", default="runs/main")
ap.add_argument("--K", type=int, default=20)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--use-all", action="store_true", help="also use the validation split")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = Dataset(args.data)
X = ds.waves if args.use_all else ds.waves[ds.train]
loader = DataLoader(torch.from_numpy(X), batch_size=args.batch_size, shuffle=True)
model = WavetableVAE(args.K).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
log = []
for ep in range(1, args.epochs + 1):
    model.train(); s = np.zeros(3); n = 0
    for x in loader:
        x = x.to(dev); opt.zero_grad()
        recon, mu, lv, _ = model(x)
        loss, mse, kl = dictionary_vae_loss(recon, x, mu, lv)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        s += [loss.item(), mse, kl]; n += 1
    sch.step()
    log.append(dict(zip(["epoch", "loss", "mse", "kl"], [ep, *(s / n)])))
    if ep % 10 == 0 or ep == 1:
        print(f"epoch {ep:3d}  loss {s[0] / n:.4f}  mse {s[1] / n:.4f}  kl {s[2] / n:.4f}", flush=True)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
np.save(out / "prototypes.npy", model.decoder.prototypes.detach().cpu().numpy())
json.dump({"args": vars(args), "n_waves": int(len(X)), "log": log}, open(out / "dictionary_log.json", "w"), indent=1)
print("saved", out / "prototypes.npy")
