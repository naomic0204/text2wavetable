"""Decompose every waveform onto the (row-normalized) prototypes with NNLS; weights are normalized to sum to 1
(uniform if NNLS returns all zeros). These alpha vectors are the CVAE / HybridNet targets and the Map&Steal outputs.
Output: <run>/alpha.npy (N, K)
Usage:
  python scripts/fit_alpha.py --data data/processed --run runs/main
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2w.data import Dataset
from t2w.models import normalize_prototypes

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--run", default="runs/main")
ap.add_argument("--prototypes", default=None, help="default: <run>/prototypes.npy")
args = ap.parse_args()

ds = Dataset(args.data)
P = normalize_prototypes(np.load(args.prototypes or Path(args.run) / "prototypes.npy"), dtype=np.float64)
K = len(P)
A = np.zeros((len(ds.waves), K))
for i, w in enumerate(ds.waves64):
    a, _ = nnls(P.T, w)
    A[i] = a / a.sum() if a.sum() > 1e-6 else np.ones(K) / K
np.save(Path(args.run) / "alpha.npy", A.astype(np.float32))
print("saved", Path(args.run) / "alpha.npy", A.shape)
