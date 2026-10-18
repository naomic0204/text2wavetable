"""Build the (text, wavetable) dataset from a collection of Vital presets.

Kept presets: OSC1 wavetable is a "Wave Source" with at least one 2048-sample keyframe, the first keyframe is finite and
not silent (max |value| >= 1e-6). The first keyframe is scaled to max |value| = 1.
Text: "Genre: ... | Category: ... | Name: ... | Description: ..." (see t2w.vital_io.build_prompt).
Split: torch.utils.data.random_split with Generator().manual_seed(42), 20 % validation.

Outputs (--out):
  dataset.csv  filename, prompt, split, synth parameters (numeric settings), mod_<i>_src/dst/amt (HybridNet targets)
  waves.npy    (N, 2048) float64
Usage:
  python scripts/build_dataset.py --presets data/presets.zip --metadata data/metadata.csv --out data/processed
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2w import vital_io as V

ap = argparse.ArgumentParser()
ap.add_argument("--presets", required=True, help="directory or .zip containing .vital files")
ap.add_argument("--metadata", default=None,
                help="optional CSV with columns name, genre, category and description (or Description_Cleaned)")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--val-ratio", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

meta = {}
if args.metadata:
    m = pd.read_csv(args.metadata, encoding="utf-8-sig").fillna("")
    dcol = "Description_Cleaned" if "Description_Cleaned" in m.columns else "description"
    for _, r in m.iterrows():
        name = str(r.get("name", "")).strip()
        if name:
            meta[name.lower()] = {"genre": str(r.get("genre", "")).strip(), "category": str(r.get("category", "")).strip(),
                          "description": str(r.get(dcol, "")).strip()}

presets = list(V.iter_presets(args.presets))
param_keys = V.scan_param_keys(presets)
rows, waves = [], []
skip = {"unreadable": 0, "not_wave_source": 0, "nonfinite": 0, "silent": 0}
for fname, preset in presets:
    if preset is None:
        skip["unreadable"] += 1
        continue
    w = V.osc1_first_cycle(preset)
    if w is None:
        skip["not_wave_source"] += 1
        continue
    w = w.astype(np.float64)
    if not np.isfinite(w).all():
        skip["nonfinite"] += 1
        continue
    mx = np.abs(w).max()
    if mx < 1e-6:
        skip["silent"] += 1
        continue
    rows.append({"filename": fname, "prompt": V.build_prompt(fname, preset, meta), **V.extract_params_and_mods(preset, param_keys)})
    waves.append(w / mx)

df = pd.DataFrame(rows)
n = len(df)
n_val = int(n * args.val_ratio)
tr, va = random_split(range(n), [n - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
split = np.empty(n, dtype=object)
split[list(tr.indices)] = "train"
split[list(va.indices)] = "val"
df.insert(2, "split", split)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "dataset.csv", index=False)
np.save(out / "waves.npy", np.stack(waves))
print(f"presets {len(presets)} / kept {n} (train {n - n_val}, val {n_val}) / skipped {skip} / synth parameters {len(param_keys)}")
