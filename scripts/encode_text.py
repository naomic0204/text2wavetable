"""SBERT (all-MiniLM-L6-v2) embeddings of the dataset prompts (<data>/text_emb.npy) and, optionally, of an evaluation
prompt list. Useful when CLAP and sentence-transformers live in different Python environments.
Usage:
  python scripts/encode_text.py --data data/processed --prompts examples/prompts.json --prompts-out runs/prompts_emb.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2w.data import Dataset, encode_text, load_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--prompts", default=None)
ap.add_argument("--prompts-out", default=None)
args = ap.parse_args()

print("dataset", Dataset(args.data).text_emb().shape)
if args.prompts:
    out = Path(args.prompts_out or Path(args.data) / "prompts_emb.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, encode_text(load_prompts(args.prompts)))
    print("saved", out)
