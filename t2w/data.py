"""Loading the processed dataset written by scripts/build_dataset.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SBERT_MODEL = "all-MiniLM-L6-v2"


class Dataset:
    def __init__(self, data_dir):
        d = Path(data_dir)
        self.dir = d
        self.df = pd.read_csv(d / "dataset.csv")
        self.waves64 = np.load(d / "waves.npy")                         # (N, 2048) float64, max |value| = 1
        self.waves = self.waves64.astype(np.float32)
        self.split = self.df["split"].values
        self.train = np.where(self.split == "train")[0]
        self.val = np.where(self.split == "val")[0]
        self.prompts = self.df["prompt"].fillna("").astype(str).tolist()
        assert len(self.waves) == len(self.df)

    def split_order(self, seed=42):
        """Train / validation indices in the order produced by random_split (the order build_dataset.py used)."""
        import torch
        from torch.utils.data import random_split
        n = len(self.df); n_val = len(self.val)
        tr, va = random_split(range(n), [n - n_val, n_val], generator=torch.Generator().manual_seed(seed))
        assert set(tr.indices) == set(self.train), "split column does not match random_split(seed)"
        return list(tr.indices), list(va.indices)

    def text_emb(self, recompute=False):
        """SBERT embeddings of the dataset prompts, cached in <data>/text_emb.npy.
        The training scripts recompute them right after seeding, as in the paper (loading SBERT consumes torch's RNG)."""
        p = self.dir / "text_emb.npy"
        if recompute or not p.exists():
            np.save(p, encode_text(self.prompts))
        return np.load(p).astype(np.float32)


def encode_text(texts):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL).encode(list(texts), batch_size=64, convert_to_numpy=True,
                                                   show_progress_bar=False).astype(np.float32)


def load_prompts(path):
    return [d["prompt"] if isinstance(d, dict) else d for d in json.load(open(path, encoding="utf-8"))]
