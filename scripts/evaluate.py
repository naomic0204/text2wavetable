"""Evaluate Map&Steal, HybridNet and DWTS-CVAE on a list of prompts (wavetable only).

Synthesis: wave = alpha @ P (P = row-normalized prototypes) scaled to max |value| = 1, played by a wavetable oscillator at
440 Hz for 3 s at 48 kHz, zero-padded to 4 s.
Metrics:
  CLAP-S  LAION-CLAP 630k text-audio cosine similarity (mean over prompts). Also used to pick the best of N CVAE candidates.
  CLAP-I  laion/larger_clap_general; never used for selection.
  FAD     Frechet distance of CLAP-S embeddings to the reference set (every dataset waveform synthesized the same way),
          in the 32-dimensional PCA space of the reference.
CVAE: z = 0, and N = 1, 5, 10 candidates from z ~ N(0, I) for seeds 0..9 (torch.manual_seed(seed) before the prompt loop;
      per prompt the candidate with the highest CLAP-S is kept). Mean and standard deviation over seeds.
Map&Steal: the NNLS alpha of the training preset whose prompt is closest in SBERT space.
Diversity: seed 0, N = 5, mean pairwise cosine distance of the CLAP-S embeddings of the 5 candidates.
Timing: seconds per prompt for N = 5 (generation + synthesis + CLAP-S scoring) on the current device.
Output: <run>/eval.json
Usage:
  python scripts/evaluate.py --data data/processed --run runs/main --prompts examples/prompts.json
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2w import audio as AU
from t2w.data import Dataset, encode_text, load_prompts
from t2w.models import VitalHybridNet, load_cvae, normalize_prototypes

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--run", default="runs/main")
ap.add_argument("--prompts", default="examples/prompts.json")
ap.add_argument("--prompt-emb", default=None, help="precomputed SBERT embeddings of the prompts (.npy); otherwise computed here")
ap.add_argument("--prototypes", default=None, help="default: <run>/prototypes.npy")
ap.add_argument("--cvae", default=None, help="default: <run>/cvae.pt")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--skip-hybridnet", action="store_true")
args = ap.parse_args()

run = Path(args.run)
ds = Dataset(args.data)
prompts = load_prompts(args.prompts)
C = np.load(args.prompt_emb).astype(np.float32) if args.prompt_emb else encode_text(prompts)
Ct = torch.from_numpy(C)
P = normalize_prototypes(np.load(args.prototypes or run / "prototypes.npy"))
A = np.load(run / "alpha.npy").astype(np.float32)
cvae, ck = load_cvae(args.cvae or run / "cvae.pt")

clap_s, clap_i = AU.ClapS(), AU.ClapI()
t_s, t_i = clap_s.text(prompts), clap_i.text(prompts)
ref_path = Path(args.data) / "ref_clap_s.npy"
if not ref_path.exists():
    ref = [clap_s.audio([AU.render_wave(w) for w in ds.waves[i:i + 64]]) for i in range(0, len(ds.waves), 64)]
    np.save(ref_path, np.vstack(ref))
ref = np.load(ref_path)
R = {"n_prompts": len(prompts), "n_reference": int(len(ref)), "K": int(len(P)), "cvae_epoch": ck.get("epoch")}


def score(audios, emb_s=None):
    e = clap_s.audio(audios) if emb_s is None else emb_s
    ci = (clap_i.audio(audios) * t_i).sum(1)
    return {"clap_s": float((e * t_s).sum(1).mean()), "clap_i": float(ci.mean()), "fad": AU.fad(e, ref),
            "per_prompt_clap_s": (e * t_s).sum(1).tolist(), "per_prompt_clap_i": ci.tolist()}


# Map&Steal
T = ds.text_emb()
Tn = T[ds.train] / (np.linalg.norm(T[ds.train], axis=1, keepdims=True) + 1e-9)
nn_idx = ds.train[np.argmax((C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)) @ Tn.T, axis=1)]
R["Map&Steal"] = {**score([AU.render_alpha(a, P) for a in A[nn_idx]]), "neighbors": ds.df["filename"].values[nn_idx].tolist()}

# HybridNet
if not args.skip_hybridnet:
    meta = pickle.load(open(run / "hybridnet_meta.pkl", "rb"))
    hy = VitalHybridNet(meta["text_dim"], meta["n_params"], meta["K"], meta["n_src"], meta["n_dst"])
    hy.load_state_dict(torch.load(run / "hybridnet.pt", map_location="cpu")); hy.eval()
    with torch.no_grad():
        a_hy = hy(Ct)[1].numpy()
    R["HybridNet"] = score([AU.render_alpha(a, P) for a in a_hy])

# CVAE
R["CVAE z=0"] = score([AU.render_alpha(a, P) for a in cvae.generate_deterministic(Ct).numpy()])


def best_of_n(N, seed):
    torch.manual_seed(seed)
    embs, auds = [], []
    for i in range(len(prompts)):
        cand = [AU.render_alpha(a, P) for a in cvae.generate(Ct[i:i + 1], n_samples=N).numpy()]
        e = clap_s.audio(cand)
        j = int(np.argmax(e @ t_s[i]))
        embs.append(e[j]); auds.append(cand[j])
    return np.stack(embs), auds


for N in [1, 5, 10]:
    per = [score(a, e) for e, a in (best_of_n(N, s) for s in range(args.seeds))]
    R[f"CVAE N={N}"] = {k: [float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))] for k in ["clap_s", "clap_i", "fad"]}
    R[f"CVAE N={N}"]["per_seed"] = {k: [p[k] for p in per] for k in ["clap_s", "clap_i", "fad"]}

torch.manual_seed(0)
div = []
for i in range(len(prompts)):
    e = clap_s.audio([AU.render_alpha(a, P) for a in cvae.generate(Ct[i:i + 1], n_samples=5).numpy()])
    iu = np.triu_indices(5, 1)
    div.append(float((1 - (e @ e.T)[iu]).mean()))
R["diversity_N5_seed0"] = float(np.mean(div))
t0 = time.time()
for i in range(len(prompts)):
    clap_s.audio([AU.render_alpha(a, P) for a in cvae.generate(Ct[i:i + 1], n_samples=5).numpy()])
R["sec_per_prompt_N5"] = (time.time() - t0) / len(prompts)

json.dump(R, open(run / "eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{'method':12s} {'CLAP-S':>16s} {'CLAP-I':>16s} {'FAD':>16s}")
for k in ["Map&Steal", "HybridNet", "CVAE z=0", "CVAE N=1", "CVAE N=5", "CVAE N=10"]:
    if k in R:
        f = lambda v: f"{v[0]:.3f}±{v[1]:.3f}" if isinstance(v, list) else f"{v:.3f}"
        print(f"{k:12s} {f(R[k]['clap_s']):>16s} {f(R[k]['clap_i']):>16s} {f(R[k]['fad']):>16s}")
print(f"diversity {R['diversity_N5_seed0']:.3f}   sec/prompt (N=5) {R['sec_per_prompt_N5']:.3f}")
