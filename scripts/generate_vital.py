"""Text -> Vital preset (.vital) with DWTS-CVAE.

1. SBERT embedding of the text.
2. N candidates alpha ~ p(alpha | z, text), z ~ N(0, I) (or z = 0 with --n 0).
3. With N > 1 the candidate whose 440 Hz test tone has the highest LAION-CLAP 630k (CLAP-S) score is kept
   (requires laion_clap; --no-clap keeps the first candidate).
4. The wavetable alpha @ P replaces OSC1's wavetable of the template preset (templates/vital_init.vital = Vital's
   initial patch); every other parameter is left unchanged.
5. Optionally renders C4 for 2 s with the Vital VST3 through pedalboard (--render, --vst3).
Usage:
  python scripts/generate_vital.py "warm pad with long attack" --run runs/main --out out/warm_pad.vital --n 5
  python scripts/generate_vital.py "dark bass" --run runs/main --out out/bass.vital --render --vst3 "/path/to/Vital.vst3"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2w import audio as AU
from t2w import vital_io as V
from t2w.data import encode_text
from t2w.models import load_cvae, normalize_prototypes

ap = argparse.ArgumentParser()
ap.add_argument("text")
ap.add_argument("--run", default="runs/main", help="folder with prototypes.npy and cvae.pt")
ap.add_argument("--out", default="out/generated.vital")
ap.add_argument("--n", type=int, default=5, help="number of candidates (0 = deterministic z = 0)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-clap", action="store_true", help="do not use CLAP-S to choose among candidates")
ap.add_argument("--template", default=str(ROOT / "templates" / "vital_init.vital"))
ap.add_argument("--text-emb", default=None, help="precomputed SBERT embedding (.npy, shape (384,) or (1, 384))")
ap.add_argument("--render", action="store_true", help="also write <out>.wav rendered with Vital (C4, 2 s, 44.1 kHz)")
ap.add_argument("--vst3", default=None, help="path to Vital.vst3 (needed with --render)")
args = ap.parse_args()

run = Path(args.run)
P = normalize_prototypes(np.load(run / "prototypes.npy"))
model, _ = load_cvae(run / "cvae.pt")
c = np.load(args.text_emb).reshape(1, -1).astype(np.float32) if args.text_emb else encode_text([args.text])
c = torch.from_numpy(c)

info = {"text": args.text, "n": args.n, "seed": args.seed}
if args.n <= 0:
    alpha = model.generate_deterministic(c).numpy()[0]
else:
    torch.manual_seed(args.seed)
    cands = model.generate(c, n_samples=args.n).numpy()
    j = 0
    if args.n > 1 and not args.no_clap:
        clap = AU.ClapS()
        s = clap.audio([AU.render_alpha(a, P) for a in cands]) @ clap.text([args.text])[0]
        j = int(np.argmax(s))
        info["clap_s_candidates"] = s.tolist()
    alpha, info["selected"] = cands[j], j
wave = AU.wave_from_alpha(alpha, P)
info["alpha"] = alpha.tolist()

template = json.load(open(args.template, encoding="utf-8"))
preset = V.preset_with_wave(template, wave, preset_name=args.text[:60])
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(preset, open(out, "w", encoding="utf-8"))
json.dump(info, open(out.with_suffix(".json"), "w", encoding="utf-8"), indent=1)
print("saved", out)

if args.render:
    import soundfile as sf
    from t2w import vital_render as VR
    if not args.vst3:
        sys.exit("--render needs --vst3 /path/to/Vital.vst3")
    plugin = VR.load_vital(args.vst3)
    VR.set_preset(plugin, template)   # start from the initial patch so nothing is carried over
    VR.set_preset(plugin, preset)
    audio = VR.render_note(plugin)
    sf.write(out.with_suffix(".wav"), audio.T, 44100, subtype="FLOAT")
    print("saved", out.with_suffix(".wav"), "peak", float(np.abs(audio).max()))
