"""Reading and writing Vital (.vital) presets.

A .vital file is JSON. The first oscillator's wavetable is stored in
settings.wavetables[0].groups[0].components[0]; for a "Wave Source" component each keyframe holds one
2048-sample cycle as base64-encoded float32 (keyframes[i].wave_data).
"""
import base64
import copy
import json
import zipfile
from pathlib import Path

import numpy as np

N_SAMPLES = 2048
MAX_MODS = 8
PARAM_SCAN_LIMIT = 600
PARAM_BLACKLIST = ["wavetables", "sample", "points", "powers", "modulation_", "modulations",
                   "name", "author", "comments", "style", "midi", "mpe", "macro", "finger", "order"]


def iter_presets(source):
    """Yield (relative_name, preset_dict or None) for every .vital file in a directory or a .zip archive.
    Order: archive order for a .zip, sorted path order for a directory. The train/validation split depends on it."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            for name in [n for n in z.namelist() if n.endswith(".vital")]:
                try:
                    with z.open(name) as f:
                        yield name, json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    yield name, None
    else:
        for p in sorted(source.rglob("*.vital")):
            try:
                yield p.relative_to(source.parent).as_posix(), json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield p.relative_to(source.parent).as_posix(), None


def osc1_component(preset):
    try:
        return preset["settings"]["wavetables"][0]["groups"][0]["components"][0]
    except (KeyError, IndexError, TypeError):
        return None


def osc1_first_cycle(preset):
    """First 2048-sample cycle of OSC1 if its wavetable is a Wave Source, else None (Audio File / Line Sources are skipped)."""
    comp = osc1_component(preset)
    if not isinstance(comp, dict) or comp.get("type", "") != "Wave Source":
        return None
    for kf in comp.get("keyframes", []):
        if not isinstance(kf, dict) or not kf.get("wave_data"):
            continue
        try:
            frame = np.frombuffer(base64.b64decode(kf["wave_data"]), dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if len(frame) == N_SAMPLES:
            return frame.copy()
    return None


def build_prompt(filename, preset, metadata):
    """Text condition "Genre: ... | Category: ... | Name: ... | Description: ..." (empty fields are omitted).
    metadata maps the lower-cased preset name -> dict(genre, category, description). The description is the listing
    description followed by the preset's own comments. Presets without metadata use their folder name as genre."""
    name = Path(filename).name.replace(".vital", "")
    comments = str(preset.get("comments", "") if isinstance(preset, dict) else "").strip()
    ext = metadata.get(name.lower())
    if ext:
        genre, category, desc = ext["genre"], ext["category"], f"{ext['description']} {comments}".strip()
    else:
        genre, category, desc = Path(filename).parent.name, "", comments
    parts = [f"{k}: {v}" for k, v in [("Genre", genre), ("Category", category), ("Name", name), ("Description", desc)] if v]
    return " | ".join(parts)


def scan_param_keys(presets):
    """Numeric top-level settings found in the first PARAM_SCAN_LIMIT presets (synth parameters used by HybridNet)."""
    keys = set()
    for _, preset in presets[:PARAM_SCAN_LIMIT]:
        if not isinstance(preset, dict):
            continue
        for k, v in preset.get("settings", {}).items():
            if not any(b in k for b in PARAM_BLACKLIST) and isinstance(v, (int, float, bool)):
                keys.add(k)
    return sorted(keys)


def extract_params_and_mods(preset, param_keys):
    s = preset.get("settings", {})
    row = {k: s.get(k, 0.0) for k in param_keys}
    active = []
    for i, mod in enumerate(s.get("modulations", [])):
        if isinstance(mod, dict) and mod.get("source") and mod.get("destination"):
            active.append((mod["source"], mod["destination"], float(s.get(f"modulation_{i + 1}_amount", 0.0))))
    for i in range(MAX_MODS):
        src, dst, amt = active[i] if i < len(active) else ("None", "None", 0.0)
        row[f"mod_{i}_src"], row[f"mod_{i}_dst"], row[f"mod_{i}_amt"] = src, dst, amt
    return row


def wave_to_b64(wave):
    wave = np.asarray(wave, dtype=np.float32)
    assert wave.shape == (N_SAMPLES,), wave.shape
    return base64.b64encode(wave.tobytes()).decode("ascii")


def preset_with_wave(template, wave, preset_name=""):
    """Copy of the template preset whose OSC1 wavetable (first keyframe) is replaced by `wave`. Nothing else is changed."""
    preset = copy.deepcopy(template)
    preset["settings"]["wavetables"][0]["groups"][0]["components"][0]["keyframes"][0]["wave_data"] = wave_to_b64(wave)
    if preset_name:
        preset["preset_name"] = preset_name
    return preset
