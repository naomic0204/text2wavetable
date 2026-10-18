"""Oscillator synthesis used for evaluation, CLAP scorers and FAD."""
import numpy as np
import torch
import torch.nn.functional as F

SR = 48_000          # sample rate for synthesis and CLAP
NOTE_FREQ = 440.0    # A4
DURATION = 3.0       # seconds of tone
TARGET_LEN = SR * 4  # every clip is zero-padded to 4 s before CLAP
PCA_DIM = 32         # FAD is computed on CLAP embeddings projected to 32 PCA dimensions of the reference set


def wave_from_alpha(alpha, P):
    """One cycle: alpha @ P (P = normalized prototypes), scaled to max |value| = 1."""
    w = np.asarray(alpha, dtype=np.float32) @ P
    mx = np.abs(w).max()
    return (w / mx).astype(np.float32) if mx > 1e-6 else w.astype(np.float32)


def oscillator(wave, freq=NOTE_FREQ, duration=DURATION, sr=SR):
    """Non-interpolating wavetable oscillator."""
    n = int(sr * duration)
    ph = np.linspace(0.0, n * freq / sr, n, endpoint=False) % 1.0
    return wave[(ph * len(wave)).astype(np.int32) % len(wave)].astype(np.float32)


def pad4s(x):
    return x[:TARGET_LEN] if len(x) >= TARGET_LEN else np.pad(x, (0, TARGET_LEN - len(x)))


def render_alpha(alpha, P):
    return pad4s(oscillator(wave_from_alpha(alpha, P)))


def render_wave(wave):
    return pad4s(oscillator(np.asarray(wave, dtype=np.float32)))


class ClapS:
    """LAION-CLAP 630k (laion_clap default checkpoint). Used both to select CVAE candidates and as a metric."""

    def __init__(self):
        import laion_clap
        self.m = laion_clap.CLAP_Module(enable_fusion=False)
        self.m.load_ckpt()
        self.m.eval()

    @torch.no_grad()
    def text(self, prompts):
        return F.normalize(self.m.get_text_embedding(list(prompts), use_tensor=True), dim=-1).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def audio(self, audios, batch_size=8):
        out = []
        for i in range(0, len(audios), batch_size):
            x = torch.from_numpy(np.stack(audios[i:i + batch_size]))
            out.append(F.normalize(self.m.get_audio_embedding_from_data(x=x, use_tensor=True), dim=-1).cpu().numpy())
        return np.vstack(out).astype(np.float32)


class ClapI:
    """laion/larger_clap_general (transformers). An independent scorer that is never used for selection."""

    def __init__(self, name="laion/larger_clap_general"):
        from transformers import ClapModel, ClapProcessor
        self.proc = ClapProcessor.from_pretrained(name)
        self.m = ClapModel.from_pretrained(name).eval()

    @torch.no_grad()
    def text(self, prompts):
        e = self.m.get_text_features(**self.proc(text=list(prompts), return_tensors="pt", padding=True))
        return F.normalize(getattr(e, "pooler_output", e), dim=-1).numpy()

    @torch.no_grad()
    def audio(self, audios):
        e = self.m.get_audio_features(**self.proc(audios=list(audios), sampling_rate=SR, return_tensors="pt"))
        return F.normalize(getattr(e, "pooler_output", e), dim=-1).numpy()


def _sqrtm_psd(A):
    w, V = np.linalg.eigh(A)
    return V @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ V.T


def fad(gen, ref, n_pca=PCA_DIM):
    """Frechet distance between generated and reference CLAP-S embeddings in the reference PCA space."""
    mean = ref.mean(axis=0)
    _, _, Vt = np.linalg.svd(ref - mean, full_matrices=False)
    comps = Vt[:n_pca]
    r, g = (ref - mean) @ comps.T, (gen - mean) @ comps.T
    eps = 1e-6
    s_r = np.cov(r, rowvar=False) + eps * np.eye(n_pca)
    s_g = np.cov(g, rowvar=False) if len(g) > 1 else np.eye(n_pca) * eps
    d = r.mean(0) - g.mean(0)
    return float(np.real(d @ d + np.trace(s_g + s_r - 2 * _sqrtm_psd(s_g @ s_r))))
