"""Models: prototype dictionary (VAE), DWTS-CVAE, and the HybridNet baseline (called MLP in the paper)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SAMPLES = 2048


# ─── Prototype dictionary: a VAE whose decoder mixes K learnable prototypes ─────────────────────────
class WavetableEncoder(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=4, padding=6), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.fc_mu = nn.Linear(128 * 64, z_dim)
        self.fc_logvar = nn.Linear(128 * 64, z_dim)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class WavetableDecoder(nn.Module):
    def __init__(self, K, z_dim=32):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(K, N_SAMPLES) * 0.1)
        self.fc = nn.Sequential(nn.Linear(z_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, K))

    def forward(self, z):
        alpha = F.softmax(self.fc(z), dim=-1)
        return torch.einsum("bk,kn->bn", alpha, self.prototypes), alpha


class WavetableVAE(nn.Module):
    def __init__(self, K, z_dim=32):
        super().__init__()
        self.encoder = WavetableEncoder(z_dim)
        self.decoder = WavetableDecoder(K, z_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        recon, alpha = self.decoder(z)
        return recon, mu, logvar, alpha


def dictionary_vae_loss(recon, x, mu, logvar, beta=0.5):
    mse = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl, mse.item(), kl.item()


def normalize_prototypes(P, dtype=np.float32):
    """Each prototype scaled to max |value| = 1 (used for NNLS, CVAE training and synthesis)."""
    P = np.asarray(P, dtype=dtype)
    return P / np.maximum(np.abs(P).max(axis=1, keepdims=True), 1e-6)


# ─── DWTS-CVAE ─────────────────────────────────────────────────────────────────────────────────────
class DWTSCVAEEncoder(nn.Module):
    def __init__(self, K, cond_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K + cond_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1))
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, alpha, c):
        h = self.net(torch.cat([alpha, c], dim=-1))
        return self.fc_mu(h), self.fc_log_var(h)


class DWTSCVAEDecoder(nn.Module):
    def __init__(self, K, cond_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, K))

    def forward(self, z, c):
        return F.softmax(self.net(torch.cat([z, c], dim=-1)), dim=-1)


class DWTSCVAENet(nn.Module):
    """Encoder q(z | alpha, text), decoder p(alpha | z, text); the wavetable is alpha @ prototypes."""

    def __init__(self, K, cond_dim=384, latent_dim=32):
        super().__init__()
        self.encoder = DWTSCVAEEncoder(K, cond_dim, latent_dim)
        self.decoder = DWTSCVAEDecoder(K, cond_dim, latent_dim)
        self.latent_dim = latent_dim
        self.K = K

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, alpha, c):
        mu, log_var = self.encoder(alpha, c)
        return self.decoder(self.reparameterize(mu, log_var), c), mu, log_var

    @torch.no_grad()
    def generate(self, c, n_samples=1):
        """z ~ N(0, I). c: (1, cond_dim) -> (n_samples, K)."""
        self.eval()
        if n_samples > 1:
            c = c.repeat(n_samples, 1)
        return self.decoder(torch.randn(len(c), self.latent_dim, device=c.device), c)

    @torch.no_grad()
    def generate_deterministic(self, c):
        """z = 0."""
        self.eval()
        return self.decoder(torch.zeros(len(c), self.latent_dim, device=c.device), c)


def dwts_cvae_loss(alpha_pred, mu, log_var, prototypes, wt_true, beta=1.0, free_bits=0.05):
    """Waveform MSE of alpha_pred @ prototypes + beta * KL with free bits per latent dimension."""
    wt_loss = F.mse_loss(torch.mm(alpha_pred, prototypes), wt_true)
    kl = torch.clamp(-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()), min=free_bits).mean()
    return wt_loss + beta * kl, wt_loss.item(), kl.item()


def load_cvae(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = DWTSCVAENet(K=ck["K"], cond_dim=ck.get("cond_dim", 384), latent_dim=ck["latent_dim"])
    m.load_state_dict(ck["model_state"])
    return m.eval(), ck


# ─── HybridNet baseline: text -> synth parameters, alpha and modulations ───────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class VitalHybridNet(nn.Module):
    def __init__(self, text_dim, n_params, K, n_src, n_dst, n_mods=8, hidden=1024):
        super().__init__()
        self.n_mods = n_mods
        self.embed_in = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3))
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.param_head = nn.Linear(hidden, n_params)
        self.wt_head = nn.Linear(hidden, K)
        self.mod_src_heads = nn.ModuleList([nn.Linear(hidden, n_src) for _ in range(n_mods)])
        self.mod_dst_heads = nn.ModuleList([nn.Linear(hidden, n_dst) for _ in range(n_mods)])
        self.mod_amt_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_mods)])

    def forward(self, text_emb):
        h = self.res2(self.res1(self.embed_in(text_emb)))
        return (torch.sigmoid(self.param_head(h)), F.softmax(self.wt_head(h), dim=-1),
                [m(h) for m in self.mod_src_heads], [m(h) for m in self.mod_dst_heads],
                [torch.tanh(m(h)).squeeze(-1) for m in self.mod_amt_heads])


HYBRID_LAMBDA = {"param": 1.0, "wt": 0.3, "mod": 0.05}


def hybrid_loss(pred, batch, label_smoothing=0.1):
    params_pred, wt_pred, src_logits, dst_logits, amt_preds = pred
    _, params_tgt, wt_tgt, wt_mask, mod_src, mod_dst, mod_amt, mod_active = batch
    param_loss = F.mse_loss(params_pred, params_tgt)
    m = wt_mask.bool()
    wt_loss = F.mse_loss(wt_pred[m], wt_tgt[m]) if m.any() else torch.tensor(0.0, device=wt_pred.device)
    mod_loss = torch.tensor(0.0, device=params_pred.device)
    n_mods = len(src_logits)
    for i in range(n_mods):
        a = mod_active[:, i].bool()
        a_loss = F.mse_loss(amt_preds[i][a], mod_amt[:, i][a]) if a.any() else torch.tensor(0.0, device=params_pred.device)
        mod_loss = mod_loss + F.cross_entropy(src_logits[i], mod_src[:, i], label_smoothing=label_smoothing) \
            + F.cross_entropy(dst_logits[i], mod_dst[:, i], label_smoothing=label_smoothing) + 0.5 * a_loss
    mod_loss = mod_loss / n_mods
    total = HYBRID_LAMBDA["param"] * param_loss + HYBRID_LAMBDA["wt"] * wt_loss + HYBRID_LAMBDA["mod"] * mod_loss
    return total, param_loss.item(), wt_loss.item(), mod_loss.item()
