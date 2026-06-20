"""
fouriergnn.py
=============
FourierGNN adapted for multi-asset equity return forecasting.

Architecture (3 layers):
  1. FourierMixingLayer  — FFT across time, learnable complex weights mix
                           frequency components across all N assets globally
  2. GNNLayer            — GraphSAGE mean-aggregation over asset adjacency A
  3. PredictionHead      — per-asset MLP → (mean, log_var) forecasts

Key differences from original FourierGNN (traffic/energy):
  - Per-asset rolling z-score input (handled in data_pipeline.py)
  - Dynamic adjacency support (swap A each forward pass)
  - Dual output head: mean + variance (aleatoric uncertainty)
  - Frequency band masking (learnable — ablation §5f)

Usage:
    from fouriergnn import FourierGNN, FourierGNNConfig
    cfg   = FourierGNNConfig(n_assets=50, lookback=60, horizon=5)
    model = FourierGNN(cfg)
    X     = torch.randn(32, 50, 60)   # (batch, N, L)
    A     = torch.randn(50, 50)       # adjacency
    mu, log_var = model(X, A)         # (32, 50, 5) each
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# CONFIG  (single object passed everywhere)
# ─────────────────────────────────────────────

@dataclass
class FourierGNNConfig:
    # Data dimensions
    n_assets  : int = 50     # N — number of assets
    lookback  : int = 60     # L — input sequence length
    horizon   : int = 5      # H — forecast horizon

    # Architecture
    d_model   : int = 64     # hidden dimension throughout
    n_fourier : int = 2      # number of stacked Fourier mixing layers
    n_gnn     : int = 2      # number of stacked GNN layers
    dropout   : float = 0.1

    # Fourier
    freq_mask : bool = True  # learnable frequency band mask (ablation §5f)

    # GNN
    gnn_type  : str = "sage" # "sage" (mean) | "gat" (attention)
    gat_heads : int = 4      # used only when gnn_type="gat"

    # Output
    predict_variance : bool = True   # output (mu, log_var) for uncertainty


# ─────────────────────────────────────────────
# LAYER 1 — Fourier Mixing
# ─────────────────────────────────────────────

class FourierMixingLayer(nn.Module):
    """
    Global frequency-domain mixing layer.

    Steps:
      1. FFT along time axis  →  X_freq : (B, N, L//2+1) complex
      2. Learnable complex weight matrix W mixes across N assets
         for each frequency bin simultaneously
      3. Optional frequency band mask (learnable sigmoid gate)
      4. Inverse FFT back to time domain

    This replaces spatial graph convolution with O(N * L log L) global mixing.
    All N assets communicate at every frequency bin — captures cross-asset
    spectral dependencies without explicit edge traversal.
    """

    def __init__(self, cfg: FourierGNNConfig):
        super().__init__()
        self.cfg   = cfg
        n_freq     = cfg.lookback // 2 + 1   # number of FFT frequency bins

        # Complex weight: shape (n_freq, N, N) — one NxN mixer per frequency bin
        # Initialised small to keep early training stable
        self.W_real = nn.Parameter(torch.randn(n_freq, cfg.n_assets, cfg.n_assets) * 0.02)
        self.W_imag = nn.Parameter(torch.randn(n_freq, cfg.n_assets, cfg.n_assets) * 0.02)

        # Learnable frequency band mask — gates which frequencies to use
        if cfg.freq_mask:
            self.freq_gate = nn.Parameter(torch.ones(n_freq))  # init: all freqs open

        self.norm    = nn.LayerNorm(cfg.lookback)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, L)
        returns : (B, N, L)
        """
        residual = x

        # FFT along time axis
        x_freq = torch.fft.rfft(x, dim=-1)          # (B, N, n_freq) complex

        # Apply learnable frequency mask (ablation: cfg.freq_mask=False skips this)
        if self.cfg.freq_mask:
            gate   = torch.sigmoid(self.freq_gate)   # (n_freq,) in [0,1]
            x_freq = x_freq * gate.unsqueeze(0).unsqueeze(0)

        # Cross-asset mixing: for each freq bin f, mix N assets with W[f]
        # x_freq: (B, N, n_freq) → permute to (B, n_freq, N) for matmul
        x_freq = x_freq.permute(0, 2, 1)             # (B, n_freq, N)

        W_complex = torch.complex(self.W_real, self.W_imag)   # (n_freq, N, N)

        # Batched matmul: (B, n_freq, N) × (n_freq, N, N) → (B, n_freq, N)
        x_mixed = torch.einsum('bfn,fnm->bfm', x_freq, W_complex)

        x_mixed = x_mixed.permute(0, 2, 1)           # (B, N, n_freq)

        # Inverse FFT back to time domain
        x_out = torch.fft.irfft(x_mixed, n=self.cfg.lookback, dim=-1)  # (B, N, L)

        x_out = self.dropout(x_out)
        return self.norm(x_out + residual)            # residual connection


# ─────────────────────────────────────────────
# LAYER 2 — GNN  (GraphSAGE or GAT)
# ─────────────────────────────────────────────

class GNNLayer(nn.Module):
    """
    Single GNN layer operating on asset node representations.

    Input  x : (B, N, d_model)   — node features
    Input  A : (N, N)            — adjacency (weighted, possibly asymmetric)
    Output   : (B, N, d_model)

    GraphSAGE (gnn_type="sage"):
        h_i = W_self * x_i  +  W_neigh * mean_{j in N(i)} x_j
        Simple, fast, no extra parameters per edge.

    GAT (gnn_type="gat"):
        h_i = sum_{j in N(i)} alpha_ij * W * x_j
        alpha_ij learned attention weights — richer but slower.
        Useful ablation: does attention over neighbours help?
    """

    def __init__(self, cfg: FourierGNNConfig):
        super().__init__()
        d = cfg.d_model

        if cfg.gnn_type == "sage":
            self.W_self  = nn.Linear(d, d, bias=False)
            self.W_neigh = nn.Linear(d, d, bias=False)

        elif cfg.gnn_type == "gat":
            H = cfg.gat_heads
            assert d % H == 0, "d_model must be divisible by gat_heads"
            self.W_val   = nn.Linear(d, d, bias=False)
            self.att_src = nn.Linear(d // H, 1, bias=False)
            self.att_dst = nn.Linear(d // H, 1, bias=False)
            self.H       = H

        self.norm    = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act     = nn.GELU()
        self.cfg     = cfg

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        residual = x                              # (B, N, d)
        B, N, d  = x.shape

        if self.cfg.gnn_type == "sage":
            # Normalise A rows so aggregation is a weighted mean
            deg      = A.sum(dim=-1, keepdim=True).clamp(min=1)
            A_norm   = A / deg                    # (N, N)

            # Neighbour aggregation: (B, N, d)
            neigh    = torch.einsum('nm,bmd->bnd', A_norm, x)
            out      = self.act(self.W_self(x) + self.W_neigh(neigh))

        elif self.cfg.gnn_type == "gat":
            H, d_h   = self.H, d // self.H
            val      = self.W_val(x).view(B, N, H, d_h)   # (B, N, H, d_h)

            # Attention scores
            e_src    = self.att_src(val)                   # (B, N, H, 1)
            e_dst    = self.att_dst(val)                   # (B, N, H, 1)
            e        = e_src.squeeze(-1).unsqueeze(2) + \
                       e_dst.squeeze(-1).unsqueeze(1)      # (B, N, N, H)
            e        = F.leaky_relu(e, 0.2)

            # Mask with adjacency (zero out non-edges)
            mask     = (A == 0).unsqueeze(0).unsqueeze(-1) # (1, N, N, 1)
            e        = e.masked_fill(mask, -1e9)
            alpha    = F.softmax(e, dim=2)                 # (B, N, N, H)

            # Aggregate
            out      = torch.einsum('bnjh,bjhd->bihd', alpha,
                                    val).reshape(B, N, d)

        out = self.dropout(out)
        return self.norm(out + residual)


# ─────────────────────────────────────────────
# LAYER 3 — Prediction Head
# ─────────────────────────────────────────────

class PredictionHead(nn.Module):
    """
    Per-asset MLP that maps d_model features to H-step forecasts.
    Shared weights across assets (parameter efficient).

    With predict_variance=True:
        Outputs mu and log_var — enables NLL loss and uncertainty quantification.
        The trading signal uses mu; log_var gives confidence intervals.
    """

    def __init__(self, cfg: FourierGNNConfig):
        super().__init__()
        d = cfg.d_model
        H = cfg.horizon

        self.mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
            nn.GELU(),
        )
        self.head_mu      = nn.Linear(d, H)
        self.head_log_var = nn.Linear(d, H) if cfg.predict_variance else None

    def forward(self, x: torch.Tensor):
        """x : (B, N, d_model)  →  mu: (B, N, H),  log_var: (B, N, H) or None"""
        h   = self.mlp(x)
        mu  = self.head_mu(h)
        lv  = self.head_log_var(h) if self.head_log_var is not None else None
        return mu, lv


# ─────────────────────────────────────────────
# TEMPORAL PROJECTION  (L → d_model)
# ─────────────────────────────────────────────

class TemporalProjection(nn.Module):
    """
    Projects the L-length time series per asset into d_model features.
    Runs before the Fourier mixing to lift input into the model dimension.
    """
    def __init__(self, cfg: FourierGNNConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.lookback, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, L) → (B, N, d_model)"""
        return self.proj(x)


# ─────────────────────────────────────────────
# FULL MODEL — FourierGNN
# ─────────────────────────────────────────────

class FourierGNN(nn.Module):
    """
    Full FourierGNN model for multi-asset return forecasting.

    Forward pass:
        X : (B, N, L)  — normalised input returns
        A : (N, N)     — asset adjacency matrix (static or dynamic)

    Returns:
        mu      : (B, N, H) — predicted returns
        log_var : (B, N, H) — log variance (None if predict_variance=False)

    Parameter count (default cfg, N=50):
        ~200K parameters — fast to train on CPU
    """

    def __init__(self, cfg: FourierGNNConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection: (B, N, L) → (B, N, d_model)
        self.input_proj = TemporalProjection(cfg)

        # Fourier mixing operates on raw time series (B, N, L)
        self.fourier_layers = nn.ModuleList([
            FourierMixingLayer(cfg) for _ in range(cfg.n_fourier)
        ])

        # After Fourier mixing, project to d_model for GNN
        self.fourier_proj = nn.Linear(cfg.lookback, cfg.d_model)

        # GNN layers operate on (B, N, d_model)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(cfg) for _ in range(cfg.n_gnn)
        ])

        # Fuse Fourier path + GNN path
        self.fusion = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )

        # Prediction head
        self.head = PredictionHead(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """
        X : (B, N, L)
        A : (N, N)
        """
        # ── Path 1: Fourier mixing (stays in time domain shape)
        x_f = X
        for layer in self.fourier_layers:
            x_f = layer(x_f)                          # (B, N, L)
        x_f = F.gelu(self.fourier_proj(x_f))          # (B, N, d_model)

        # ── Path 2: Node features from raw input → GNN
        x_g = self.input_proj(X)                      # (B, N, d_model)
        for layer in self.gnn_layers:
            x_g = layer(x_g, A)                       # (B, N, d_model)

        # ── Fuse both paths
        x = self.fusion(torch.cat([x_f, x_g], dim=-1))  # (B, N, d_model)

        # ── Predict
        mu, log_var = self.head(x)                    # (B, N, H) each
        return mu, log_var

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────

def mse_loss(mu, y):
    """Standard MSE — use for ablation without variance."""
    return F.mse_loss(mu, y)


def gaussian_nll_loss(mu, log_var, y, eps=1e-6):
    """
    Gaussian negative log-likelihood loss.
    Trains both mean and variance simultaneously.
    NLL = 0.5 * (log_var + (y - mu)^2 / var)

    Use this as the primary loss when predict_variance=True.
    Better than MSE: model learns to be uncertain where it should be.
    """
    var  = torch.exp(log_var) + eps
    loss = 0.5 * (log_var + (y - mu).pow(2) / var)
    return loss.mean()


def combined_loss(mu, log_var, y, lambda_mse=0.5):
    """
    Weighted combination of NLL and MSE.
    MSE term anchors the mean; NLL term calibrates uncertainty.
    lambda_mse=0.5 is a reasonable default — tune as hyperparameter.
    """
    nll  = gaussian_nll_loss(mu, log_var, y)
    mse  = mse_loss(mu, y)
    return nll + lambda_mse * mse, nll.item(), mse.item()


# ─────────────────────────────────────────────
# MAIN — smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    cfg   = FourierGNNConfig(n_assets=50, lookback=60, horizon=5, d_model=64)
    model = FourierGNN(cfg)
    print(f"Parameters: {model.count_parameters():,}")

    # Dummy forward pass
    B     = 32
    X     = torch.randn(B, cfg.n_assets, cfg.lookback)
    A     = torch.rand(cfg.n_assets, cfg.n_assets)
    A     = (A + A.T) / 2                             # make symmetric
    A.fill_diagonal_(0)

    mu, lv = model(X, A)
    print(f"Output shapes — mu: {tuple(mu.shape)}, log_var: {tuple(lv.shape)}")

    # Loss
    y    = torch.randn(B, cfg.n_assets, cfg.horizon)
    loss, nll, mse = combined_loss(mu, lv, y)
    print(f"Loss: {loss.item():.4f}  (NLL: {nll:.4f}, MSE: {mse:.4f})")

    # One backward pass — confirm gradients flow
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Gradient norm (mean): {sum(grad_norms)/len(grad_norms):.4f}  ✓")

    # Ablation variants
    print("\n── Ablation variants ─────────────────────────")
    for freq_mask, gnn_type, pv in [
        (True,  "sage", True),   # full model
        (False, "sage", True),   # no freq mask
        (True,  "gat",  True),   # GAT instead of SAGE
        (True,  "sage", False),  # no variance output
    ]:
        c = FourierGNNConfig(n_assets=50, lookback=60, horizon=5,
                             freq_mask=freq_mask, gnn_type=gnn_type,
                             predict_variance=pv)
        m = FourierGNN(c)
        print(f"  freq_mask={str(freq_mask):<5}  gnn={gnn_type:<4}  "
              f"variance={str(pv):<5}  params={m.count_parameters():>8,}")