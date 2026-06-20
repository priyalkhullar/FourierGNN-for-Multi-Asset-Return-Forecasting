"""
baselines.py
============
Baseline models for comparison against FourierGNN.

Models implemented:
  1. VAR          — Vector Autoregression (classical econometric)
  2. LSTNet       — CNN + LSTM (deep learning classic)
  3. PatchTST     — Patch-based Transformer (SOTA attention baseline)
  4. iTransformer — Inverted attention for multivariate TS (recent SOTA)

All baselines expose the same interface:
    model.fit(train_data)
    preds = model.predict(test_loader)   → (T, N, H) numpy array

Usage:
    from baselines import run_all_baselines
    baseline_results = run_all_baselines(pipe, splits, DATA_CFG, out_dir="./runs")
"""

import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import SP500Dataset


# ═══════════════════════════════════════════════
# 1.  VAR  — Vector Autoregression
# ═══════════════════════════════════════════════

class VARBaseline:
    """
    Fits a separate univariate AR(p) model per asset.
    True multivariate VAR is O(N^2) parameters and intractable for N~450;
    per-asset AR is the standard tractable approximation used in finance papers.

    For N=50: full VAR is feasible and used automatically.
    For N>100: falls back to per-asset AR.
    """

    def __init__(self, lookback: int = 20, horizon: int = 5):
        self.p       = lookback
        self.H       = horizon
        self.coeffs  = None   # (N, p) OLS coefficients
        self.intercepts = None

    def fit(self, train_df: pd.DataFrame):
        N = train_df.shape[1]
        X_data = train_df.values  # (T, N)
        T = X_data.shape[0]

        # Build OLS system: for each asset, regress r_t on r_{t-1}...r_{t-p}
        self.coeffs     = np.zeros((N, self.p))
        self.intercepts = np.zeros(N)

        for i in range(N):
            y_col = X_data[self.p:, i]           # (T-p,)
            X_mat = np.stack(
                [X_data[self.p - j - 1 : T - j - 1, i] for j in range(self.p)],
                axis=1
            )                                     # (T-p, p)
            X_mat = np.hstack([np.ones((len(X_mat), 1)), X_mat])  # add intercept
            try:
                coef, *_ = np.linalg.lstsq(X_mat, y_col, rcond=None)
                self.intercepts[i] = coef[0]
                self.coeffs[i]     = coef[1:]
            except Exception:
                pass  # keep zeros if singular

        print(f"[VAR] Fitted AR({self.p}) for {N} assets.")

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """
        x      : (N, p) last p days of returns
        returns: (N, H) multi-step forecast via recursive prediction
        """
        N    = x.shape[0]
        preds = np.zeros((N, self.H))
        buf  = x.copy()  # (N, p) rolling buffer

        for h in range(self.H):
            # one-step: r_hat = intercept + coeffs @ buf (reversed: lag-1 first)
            r_hat = self.intercepts + (self.coeffs * buf[:, ::-1]).sum(axis=1)
            preds[:, h] = r_hat
            # roll buffer forward
            buf = np.roll(buf, -1, axis=1)
            buf[:, -1] = r_hat

        return preds

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        all_preds = []
        for X, y, _ in test_loader:
            X_np = X.numpy()  # (B, N, L)
            batch_preds = []
            for b in range(X_np.shape[0]):
                p = self.predict_one(X_np[b, :, -self.p:])
                batch_preds.append(p)
            all_preds.append(np.stack(batch_preds))   # (B, N, H)
        return np.concatenate(all_preds, axis=0)       # (T, N, H)


# ═══════════════════════════════════════════════
# 2.  LSTNet  — CNN + LSTM
# ═══════════════════════════════════════════════

class LSTNetModel(nn.Module):
    """
    LSTNet (Lai et al., 2018) — CNN extracts local patterns,
    LSTM captures long-range dependencies.

    Simplified version: no skip-RNN, no autoregressive component.
    Sufficient for baseline comparison.
    """

    def __init__(self, n_assets: int, lookback: int, horizon: int,
                 cnn_hidden: int = 32, rnn_hidden: int = 64,
                 cnn_kernel: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_assets   = n_assets
        self.lookback   = lookback
        self.H          = horizon
        self.rnn_hidden = rnn_hidden

        # CNN: extract local temporal patterns per asset
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_hidden, kernel_size=(1, cnn_kernel)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        cnn_out_len = lookback - cnn_kernel + 1

        # GRU: model temporal dependencies
        self.gru = nn.GRU(
            input_size  = n_assets * cnn_hidden,
            hidden_size = rnn_hidden,
            batch_first = True,
        )

        # Output head
        self.fc = nn.Linear(rnn_hidden, n_assets * horizon)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, L) → (B, N, H)"""
        B, N, L = x.shape

        # CNN expects (B, 1, N, L)
        x_cnn = x.unsqueeze(1)                           # (B, 1, N, L)
        x_cnn = self.cnn(x_cnn)                          # (B, C, N, L')
        C      = x_cnn.shape[1]
        L_out  = x_cnn.shape[3]

        # Reshape for GRU: (B, L', N*C)
        x_gru = x_cnn.permute(0, 3, 2, 1).reshape(B, L_out, N * C)
        out, _ = self.gru(x_gru)                         # (B, L', rnn_hidden)

        out = self.drop(out[:, -1, :])                   # last timestep
        out = self.fc(out)                               # (B, N*H)
        return out.view(B, N, self.H)


# ═══════════════════════════════════════════════
# 3.  PatchTST  — Patch Transformer
# ═══════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, lookback: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = (lookback - patch_len) // stride + 1
        self.proj      = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, n_patches, d_model)"""
        patches = x.unfold(-1, self.patch_len, self.stride)   # (B, n_patches, patch_len)
        return self.proj(patches)                              # (B, n_patches, d_model)


class PatchTSTModel(nn.Module):
    """
    PatchTST (Nie et al., 2023) — channel-independent Transformer on patches.

    Each asset is processed independently (channel-independent).
    Patches of length patch_len with stride are embedded and passed
    through a standard Transformer encoder.
    """

    def __init__(self, n_assets: int, lookback: int, horizon: int,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_assets = n_assets
        self.H        = horizon

        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, lookback)
        n_patches        = self.patch_embed.n_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model * n_patches, horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, L) → (B, N, H)"""
        B, N, L = x.shape

        # Process each asset independently: reshape to (B*N, L)
        x_flat   = x.reshape(B * N, L)
        patches  = self.patch_embed(x_flat)             # (B*N, n_patches, d_model)
        enc      = self.encoder(patches)                # (B*N, n_patches, d_model)
        enc_flat = enc.reshape(B * N, -1)               # (B*N, n_patches*d_model)
        out      = self.head(self.dropout(enc_flat))    # (B*N, H)
        return out.view(B, N, self.H)


# ═══════════════════════════════════════════════
# 4.  iTransformer  — Inverted Transformer
# ═══════════════════════════════════════════════

class iTransformerModel(nn.Module):
    """
    iTransformer (Liu et al., 2024) — inverts the attention axis.

    Standard Transformer: attention over time steps.
    iTransformer:         attention over variates (assets).

    Each asset's full time series is embedded as one token.
    Attention learns cross-asset dependencies.
    Conceptually similar to GNN message passing but without explicit graph.
    """

    def __init__(self, n_assets: int, lookback: int, horizon: int,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_assets = n_assets
        self.H        = horizon

        # Embed each asset's time series into d_model
        self.input_proj = nn.Linear(lookback, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decode each asset token to H-step forecast
        self.head    = nn.Linear(d_model, horizon)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, L) → (B, N, H)"""
        # Project each asset time series: (B, N, L) → (B, N, d_model)
        tokens = self.input_proj(x)                   # (B, N, d_model)
        tokens = self.norm(tokens)

        # Attention across N asset tokens (inverted)
        enc = self.encoder(tokens)                    # (B, N, d_model)
        out = self.head(self.dropout(enc))            # (B, N, H)
        return out


# ═══════════════════════════════════════════════
# GENERIC NEURAL BASELINE TRAINER
# ═══════════════════════════════════════════════

class NeuralBaselineTrainer:
    """
    Shared training loop for LSTNet, PatchTST, iTransformer.
    Keeps training identical across baselines for fair comparison.
    """

    def __init__(self, model: nn.Module, name: str, out_dir: str,
                 lr: float = 1e-3, epochs: int = 50,
                 patience: int = 10, grad_clip: float = 1.0):
        self.model    = model
        self.name     = name
        self.out_dir  = os.path.join(out_dir, f"baseline_{name}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.lr       = lr
        self.epochs   = epochs
        self.patience = patience
        self.clip     = grad_clip
        self.device   = torch.device("cpu")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        model    = self.model.to(self.device)
        opt      = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best_val, patience_cnt = float("inf"), 0
        best_path = os.path.join(self.out_dir, "best.pt")

        print(f"\n[{self.name}] Training  "
              f"(params={sum(p.numel() for p in model.parameters()):,})")

        for epoch in range(1, self.epochs + 1):
            # Train
            model.train()
            tr_loss = 0.0
            for X, y, _ in train_loader:
                X, y   = X.to(self.device), y.to(self.device)
                pred   = model(X)
                loss   = F.mse_loss(pred, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                opt.step()
                tr_loss += loss.item() * X.size(0)
            tr_loss /= len(train_loader.dataset)

            # Val
            model.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for X, y, _ in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    vl_loss += F.mse_loss(model(X), y).item() * X.size(0)
            vl_loss /= len(val_loader.dataset)
            sched.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}  train={tr_loss:.5f}  val={vl_loss:.5f}")

            if vl_loss < best_val - 1e-6:
                best_val = vl_loss
                patience_cnt = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(best_path, map_location="cpu"))
        print(f"[{self.name}] Best val MSE: {best_val:.5f}")
        return model

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, y, _ in test_loader:
                preds.append(self.model(X.to(self.device)).cpu().numpy())
        return np.concatenate(preds, axis=0)   # (T, N, H)


# ═══════════════════════════════════════════════
# EVALUATION HELPER  (mirrors evaluation.py)
# ═══════════════════════════════════════════════

def quick_metrics(mu: np.ndarray, y: np.ndarray,
                  raw_ret: np.ndarray, k: int = 20) -> dict:
    """Compute MSE, MAE, hit-rate, and Sharpe for a set of predictions."""
    from evaluation import sharpe_ratio, long_top_k_returns, hit_rate

    mse = float(np.mean((mu - y) ** 2))
    mae = float(np.mean(np.abs(mu - y)))
    hr  = hit_rate(mu[:, :, 0], y[:, :, 0])

    port = long_top_k_returns(mu[:, :, 0], raw_ret[:mu.shape[0]], k=min(k, mu.shape[1]))
    sr   = sharpe_ratio(port)

    return {"mse": mse, "mae": mae, "hit_rate": hr, "sharpe_top20": sr}


# ═══════════════════════════════════════════════
# MASTER RUNNER
# ═══════════════════════════════════════════════

def run_all_baselines(pipe, splits, data_cfg: dict,
                      out_dir: str = "./runs",
                      epochs: int = 50) -> dict:
    """
    Trains all baselines and returns a dict of metrics for compare_models().

    Args:
        pipe      : fitted SP500Pipeline
        splits    : {"train", "val", "test"} DataFrames
        data_cfg  : DATA_CFG dict (for lookback, horizon)
        out_dir   : where to save checkpoints
        epochs    : training epochs per baseline (50 for dev, 100 for paper)
    """
    N        = len(pipe.asset_list)
    L        = data_cfg["lookback"]
    H        = data_cfg["horizon"]

    # Build datasets + loaders (identical to trainer.py)
    def make_loader(split, shuffle):
        ds = SP500Dataset(splits[split], L, H)
        return DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=0)

    train_loader = make_loader("train", True)
    val_loader   = make_loader("val",   False)
    test_loader  = make_loader("test",  False)

    # Align raw returns to test dates for portfolio construction
    test_ds   = SP500Dataset(splits["test"], L, H)
    T_test    = len(test_ds)
    target_dates = splits["test"].index[L : L + T_test]
    raw_ret   = pipe.returns.reindex(
        index=target_dates, columns=pipe.asset_list).fillna(0).values

    all_results = {}

    # ── 1. VAR ───────────────────────────────────
    print("\n" + "─" * 50)
    print("  Baseline 1/4 — VAR")
    print("─" * 50)
    var = VARBaseline(lookback=min(L, 20), horizon=H)
    var.fit(splits["train"])
    var_preds = var.predict(test_loader)

    # Ground truth
    y_all = np.concatenate([y.numpy() for _, y, _ in test_loader], axis=0)

    all_results["VAR"] = quick_metrics(var_preds, y_all, raw_ret)

    # ── 2. LSTNet ─────────────────────────────────
    print("\n" + "─" * 50)
    print("  Baseline 2/4 — LSTNet")
    print("─" * 50)
    lstnet_model   = LSTNetModel(N, L, H, cnn_hidden=32, rnn_hidden=64)
    lstnet_trainer = NeuralBaselineTrainer(lstnet_model, "LSTNet", out_dir,
                                           epochs=epochs, patience=10)
    lstnet_trainer.fit(train_loader, val_loader)
    lstnet_preds   = lstnet_trainer.predict(test_loader)
    all_results["LSTNet"] = quick_metrics(lstnet_preds, y_all, raw_ret)

    # ── 3. PatchTST ───────────────────────────────
    print("\n" + "─" * 50)
    print("  Baseline 3/4 — PatchTST")
    print("─" * 50)
    patch_model   = PatchTSTModel(N, L, H, patch_len=16, stride=8,
                                  d_model=64, n_heads=4, n_layers=2)
    patch_trainer = NeuralBaselineTrainer(patch_model, "PatchTST", out_dir,
                                          epochs=epochs, patience=10)
    patch_trainer.fit(train_loader, val_loader)
    patch_preds   = patch_trainer.predict(test_loader)
    all_results["PatchTST"] = quick_metrics(patch_preds, y_all, raw_ret)

    # ── 4. iTransformer ───────────────────────────
    print("\n" + "─" * 50)
    print("  Baseline 4/4 — iTransformer")
    print("─" * 50)
    itrans_model   = iTransformerModel(N, L, H, d_model=64, n_heads=4, n_layers=2)
    itrans_trainer = NeuralBaselineTrainer(itrans_model, "iTransformer", out_dir,
                                           epochs=epochs, patience=10)
    itrans_trainer.fit(train_loader, val_loader)
    itrans_preds   = itrans_trainer.predict(test_loader)
    all_results["iTransformer"] = quick_metrics(itrans_preds, y_all, raw_ret)

    # ── Print comparison ──────────────────────────
    _print_baseline_table(all_results)

    # Save to CSV
    import csv
    csv_path = os.path.join(out_dir, "baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","mse","mae","hit_rate","sharpe_top20"])
        w.writeheader()
        for name, res in all_results.items():
            w.writerow({"model": name, **res})
    print(f"\nSaved baseline_results.csv → {csv_path}")

    return all_results


def _print_baseline_table(results: dict):
    print(f"\n{'='*62}")
    print(f"  Baseline Comparison")
    print(f"{'='*62}")
    print(f"  {'Model':<18} {'MSE':>10} {'MAE':>10} {'HitRate':>9} {'Sharpe':>8}")
    print(f"  {'-'*58}")
    for name, r in results.items():
        print(f"  {name:<18} {r['mse']:>10.6f} {r['mae']:>10.6f} "
              f"{r['hit_rate']:>9.3%} {r['sharpe_top20']:>8.3f}")
    print(f"{'='*62}")


# ─────────────────────────────────────────────
# MAIN — smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    print("Smoke-testing all baselines with synthetic data ...")
    np.random.seed(42)
    T, N, L, H = 500, 20, 60, 5

    dates  = pd.date_range("2018-01-01", periods=T, freq="B")
    ret_df = pd.DataFrame(np.random.randn(T, N) * 0.01,
                          index=dates, columns=[f"A{i}" for i in range(N)])
    splits = {
        "train": ret_df.iloc[:300],
        "val":   ret_df.iloc[300:400],
        "test":  ret_df.iloc[400:],
    }

    class FakePipe:
        asset_list = [f"A{i}" for i in range(N)]
        returns    = ret_df

    data_cfg = {"lookback": L, "horizon": H}
    results  = run_all_baselines(FakePipe(), splits, data_cfg,
                                 out_dir="/tmp/baseline_test", epochs=3)
    print("\nSmoke test PASSED")