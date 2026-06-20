"""
evaluation.py
=============
Financial evaluation module for FourierGNN predictions.

Metrics:
  - MSE / MAE (statistical accuracy)
  - Sharpe ratio (annualised)
  - Maximum drawdown
  - Calmar ratio
  - Hit rate (directional accuracy)
  - Regime analysis (low / high VIX)

Usage:
    from evaluation import FinancialEvaluator
    evaluator = FinancialEvaluator(predictions_path="runs/fouriergnn_v1/test_predictions.pt",
                                   returns_df=pipe.returns,
                                   test_split=splits["test"])
    results = evaluator.run_all()
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CORE FINANCIAL METRICS
# ─────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """
    Annualised Sharpe ratio (assuming risk-free rate = 0).
    freq=252 for daily returns.
    """
    if returns.std() < 1e-10:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(freq))


def max_drawdown(cum_returns: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown of a cumulative return series.
    Returns a positive number (e.g. 0.25 = 25% drawdown).
    """
    peak    = np.maximum.accumulate(cum_returns)
    dd      = (cum_returns - peak) / (peak + 1e-10)
    return float(-dd.min())


def calmar_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """Annualised return / max drawdown."""
    ann_return = returns.mean() * freq
    cum        = np.cumprod(1 + returns)
    mdd        = max_drawdown(cum)
    return float(ann_return / mdd) if mdd > 1e-6 else 0.0


def hit_rate(mu: np.ndarray, y: np.ndarray) -> float:
    """
    Directional accuracy: fraction of times predicted sign == actual sign.
    Computed per asset per day, then averaged.
    """
    correct = (np.sign(mu) == np.sign(y))
    return float(correct.mean())


# ─────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────

def long_top_k_returns(
    mu: np.ndarray,
    actual_returns: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """
    Daily-rebalanced long-only portfolio: buy top-k assets by predicted return.

    mu             : (T, N)  predicted returns for H=1 (first horizon step used)
    actual_returns : (T, N)  realised returns on holding day
    k              : number of assets to hold each day

    Returns: portfolio daily returns (T,)

    Note: This is a GROSS return (no transaction costs).
    For the paper, mention this limitation explicitly.
    """
    T, N    = mu.shape
    port_ret = np.zeros(T)
    k       = min(k, N)

    for t in range(T):
        top_k      = np.argsort(mu[t])[-k:]        # indices of top-k predicted
        port_ret[t] = actual_returns[t, top_k].mean()

    return port_ret


def equal_weight_returns(actual_returns: np.ndarray) -> np.ndarray:
    """Benchmark: equal-weight portfolio across all N assets."""
    return actual_returns.mean(axis=1)


# ─────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────

class FinancialEvaluator:
    """
    Loads model predictions and computes full financial evaluation.

    Args:
        predictions_path : path to test_predictions.pt (from trainer)
        returns_df       : raw log-returns DataFrame (T_full, N) from pipeline
        test_split       : normalised test split DataFrame (T_test, N)
        asset_list       : list of N ticker strings
        vix_path         : optional path to VIX CSV for regime analysis
        out_dir          : where to save plots and CSVs
    """

    def __init__(
        self,
        predictions_path : str,
        returns_df       : pd.DataFrame,
        test_split       : pd.DataFrame,
        asset_list       : list,
        out_dir          : str = "./runs/evaluation",
        vix_series       : pd.Series = None,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir    = out_dir
        self.asset_list = asset_list
        self.N          = len(asset_list)

        # Load predictions
        ckpt      = torch.load(predictions_path, map_location="cpu")
        self.mu   = ckpt["mu"].numpy()   # (T_test, N, H)
        self.y    = ckpt["y"].numpy()    # (T_test, N, H)
        self.T    = self.mu.shape[0]

        # Align raw returns to test dates
        # test_split index gives us the target dates
        lookback  = 60   # must match training config
        horizon   = self.mu.shape[2]
        # Target dates: test_split.index[lookback : lookback + T]
        target_dates = test_split.index[lookback : lookback + self.T]
        self.dates   = target_dates

        # Raw (un-normalised) actual returns aligned to same dates/assets
        common_assets = [a for a in asset_list if a in returns_df.columns]
        self.raw_ret  = returns_df.loc[
            returns_df.index.isin(target_dates), common_assets
        ].reindex(index=target_dates, columns=common_assets).fillna(0).values
        # (T, N_common)  — use for portfolio construction

        # Use H=1 slice for daily portfolio (first forecast step)
        self.mu_h1 = self.mu[:, :len(common_assets), 0]   # (T, N)
        self.y_h1  = self.y[:, :len(common_assets), 0]

        self.vix = vix_series   # optional pd.Series indexed by date

    # ── Statistical metrics ───────────────────

    def statistical_metrics(self) -> dict:
        mse = float(np.mean((self.mu - self.y) ** 2))
        mae = float(np.mean(np.abs(self.mu - self.y)))
        hr  = hit_rate(self.mu_h1, self.y_h1)
        print(f"\n── Statistical Metrics ──────────────────────")
        print(f"  MSE       : {mse:.6f}")
        print(f"  MAE       : {mae:.6f}")
        print(f"  Hit rate  : {hr:.3%}  (>50% = directional edge)")
        return {"mse": mse, "mae": mae, "hit_rate": hr}

    # ── Portfolio metrics ─────────────────────

    def portfolio_metrics(self, k_values=(10, 20, 50)) -> dict:
        ew_ret  = equal_weight_returns(self.raw_ret)
        results = {}

        print(f"\n── Portfolio Metrics (gross, no tx costs) ───")
        print(f"  {'Strategy':<20} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'AnnRet':>8}")
        print(f"  {'-'*56}")

        # Equal-weight benchmark
        ew_cum    = np.cumprod(1 + ew_ret)
        ew_sharpe = sharpe_ratio(ew_ret)
        ew_mdd    = max_drawdown(ew_cum)
        ew_calmar = calmar_ratio(ew_ret)
        ew_ann    = ew_ret.mean() * 252
        print(f"  {'EW benchmark':<20} {ew_sharpe:>8.3f} {ew_mdd:>8.3%} "
              f"{ew_calmar:>8.3f} {ew_ann:>8.3%}")
        results["eq_weight"] = dict(sharpe=ew_sharpe, max_dd=ew_mdd,
                                    calmar=ew_calmar, ann_ret=ew_ann)

        # Long-top-k strategies
        for k in k_values:
            if k > self.N:
                continue
            port_ret  = long_top_k_returns(self.mu_h1, self.raw_ret, k=k)
            cum       = np.cumprod(1 + port_ret)
            sr        = sharpe_ratio(port_ret)
            mdd       = max_drawdown(cum)
            calmar    = calmar_ratio(port_ret)
            ann_ret   = port_ret.mean() * 252
            label     = f"Long top-{k}"
            print(f"  {label:<20} {sr:>8.3f} {mdd:>8.3%} {calmar:>8.3f} {ann_ret:>8.3%}")
            results[f"top_{k}"] = dict(sharpe=sr, max_dd=mdd,
                                       calmar=calmar, ann_ret=ann_ret)
        return results

    # ── Regime analysis ───────────────────────

    def regime_analysis(self, k: int = 20) -> dict:
        """
        Split test period into low-VIX and high-VIX regimes.
        If VIX data unavailable, uses cross-asset volatility as proxy.
        """
        print(f"\n── Regime Analysis ──────────────────────────")

        # Build regime mask
        if self.vix is not None:
            vix_aligned = self.vix.reindex(self.dates, method="ffill").fillna(20)
            vix_vals    = vix_aligned.values
        else:
            # Proxy: rolling 20-day cross-asset std of raw returns
            vix_vals = pd.DataFrame(self.raw_ret).rolling(20).std().mean(axis=1).values
            vix_vals = np.nan_to_num(vix_vals, nan=np.nanmedian(vix_vals))

        low_mask  = vix_vals <= np.percentile(vix_vals, 33)
        high_mask = vix_vals >= np.percentile(vix_vals, 67)

        results = {}
        print(f"  {'Regime':<15} {'Days':>6} {'Sharpe':>8} {'HitRate':>9} {'MSE':>10}")
        print(f"  {'-'*50}")

        for label, mask in [("Low vol",  low_mask), ("High vol", high_mask)]:
            if mask.sum() < 10:
                continue
            mu_r  = self.mu_h1[mask]
            y_r   = self.y_h1[mask]
            ret_r = self.raw_ret[mask]

            port  = long_top_k_returns(mu_r, ret_r, k=min(k, self.N))
            sr    = sharpe_ratio(port)
            hr    = hit_rate(mu_r, y_r)
            mse_r = float(np.mean((mu_r - y_r) ** 2))
            days  = int(mask.sum())

            print(f"  {label:<15} {days:>6} {sr:>8.3f} {hr:>9.3%} {mse_r:>10.6f}")
            results[label] = dict(sharpe=sr, hit_rate=hr, mse=mse_r, days=days)

        return results

    # ── Plots ─────────────────────────────────

    def plot_cumulative_returns(self, k_values=(10, 20), save: bool = True):
        """
        Plots cumulative return curves for each strategy.
        This is Figure X in the paper.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        colors  = ["#534AB7", "#0F6E56", "#993C1D", "#BA7517"]

        ew_ret = equal_weight_returns(self.raw_ret)
        ax.plot(self.dates[:len(ew_ret)],
                np.cumprod(1 + ew_ret) - 1,
                label="EW benchmark", color="gray", linewidth=1.2, linestyle="--")

        for i, k in enumerate(k_values):
            if k > self.N:
                continue
            port_ret = long_top_k_returns(self.mu_h1, self.raw_ret, k=k)
            ax.plot(self.dates[:len(port_ret)],
                    np.cumprod(1 + port_ret) - 1,
                    label=f"Long top-{k}",
                    color=colors[i % len(colors)], linewidth=1.5)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Cumulative Returns — FourierGNN Portfolio (Test Period)")
        ax.set_ylabel("Cumulative return")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            path = os.path.join(self.out_dir, "cumulative_returns.png")
            plt.savefig(path, dpi=150)
            print(f"[Plot] Saved cumulative_returns.png")
        plt.show()

    def plot_prediction_scatter(self, n_assets: int = 5, save: bool = True):
        """
        Scatter of predicted vs actual returns for a few assets.
        Useful sanity check — should show weak but positive correlation.
        """
        fig, axes = plt.subplots(1, n_assets, figsize=(4 * n_assets, 4))
        for i in range(min(n_assets, self.N)):
            axes[i].scatter(self.mu_h1[:, i], self.y_h1[:, i],
                            alpha=0.3, s=8, color="#534AB7")
            axes[i].axhline(0, color="gray", linewidth=0.5)
            axes[i].axvline(0, color="gray", linewidth=0.5)
            axes[i].set_title(self.asset_list[i], fontsize=10)
            axes[i].set_xlabel("Predicted")
            if i == 0:
                axes[i].set_ylabel("Actual")

            # Add correlation annotation
            corr = np.corrcoef(self.mu_h1[:, i], self.y_h1[:, i])[0, 1]
            axes[i].annotate(f"r={corr:.3f}", xy=(0.05, 0.92),
                             xycoords="axes fraction", fontsize=9)

        plt.suptitle("Predicted vs Actual Returns (H=1)", y=1.01)
        plt.tight_layout()
        if save:
            path = os.path.join(self.out_dir, "prediction_scatter.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[Plot] Saved prediction_scatter.png")
        plt.show()

    def plot_regime_sharpe(self, regime_results: dict, save: bool = True):
        """Bar chart comparing Sharpe across regimes — goes in the paper."""
        if not regime_results:
            return
        labels = list(regime_results.keys())
        sharpes = [regime_results[l]["sharpe"] for l in labels]
        colors  = ["#0F6E56" if s > 0 else "#993C1D" for s in sharpes]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, sharpes, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Sharpe Ratio by Volatility Regime")
        ax.set_ylabel("Annualised Sharpe")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save:
            path = os.path.join(self.out_dir, "regime_sharpe.png")
            plt.savefig(path, dpi=150)
            print(f"[Plot] Saved regime_sharpe.png")
        plt.show()

    # ── Run all ───────────────────────────────

    def run_all(self, k_values=(10, 20, 50)) -> dict:
        print("\n" + "="*55)
        print("  Financial Evaluation — FourierGNN")
        print("="*55)

        stat    = self.statistical_metrics()
        port    = self.portfolio_metrics(k_values=k_values)
        regime  = self.regime_analysis(k=20)

        self.plot_cumulative_returns(k_values=list(k_values)[:2])
        self.plot_prediction_scatter()
        self.plot_regime_sharpe(regime)

        # Save summary CSV
        rows = []
        for k, v in port.items():
            rows.append({"strategy": k, **v})
        pd.DataFrame(rows).to_csv(
            os.path.join(self.out_dir, "portfolio_metrics.csv"), index=False)

        print(f"\n[Done] All outputs saved to {self.out_dir}")
        return {"statistical": stat, "portfolio": port, "regime": regime}


# ─────────────────────────────────────────────
# BASELINE COMPARISON TABLE
# ─────────────────────────────────────────────

def compare_models(results_dict: dict):
    """
    Print paper-ready comparison table given a dict of
    {model_name: {"statistical": ..., "portfolio": ...}} results.

    Example:
        compare_models({
            "FourierGNN" : fouriergnn_results,
            "PatchTST"   : patchtst_results,
            "VAR"        : var_results,
        })
    """
    print(f"\n{'='*70}")
    print(f"  Model Comparison Table")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'MSE':>8} {'MAE':>8} {'HitRate':>9} "
          f"{'Sharpe20':>10} {'MaxDD':>8} {'Calmar':>8}")
    print(f"  {'-'*66}")

    for name, res in results_dict.items():
        stat = res.get("statistical", {})
        port = res.get("portfolio", {}).get("top_20", {})
        print(f"  {name:<18} "
              f"{stat.get('mse', 0):>8.5f} "
              f"{stat.get('mae', 0):>8.5f} "
              f"{stat.get('hit_rate', 0):>9.3%} "
              f"{port.get('sharpe', 0):>10.3f} "
              f"{port.get('max_dd', 0):>8.3%} "
              f"{port.get('calmar', 0):>8.3f}")


# ─────────────────────────────────────────────
# MAIN — end-to-end test with synthetic data
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("Running evaluation smoke test with synthetic data ...")
    np.random.seed(42)
    T, N, H = 300, 20, 5
    dates   = pd.date_range("2023-01-01", periods=T, freq="B")
    assets  = [f"A{i}" for i in range(N)]

    # Simulate realistic predictions (weak signal + noise)
    y_true  = np.random.randn(T, N, H) * 0.01
    mu_pred = y_true * 0.15 + np.random.randn(T, N, H) * 0.01  # noisy signal

    # Save fake predictions
    tmpdir = tempfile.mkdtemp()
    pred_path = os.path.join(tmpdir, "test_predictions.pt")
    torch.save({"mu": torch.tensor(mu_pred, dtype=torch.float32),
                "y":  torch.tensor(y_true,  dtype=torch.float32)}, pred_path)

    # Fake returns DataFrame
    returns_df = pd.DataFrame(
        np.random.randn(T + 100, N) * 0.01,
        index=pd.date_range("2022-06-01", periods=T + 100, freq="B"),
        columns=assets,
    )

    # Fake test split (normalised) — needs lookback rows before target dates
    test_split = pd.DataFrame(
        np.random.randn(T + 70, N) * 1.0,
        index=pd.date_range("2022-09-01", periods=T + 70, freq="B"),
        columns=assets,
    )

    evaluator = FinancialEvaluator(
        predictions_path = pred_path,
        returns_df       = returns_df,
        test_split       = test_split,
        asset_list       = assets,
        out_dir          = os.path.join(tmpdir, "eval"),
    )

    results = evaluator.run_all(k_values=(5, 10, 20))
    print("\nSmoke test PASSED")