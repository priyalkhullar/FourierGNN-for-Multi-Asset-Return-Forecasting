"""
graph_builder.py
================
Builds the asset adjacency matrix A for FourierGNN.

Three methods (all become ablations in the paper):
  1. Pearson correlation          — dense, fast
  2. Partial correlation (Lasso)  — sparse, more interpretable
  3. Threshold sparsification     — apply to either of the above

Usage:
    from graph_builder import GraphBuilder
    from data_pipeline import SP500Pipeline, CFG

    pipe    = SP500Pipeline(cfg=CFG)
    splits  = pipe.run()
    builder = GraphBuilder(pipe.returns, pipe.asset_list)

    A_static  = builder.pearson(threshold=0.3)
    A_partial = builder.partial_correlation(alpha=0.1)
    A_dynamic = builder.rolling_pearson(window=60)   # one matrix per day
"""

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")


class GraphBuilder:
    """
    Builds asset adjacency matrices from return data.

    Args:
        returns    : DataFrame (T, N) of raw log-returns (un-normalised)
        asset_list : list of N ticker strings
        sector_map : dict {ticker: sector} for visualisation (optional)
    """

    def __init__(self, returns: pd.DataFrame, asset_list: list, sector_map: dict = None):
        self.returns    = returns.fillna(0.0)
        self.asset_list = asset_list
        self.sector_map = sector_map or {}
        self.N          = len(asset_list)

    # ─────────────────────────────────────────
    # Method 1 — Pearson correlation
    # ─────────────────────────────────────────

    def pearson(self, threshold: float = 0.3) -> torch.Tensor:
        """
        Full-sample Pearson correlation matrix, thresholded for sparsity.

        threshold : zero out edges where |r| < threshold
                    0.0 = fully connected, 0.3 = moderate sparsity (recommended)

        Returns: A (N, N) float32 tensor, diagonal = 0 (no self-loops)
        """
        corr = self.returns[self.asset_list].corr().values.astype(np.float32)
        np.fill_diagonal(corr, 0.0)           # remove self-loops
        corr[np.abs(corr) < threshold] = 0.0  # sparsify
        A = torch.tensor(corr, dtype=torch.float32)
        print(f"[Pearson] Sparsity: {self._sparsity(A):.1%}  |  "
              f"Non-zero edges: {int((A != 0).sum()) // 2}")
        return A

    # ─────────────────────────────────────────
    # Method 2 — Partial correlation (Lasso)
    # ─────────────────────────────────────────

    def partial_correlation(self, alpha: float = None, max_iter: int = 500) -> torch.Tensor:
        """
        Sparse partial correlation via Graphical Lasso.
        Captures DIRECT dependencies (controls for all other assets),
        unlike Pearson which captures total correlation.

        alpha : regularisation strength. None = cross-validated (slower but better).
                Start with alpha=0.1 for speed; use CV for the paper.

        Returns: A (N, N) float32 tensor
        """
        X = self.returns[self.asset_list].values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        print(f"[Partial] Fitting Graphical Lasso (alpha={'CV' if alpha is None else alpha}) ...")
        if alpha is None:
            model = GraphicalLassoCV(max_iter=max_iter, cv=3, n_jobs=-1)
        else:
            model = GraphicalLasso(alpha=alpha, max_iter=max_iter)

        model.fit(X)
        prec = model.precision_                        # precision matrix = inverse covariance
        # Convert precision to partial correlation
        diag  = np.sqrt(np.diag(prec))
        pcorr = -prec / np.outer(diag, diag)
        np.fill_diagonal(pcorr, 0.0)
        A = torch.tensor(pcorr.astype(np.float32), dtype=torch.float32)
        print(f"[Partial] Sparsity: {self._sparsity(A):.1%}  |  "
              f"Non-zero edges: {int((A != 0).sum()) // 2}")
        return A

    # ─────────────────────────────────────────
    # Method 3 — Rolling Pearson (dynamic graph)
    # ─────────────────────────────────────────

    def rolling_pearson(self, window: int = 60, step: int = 20,
                        threshold: float = 0.3) -> dict:
        """
        Computes a Pearson correlation matrix every `step` days using a
        rolling window of `window` days.

        Returns a dict {date: A_tensor} — one adjacency matrix per step.
        Use this to implement dynamic graph updates during training.

        window    : look-back window in trading days (60 = ~3 months)
        step      : recompute every N days (20 = monthly refresh)
        threshold : sparsification threshold
        """
        returns = self.returns[self.asset_list]
        dates   = returns.index
        result  = {}

        indices = range(window, len(dates), step)
        print(f"[Rolling] Computing {len(list(indices))} adjacency matrices "
              f"(window={window}d, step={step}d) ...")

        for i in indices:
            window_data = returns.iloc[i - window : i]
            corr = window_data.corr().values.astype(np.float32)
            np.fill_diagonal(corr, 0.0)
            corr[np.abs(corr) < threshold] = 0.0
            result[dates[i]] = torch.tensor(corr, dtype=torch.float32)

        print(f"[Rolling] Done. {len(result)} matrices, shape each: ({self.N}, {self.N})")
        return result

    # ─────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────

    def _sparsity(self, A: torch.Tensor) -> float:
        """Fraction of zero entries (excluding diagonal)."""
        mask = ~torch.eye(self.N, dtype=torch.bool)
        return float((A[mask] == 0).float().mean())

    def to_edge_index(self, A: torch.Tensor):
        """
        Converts dense adjacency matrix to sparse edge_index format (PyG style).
        Returns: edge_index (2, E),  edge_weight (E,)
        """
        src, dst    = (A != 0).nonzero(as_tuple=True)
        edge_weight = A[src, dst]
        edge_index  = torch.stack([src, dst], dim=0)
        return edge_index, edge_weight

    def summary(self, A: torch.Tensor, name: str = "A"):
        """Prints graph statistics."""
        mask     = ~torch.eye(self.N, dtype=torch.bool)
        nonzero  = int((A[mask] != 0).sum()) // 2
        avg_w    = float(A[mask][A[mask] != 0].abs().mean()) if nonzero > 0 else 0
        deg      = (A != 0).float().sum(dim=1)
        print(f"\n── Graph summary: {name} ───────────────────")
        print(f"  Nodes           : {self.N}")
        print(f"  Edges           : {nonzero}")
        print(f"  Avg degree      : {float(deg.mean()):.1f}")
        print(f"  Max degree      : {int(deg.max())}")
        print(f"  Sparsity        : {self._sparsity(A):.1%}")
        print(f"  Avg edge weight : {avg_w:.4f}")
        print("────────────────────────────────────────────\n")

    # ─────────────────────────────────────────
    # Visualisation
    # ─────────────────────────────────────────

    def plot_adjacency(self, A: torch.Tensor, title: str = "Asset Adjacency Matrix",
                       save_path: str = None):
        """
        Heatmap of A sorted by sector (if sector_map provided).
        This is Figure X in the paper — shows learned graph structure.
        """
        A_np = A.numpy()

        # Sort assets by sector for visual clustering
        if self.sector_map:
            order = sorted(range(self.N),
                           key=lambda i: self.sector_map.get(self.asset_list[i], "ZZZ"))
        else:
            order = list(range(self.N))

        A_sorted = A_np[np.ix_(order, order)]
        labels   = [self.asset_list[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(A_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(title, fontsize=13)

        # Tick labels — only show if N is small enough
        if self.N <= 60:
            ax.set_xticks(range(self.N))
            ax.set_yticks(range(self.N))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_xlabel("Assets (sorted by sector)")
            ax.set_ylabel("Assets (sorted by sector)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()

    def plot_degree_distribution(self, A: torch.Tensor, title: str = "Degree Distribution",
                                  save_path: str = None):
        """Bar chart of node degrees — useful for detecting hub assets."""
        deg    = (A != 0).float().sum(dim=1).numpy()
        assets = self.asset_list

        order  = np.argsort(deg)[::-1][:40]   # top 40 by degree
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar([assets[i] for i in order], deg[order], color="#534AB7")
        ax.set_title(title)
        ax.set_ylabel("Degree (# connections)")
        ax.set_xlabel("Asset")
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ─────────────────────────────────────────────
# MAIN — run standalone to test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import SP500Pipeline, CFG

    CFG["start_date"] = "2018-01-01"

    pipe   = SP500Pipeline(cfg=CFG)
    splits = pipe.run()

    builder = GraphBuilder(pipe.returns, pipe.asset_list, pipe.sector_map)

    # --- Method 1: Pearson
    A_pearson = builder.pearson(threshold=0.3)
    builder.summary(A_pearson, name="Pearson (thr=0.3)")

    # --- Method 2: Partial correlation
    A_partial = builder.partial_correlation(alpha=0.1)
    builder.summary(A_partial, name="Partial correlation")

    # --- Method 3: Rolling (dynamic graph)
    A_rolling = builder.rolling_pearson(window=60, step=20, threshold=0.3)
    print(f"Dynamic graph: {len(A_rolling)} snapshots")

    # --- Visualise
    builder.plot_adjacency(A_pearson,  title="Pearson Adjacency",  save_path="adj_pearson.png")
    builder.plot_adjacency(A_partial,  title="Partial Correlation", save_path="adj_partial.png")
    builder.plot_degree_distribution(A_pearson, save_path="degree_dist.png")

    # --- Edge index (for PyG-style models)
    edge_index, edge_weight = builder.to_edge_index(A_pearson)
    print(f"\nEdge index shape : {edge_index.shape}")
    print(f"Edge weight shape: {edge_weight.shape}")