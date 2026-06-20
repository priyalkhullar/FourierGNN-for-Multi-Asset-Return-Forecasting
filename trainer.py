"""
trainer.py
==========
Training loop for FourierGNN — connects data_pipeline, graph_builder, fouriergnn.

Features:
  - Train / val loop with early stopping
  - Cosine LR schedule with linear warmup
  - Gradient clipping (essential for complex-valued weights)
  - Checkpoint saving (best val loss)
  - Dynamic graph support (swap A every K epochs)
  - CSV loss log for paper plots
  - Ablation runner (train multiple configs in one call)

Usage:
    python trainer.py
    # or import and call programmatically:
    from trainer import Trainer, TrainerConfig
"""

import os, json, time, csv
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_pipeline  import SP500Pipeline, SP500Dataset, CFG as DATA_CFG
from graph_builder  import GraphBuilder
from fouriergnn     import FourierGNN, FourierGNNConfig, combined_loss, mse_loss


# ─────────────────────────────────────────────
# TRAINER CONFIG
# ─────────────────────────────────────────────

@dataclass
class TrainerConfig:
    # Training
    epochs          : int   = 100
    batch_size      : int   = 32
    lr              : float = 1e-3
    weight_decay    : float = 1e-4
    grad_clip       : float = 1.0     # clip gradient norm — important for FFT layers
    warmup_epochs   : int   = 5

    # Early stopping
    patience        : int   = 15      # stop if val loss doesn't improve for N epochs
    min_delta       : float = 1e-5

    # Graph
    graph_method    : str   = "pearson"   # "pearson" | "partial"
    pearson_thr     : float = 0.3
    partial_alpha   : float = 0.1
    dynamic_graph   : bool  = False       # refresh A every K epochs
    dynamic_k       : int   = 10

    # Loss
    use_nll         : bool  = True        # NLL + MSE combined loss
    lambda_mse      : float = 0.5

    # Output
    run_name        : str   = "run_001"
    out_dir         : str   = "./runs"
    save_every      : int   = 10          # save checkpoint every N epochs

    # Misc
    num_workers     : int   = 0           # 0 for Windows compatibility
    seed            : int   = 42


# ─────────────────────────────────────────────
# LR SCHEDULE  — cosine with linear warmup
# ─────────────────────────────────────────────

def build_scheduler(optimizer, cfg: TrainerConfig):
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(cfg.warmup_epochs, 1)
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model_cfg  : FourierGNNConfig,
        train_cfg  : TrainerConfig,
        data_splits: dict,            # from SP500Pipeline.run()
        graph_builder: GraphBuilder,
    ):
        torch.manual_seed(train_cfg.seed)
        self.mcfg    = model_cfg
        self.tcfg    = train_cfg
        self.splits  = data_splits
        self.builder = graph_builder
        self.device  = torch.device("cpu")   # CPU for local dev

        # Output directory
        self.run_dir = os.path.join(train_cfg.out_dir, train_cfg.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Build model
        self.model = FourierGNN(model_cfg).to(self.device)
        print(f"[Trainer] Model parameters: {self.model.count_parameters():,}")

        # Optimizer + scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = build_scheduler(self.optimizer, train_cfg)

        # Build adjacency matrix
        self.A = self._build_adjacency().to(self.device)

        # DataLoaders
        self.loaders = self._build_loaders()

        # Logging
        self.history = {"epoch": [], "train_loss": [], "val_loss": [],
                        "train_mse": [], "val_mse": [], "lr": []}
        self.best_val_loss  = float("inf")
        self.patience_count = 0

        # Save configs
        self._save_configs()

    # ── Build adjacency ───────────────────────

    def _build_adjacency(self) -> torch.Tensor:
        m = self.tcfg.graph_method
        if m == "pearson":
            A = self.builder.pearson(threshold=self.tcfg.pearson_thr)
        elif m == "partial":
            A = self.builder.partial_correlation(alpha=self.tcfg.partial_alpha)
        else:
            raise ValueError(f"Unknown graph_method: {m}")
        return A

    # ── DataLoaders ───────────────────────────

    def _build_loaders(self) -> dict:
        cfg = self.tcfg
        mc  = self.mcfg
        datasets = {
            s: SP500Dataset(self.splits[s], mc.lookback, mc.horizon)
            for s in ["train", "val", "test"]
        }
        loaders = {
            "train": DataLoader(datasets["train"], batch_size=cfg.batch_size,
                                shuffle=True,  num_workers=cfg.num_workers),
            "val":   DataLoader(datasets["val"],   batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers),
            "test":  DataLoader(datasets["test"],  batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers),
        }
        sizes = {s: len(datasets[s]) for s in datasets}
        print(f"[Trainer] Samples — train: {sizes['train']} | "
              f"val: {sizes['val']} | test: {sizes['test']}")
        return loaders

    # ── One epoch ─────────────────────────────

    def _run_epoch(self, split: str) -> tuple:
        """Returns (avg_loss, avg_mse)."""
        is_train = split == "train"
        self.model.train(is_train)
        loader = self.loaders[split]

        total_loss, total_mse, n = 0.0, 0.0, 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for X, y, _ in loader:
                X, y = X.to(self.device), y.to(self.device)

                mu, lv = self.model(X, self.A)

                if self.tcfg.use_nll and lv is not None:
                    loss, nll_val, mse_val = combined_loss(
                        mu, lv, y, self.tcfg.lambda_mse)
                else:
                    loss     = mse_loss(mu, y)
                    mse_val  = loss.item()

                mse_raw = mse_loss(mu, y).item()

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.tcfg.grad_clip)
                    self.optimizer.step()

                total_loss += loss.item() * X.size(0)
                total_mse  += mse_raw    * X.size(0)
                n          += X.size(0)

        return total_loss / n, total_mse / n

    # ── Main train loop ───────────────────────

    def train(self) -> dict:
        cfg  = self.tcfg
        best_path = os.path.join(self.run_dir, "best_model.pt")

        print(f"\n{'='*55}")
        print(f"  Training: {cfg.run_name}")
        print(f"  Epochs: {cfg.epochs}  |  Patience: {cfg.patience}")
        print(f"  Graph: {cfg.graph_method}  |  NLL loss: {cfg.use_nll}")
        print(f"{'='*55}")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # Optional: refresh dynamic graph
            if cfg.dynamic_graph and epoch % cfg.dynamic_k == 0:
                self.A = self._build_adjacency().to(self.device)

            train_loss, train_mse = self._run_epoch("train")
            val_loss,   val_mse   = self._run_epoch("val")
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mse"].append(train_mse)
            self.history["val_mse"].append(val_mse)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            print(f"Epoch {epoch:>3}/{cfg.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"mse={val_mse:.4f}  lr={current_lr:.2e}  "
                  f"[{elapsed:.1f}s]")

            # Checkpoint: best model
            improved = val_loss < self.best_val_loss - cfg.min_delta
            if improved:
                self.best_val_loss  = val_loss
                self.patience_count = 0
                torch.save({
                    "epoch"      : epoch,
                    "model_state": self.model.state_dict(),
                    "val_loss"   : val_loss,
                    "val_mse"    : val_mse,
                }, best_path)
                print(f"  ↑ Best model saved (val_loss={val_loss:.5f})")
            else:
                self.patience_count += 1
                if self.patience_count >= cfg.patience:
                    print(f"\n[Early stop] No improvement for {cfg.patience} epochs.")
                    break

            # Periodic checkpoint
            if epoch % cfg.save_every == 0:
                ckpt_path = os.path.join(self.run_dir, f"epoch_{epoch:03d}.pt")
                torch.save(self.model.state_dict(), ckpt_path)

        self._save_history()
        self._plot_curves()
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.5f}")
        return self.history

    # ── Load best & evaluate test ─────────────

    def evaluate_test(self) -> dict:
        """Load best checkpoint and evaluate on test set."""
        best_path = os.path.join(self.run_dir, "best_model.pt")
        ckpt = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"\n[Eval] Loaded best model from epoch {ckpt['epoch']} "
              f"(val_loss={ckpt['val_loss']:.5f})")

        self.model.eval()
        all_mu, all_y = [], []

        with torch.no_grad():
            for X, y, _ in self.loaders["test"]:
                X = X.to(self.device)
                mu, _ = self.model(X, self.A)
                all_mu.append(mu.cpu())
                all_y.append(y)

        mu_all = torch.cat(all_mu, dim=0)   # (T_test, N, H)
        y_all  = torch.cat(all_y,  dim=0)

        mse  = mse_loss(mu_all, y_all).item()
        mae  = (mu_all - y_all).abs().mean().item()

        results = {"test_mse": mse, "test_mae": mae}
        print(f"[Eval] Test MSE: {mse:.5f}  |  Test MAE: {mae:.5f}")

        # Save predictions for financial evaluation module
        torch.save({"mu": mu_all, "y": y_all}, 
                   os.path.join(self.run_dir, "test_predictions.pt"))

        return results

    # ── Helpers ───────────────────────────────

    def _save_configs(self):
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump({
                "model": asdict(self.mcfg),
                "trainer": asdict(self.tcfg),
            }, f, indent=2)

    def _save_history(self):
        csv_path = os.path.join(self.run_dir, "history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history.keys())
            writer.writeheader()
            for i in range(len(self.history["epoch"])):
                writer.writerow({k: self.history[k][i] for k in self.history})

    def _plot_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history["epoch"], self.history["train_loss"], label="Train loss")
        axes[0].plot(self.history["epoch"], self.history["val_loss"],   label="Val loss")
        axes[0].set_title("Loss curves")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history["epoch"], self.history["val_mse"], color="#993C1D")
        axes[1].set_title("Val MSE")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "training_curves.png"), dpi=150)
        plt.close()
        print(f"[Plot] Saved training_curves.png")


# ─────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────

def run_ablations(data_splits, graph_builder, base_mcfg, base_tcfg):
    """
    Trains all ablation variants and saves a comparison table.
    Each ablation differs by exactly one config change from the full model.
    """
    ablations = [
        ("full_model",      dict()),
        ("no_freq_mask",    dict(freq_mask=False)),
        ("gat_gnn",         dict(gnn_type="gat")),
        ("no_variance",     dict(predict_variance=False)),
        ("lookback_20",     dict(lookback=20)),
        ("lookback_120",    dict(lookback=120)),
    ]

    results = []
    for name, overrides in ablations:
        print(f"\n{'#'*55}")
        print(f"  Ablation: {name}")
        print(f"{'#'*55}")

        # Build model config with overrides
        mcfg_dict = asdict(base_mcfg)
        mcfg_dict.update(overrides)

        # lookback change needs data re-split at dataset level — handled in SP500Dataset
        mcfg = FourierGNNConfig(**mcfg_dict)

        tcfg = TrainerConfig(**{**asdict(base_tcfg), "run_name": f"ablation_{name}"})

        trainer = Trainer(mcfg, tcfg, data_splits, graph_builder)
        trainer.train()
        test_res = trainer.evaluate_test()
        results.append({"name": name, **test_res})

    # Print comparison table
    print(f"\n{'='*50}")
    print(f"  Ablation Results")
    print(f"{'='*50}")
    print(f"  {'Name':<25} {'MSE':>10} {'MAE':>10}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['name']:<25} {r['test_mse']:>10.5f} {r['test_mae']:>10.5f}")

    # Save table
    out_dir = base_tcfg.out_dir
    with open(os.path.join(out_dir, "ablation_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved ablation_results.csv to {out_dir}")
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Data
    DATA_CFG["start_date"] = "2018-01-01"
    pipe   = SP500Pipeline(cfg=DATA_CFG)
    splits = pipe.run()

    # ── 2. Graph
    builder = GraphBuilder(pipe.returns, pipe.asset_list, pipe.sector_map)

    # ── 3. Configs
    model_cfg = FourierGNNConfig(
        n_assets  = len(pipe.asset_list),
        lookback  = DATA_CFG["lookback"],
        horizon   = DATA_CFG["horizon"],
        d_model   = 64,
        n_fourier = 2,
        n_gnn     = 2,
        dropout   = 0.1,
        freq_mask = True,
        gnn_type  = "sage",
    )

    train_cfg = TrainerConfig(
        epochs       = 50,      # increase to 100 for paper runs
        batch_size   = 32,
        lr           = 1e-3,
        patience     = 15,
        graph_method = "pearson",
        use_nll      = True,
        run_name     = "fouriergnn_v1",
        out_dir      = "./runs",
    )

    # ── 4. Train
    trainer = Trainer(model_cfg, train_cfg, splits, builder)
    history = trainer.train()

    # ── 5. Evaluate
    test_results = trainer.evaluate_test()

    print("\nDone. Outputs saved to ./runs/fouriergnn_v1/")
    print("  best_model.pt       — best checkpoint")
    print("  test_predictions.pt — (mu, y) tensors for financial evaluation")
    print("  training_curves.png — loss plots")
    print("  history.csv         — full training log")