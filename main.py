"""
main.py
=======
End-to-end pipeline for FourierGNN multi-asset return forecasting.

Run modes:
    python main.py                   # full run: data → graph → train → baselines → eval
    python main.py --mode data       # data download & cache only
    python main.py --mode train      # train FourierGNN only
    python main.py --mode baselines  # train baselines only
    python main.py --mode eval       # evaluation + comparison table
    python main.py --ablations       # train all 6 ablation variants
"""

import os
import argparse
import torch
import numpy as np

from data_pipeline import SP500Pipeline, CFG as DATA_CFG
from graph_builder import GraphBuilder
from fouriergnn    import FourierGNNConfig
from trainer       import Trainer, TrainerConfig, run_ablations
from baselines     import run_all_baselines, _print_baseline_table
from evaluation    import FinancialEvaluator, compare_models


# ═══════════════════════════════════════════════
# EXPERIMENT CONFIG  ← only edit this section
# ═══════════════════════════════════════════════

# ── Data ──────────────────────────────────────
DATA_CFG["start_date"] = "2018-01-01"   # use "2014-01-01" for full paper run
DATA_CFG["end_date"]   = "2024-12-31"
DATA_CFG["lookback"]   = 60
DATA_CFG["horizon"]    = 5

# ── Model ─────────────────────────────────────
MODEL_CFG = FourierGNNConfig(
    n_assets         = None,      # set automatically after data loads
    lookback         = DATA_CFG["lookback"],
    horizon          = DATA_CFG["horizon"],
    d_model          = 64,
    n_fourier        = 2,
    n_gnn            = 2,
    dropout          = 0.1,
    freq_mask        = True,
    gnn_type         = "sage",    # "sage" | "gat"
    predict_variance = True,
)

# ── Training ──────────────────────────────────
TRAIN_CFG = TrainerConfig(
    epochs        = 100,
    batch_size    = 32,
    lr            = 1e-3,
    weight_decay  = 1e-4,
    patience      = 15,
    grad_clip     = 1.0,
    warmup_epochs = 5,
    graph_method  = "pearson",    # "pearson" | "partial"
    pearson_thr   = 0.3,
    partial_alpha = 0.1,
    dynamic_graph = False,
    use_nll       = True,
    lambda_mse    = 0.5,
    run_name      = "fouriergnn_v1",
    out_dir       = "./runs",
    save_every    = 10,
    num_workers   = 0,
)

# ── Baselines ─────────────────────────────────
BASELINE_EPOCHS = 50     # increase to 100 for paper run

# ── Evaluation ────────────────────────────────
EVAL_K_VALUES = (10, 20, 50)


# ═══════════════════════════════════════════════
# STAGES
# ═══════════════════════════════════════════════

def stage_data():
    print("\n" + "▓"*55)
    print("  STAGE 1 — Data Pipeline")
    print("▓"*55)
    pipe   = SP500Pipeline(cfg=DATA_CFG)
    splits = pipe.run()
    pipe.summary()
    return pipe, splits


def stage_graph(pipe):
    print("\n" + "▓"*55)
    print("  STAGE 2 — Graph Construction")
    print("▓"*55)
    builder = GraphBuilder(pipe.returns, pipe.asset_list, pipe.sector_map)
    method  = TRAIN_CFG.graph_method
    A = (builder.pearson(threshold=TRAIN_CFG.pearson_thr)
         if method == "pearson"
         else builder.partial_correlation(alpha=TRAIN_CFG.partial_alpha))
    builder.summary(A, name=method)
    return builder, A


def stage_train(pipe, splits, builder):
    print("\n" + "▓"*55)
    print("  STAGE 3 — FourierGNN Training")
    print("▓"*55)
    MODEL_CFG.n_assets = len(pipe.asset_list)
    trainer  = Trainer(MODEL_CFG, TRAIN_CFG, splits, builder)
    history  = trainer.train()
    test_res = trainer.evaluate_test()
    print(f"\n  ✓ FourierGNN  →  MSE: {test_res['test_mse']:.5f} | "
          f"MAE: {test_res['test_mae']:.5f}")
    return trainer, test_res


def stage_baselines(pipe, splits):
    print("\n" + "▓"*55)
    print("  STAGE 4 — Baseline Models")
    print("▓"*55)
    results = run_all_baselines(
        pipe, splits, DATA_CFG,
        out_dir = TRAIN_CFG.out_dir,
        epochs  = BASELINE_EPOCHS,
    )
    return results


def stage_eval(pipe, splits, fouriergnn_results=None, baseline_results=None):
    print("\n" + "▓"*55)
    print("  STAGE 5 — Financial Evaluation & Comparison")
    print("▓"*55)

    run_dir   = os.path.join(TRAIN_CFG.out_dir, TRAIN_CFG.run_name)
    pred_path = os.path.join(run_dir, "test_predictions.pt")

    if not os.path.exists(pred_path):
        print(f"[Skip] {pred_path} not found — run training first.")
        return None

    evaluator = FinancialEvaluator(
        predictions_path = pred_path,
        returns_df       = pipe.returns,
        test_split       = splits["test"],
        asset_list       = pipe.asset_list,
        out_dir          = os.path.join(run_dir, "evaluation"),
    )
    fg_results = evaluator.run_all(k_values=EVAL_K_VALUES)

    # ── Full model comparison table ────────────
    if baseline_results:
        print("\n" + "="*68)
        print("  FULL COMPARISON TABLE  (paper Table 1)")
        print("="*68)
        print(f"  {'Model':<18} {'MSE':>10} {'MAE':>10} "
              f"{'HitRate':>9} {'Sharpe(k=20)':>13} {'MaxDD':>8}")
        print(f"  {'-'*65}")

        def fmt_row(name, stat, port):
            print(f"  {name:<18} "
                  f"{stat.get('mse',0):>10.5f} "
                  f"{stat.get('mae',0):>10.5f} "
                  f"{stat.get('hit_rate',0):>9.3%} "
                  f"{port.get('sharpe',0):>13.3f} "
                  f"{port.get('max_dd',0):>8.3%}")

        # FourierGNN row
        fmt_row("FourierGNN ★",
                fg_results["statistical"],
                fg_results["portfolio"].get("top_20", {}))

        # Baseline rows
        for name, bres in baseline_results.items():
            print(f"  {name:<18} "
                  f"{bres.get('mse',0):>10.5f} "
                  f"{bres.get('mae',0):>10.5f} "
                  f"{bres.get('hit_rate',0):>9.3%} "
                  f"{bres.get('sharpe_top20',0):>13.3f} "
                  f"{'—':>8}")
        print("="*68)

        # Save combined CSV
        import csv
        rows = [{"model": "FourierGNN",
                 "mse": fg_results["statistical"]["mse"],
                 "mae": fg_results["statistical"]["mae"],
                 "hit_rate": fg_results["statistical"]["hit_rate"],
                 "sharpe_top20": fg_results["portfolio"].get("top_20",{}).get("sharpe",0),
                 "max_dd": fg_results["portfolio"].get("top_20",{}).get("max_dd",0)}]
        for name, bres in baseline_results.items():
            rows.append({"model": name, **bres, "max_dd": "—"})

        csv_path = os.path.join(TRAIN_CFG.out_dir, "full_comparison.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\n  Saved → {csv_path}")

    return fg_results


def stage_ablations(pipe, splits, builder):
    print("\n" + "▓"*55)
    print("  ABLATIONS")
    print("▓"*55)
    MODEL_CFG.n_assets = len(pipe.asset_list)
    run_ablations(splits, builder, MODEL_CFG, TRAIN_CFG)


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FourierGNN Finance Pipeline")
    p.add_argument("--mode", default="all",
                   choices=["all", "data", "train", "baselines", "eval"],
                   help="Which stage(s) to run")
    p.add_argument("--ablations", action="store_true",
                   help="Run ablation study")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*55)
    print("  FourierGNN — Multi-Asset Return Forecasting")
    print("="*55)
    print(f"  Mode     : {args.mode}")
    print(f"  Data     : {DATA_CFG['start_date']} → {DATA_CFG['end_date']}")
    print(f"  Lookback : {DATA_CFG['lookback']}d   Horizon : {DATA_CFG['horizon']}d")
    print(f"  Graph    : {TRAIN_CFG.graph_method}   Loss : {'NLL+MSE' if TRAIN_CFG.use_nll else 'MSE'}")
    print("="*55)

    pipe, splits     = stage_data()
    if args.mode == "data":
        return

    builder, A       = stage_graph(pipe)

    if args.ablations:
        stage_ablations(pipe, splits, builder)
        return

    fouriergnn_res   = None
    baseline_res     = None

    if args.mode in ("all", "train"):
        _, fouriergnn_res = stage_train(pipe, splits, builder)

    if args.mode in ("all", "baselines"):
        baseline_res = stage_baselines(pipe, splits)

    if args.mode in ("all", "eval"):
        stage_eval(pipe, splits, fouriergnn_res, baseline_res)

    # ── Final summary ──────────────────────────
    run_dir = os.path.join(TRAIN_CFG.out_dir, TRAIN_CFG.run_name)
    print("\n" + "="*55)
    print("  Pipeline complete. Outputs:")
    print(f"  {run_dir}/")
    print(f"    best_model.pt           — FourierGNN checkpoint")
    print(f"    test_predictions.pt     — predictions tensor")
    print(f"    training_curves.png     — loss curves")
    print(f"    evaluation/             — plots + portfolio_metrics.csv")
    print(f"  {TRAIN_CFG.out_dir}/")
    print(f"    baseline_results.csv    — VAR / LSTNet / PatchTST / iTransformer")
    print(f"    full_comparison.csv     — paper Table 1")
    print("="*55)


if __name__ == "__main__":
    main()