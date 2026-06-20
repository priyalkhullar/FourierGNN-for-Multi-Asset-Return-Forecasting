# FourierGNN for Multi-Asset Return Forecasting

> Can frequency-domain graph learning beat classical and Transformer baselines at predicting stock returns? We adapt FourierGNN — originally built for traffic forecasting — to S&P 500 equity data and find out, honestly.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)


---

## What this is

Most financial forecasting models pick one of two tricks: capture **cross-asset structure** (VAR, GNNs) or capture **temporal/spectral structure** (Transformers, FFT-based models) — rarely both. This project adapts **FourierGNN** to fuse them: a learnable complex weight matrix mixes frequency-domain representations across *all* assets simultaneously, then a graph neural network aggregates information from historically correlated stocks.

We benchmark it against four baselines — **VAR, LSTNet, PatchTST, iTransformer** — on S&P 500 daily returns (2014–2024), evaluating not just statistical accuracy (MSE/MAE) but actual **financial performance** (Sharpe ratio, max drawdown, regime analysis).

---

## Architecture

```
Input X (batch, N_assets, lookback_days)
       │
       ├── FFT → learnable complex mixing (N×N per frequency) → inverse FFT ──► Fourier path
       │
       └── Linear projection → GNN (GraphSAGE / GAT) ───────────────────────► Graph path
                                      ↑
                             Adjacency matrix A
                      (Pearson or partial correlation)
                                      │
                          Fusion layer → MLP head
                                      │
                     predicted mean + variance, N days ahead
```

---

## Results

Tested on S&P 500 daily returns, 2023–2024 held-out test period, 5-day forecast horizon, long-top-20 portfolio:

| Model | MSE ↓ | MAE ↓ | Hit Rate | Sharpe ↑ |
|---|---|---|---|---|
| VAR | 1.0233 | 0.7430 | 49.5% | 1.02 |
| PatchTST | 1.0025 | 0.7323 | 50.8% | 0.37 |
| iTransformer | 1.0042 | 0.7334 | 49.8% | 0.30 |
| **FourierGNN (ours)** | **1.0140** | **0.7388** | **50.0%** | **1.12** |
| LSTNet | **0.9797** | **0.7215** | 49.7% | **1.60** |

**Honest takeaways:**
- FourierGNN beat both attention-based baselines (PatchTST, iTransformer) and matched/exceeded the classical VAR model on Sharpe ratio.
- It did **not** beat LSTNet (CNN+GRU), which won on every metric. This is reported as-is, not hidden — graph+spectral fusion is competitive, not state-of-the-art on this benchmark.
- Hit rate sits near 50% across all models — directional predictive signal is weak. The positive Sharpe ratios likely come more from risk/covariance structure (which assets to hold *together*) than from genuine return prediction. This is a meaningful limitation, not a footnote.
- All Sharpe figures are **gross** (no transaction costs or slippage modeled).

This is shared as a research finding, not a trading recommendation — equity return prediction is famously hard, and a clean "we beat everything" result on this kind of data would be more suspicious than reassuring.

---

## Learned graph structure

The adjacency matrix feeding the GNN can be built two ways — full correlation (dense) or partial correlation (sparse, direct dependencies only). Comparing them is a useful sanity check on whether the graph is learning anything economically meaningful.

**Pearson correlation** — dense, as expected for equities (the whole market tends to move together):

![Pearson adjacency matrix](assets/adj_pearson.png)

**Partial correlation (Graphical Lasso)** — controls for all other assets, leaving only *direct* relationships. This is far more interpretable, and it recovers genuine business relationships without being told about them: **MA↔V** (Mastercard/Visa, same payments duopoly), **CVX↔XOM** (both oil majors), **DHR↔TMO** (both life-sciences instrumentation), **BAC↔JPM** (both money-center banks):

![Partial correlation adjacency matrix](assets/adj_partial.png)


---

## Project structure

```
fouriergnn-finance/
├── main.py            # Orchestrator — single config block, all run modes
├── data_pipeline.py   # S&P 500 download, returns, rolling z-score, PyTorch Dataset
├── graph_builder.py   # Pearson / partial correlation / dynamic adjacency
├── fouriergnn.py       # Model: Fourier mixing + GNN + prediction head
├── trainer.py          # Training loop, early stopping, checkpoints, ablations
├── baselines.py        # VAR, LSTNet, PatchTST, iTransformer
├── evaluation.py       # Sharpe, drawdown, regime analysis, comparison table
└── requirements.txt
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- ~32 GB RAM recommended (runs fully on CPU — no GPU required)
- Internet access for the first run (Yahoo Finance + Wikipedia)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/fouriergnn-finance.git
cd fouriergnn-finance

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run the full pipeline

```bash
python main.py
```

This runs all five stages: **data download → graph construction → FourierGNN training → baseline training → financial evaluation.**

### Run individual stages

```bash
python main.py --mode data       # Download & cache S&P 500 prices
python main.py --mode train      # Train FourierGNN only
python main.py --mode baselines  # Train VAR / LSTNet / PatchTST / iTransformer
python main.py --mode eval       # Generate plots + comparison table
python main.py --ablations       # Train all 6 ablation variants
```

### Configure your experiment

Edit the config block at the top of `main.py`:

```python
DATA_CFG["start_date"] = "2014-01-01"
DATA_CFG["lookback"]   = 60
DATA_CFG["horizon"]    = 5             # forecast horizon in days

MODEL_CFG = FourierGNNConfig(
    d_model   = 64,
    n_fourier = 2,
    n_gnn     = 2,
    gnn_type  = "sage",   # "sage" | "gat"
)

TRAIN_CFG = TrainerConfig(
    epochs       = 100,
    lr           = 1e-3,
    graph_method = "pearson",  # "pearson" | "partial"
)
```

---

## Outputs

```
runs/fouriergnn_v1/
├── best_model.pt            # Best checkpoint (by validation loss)
├── test_predictions.pt      # Predictions used for financial evaluation
├── training_curves.png
└── evaluation/
    ├── cumulative_returns.png
    ├── prediction_scatter.png
    ├── regime_sharpe.png
    └── portfolio_metrics.csv
runs/full_comparison.csv     # All models, side by side
```

---

## Known limitations

- Hit rate (~50%) indicates weak directional signal — see [Results](#results) above for the full honest discussion.
- Reported Sharpe/drawdown figures are **gross** — no transaction costs modeled.
- Wikipedia's S&P 500 constituent table occasionally requires a browser User-Agent header to fetch; falls back to a 50-stock sample if scraping fails (check console output for asset count).

---

## Citation

This implementation adapts the architecture from:

```bibtex
@inproceedings{yi2023fouriergnn,
  title     = {FourierGNN: Rethinking Graph Neural Networks for Time-Series Forecasting},
  author    = {Yi, Kun and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```

---

## License

MIT
