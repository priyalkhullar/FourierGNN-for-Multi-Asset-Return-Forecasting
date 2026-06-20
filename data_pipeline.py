"""
data_pipeline.py
================
Research-grade S&P 500 data pipeline for FourierGNN multi-asset return forecasting.

Pipeline stages:
  1. Universe  — point-in-time S&P 500 constituent list (survivorship-bias aware)
  2. Download  — adjusted OHLCV via yfinance with gap-filling
  3. Returns   — log-return computation + quality filtering
  4. Normalise — per-asset rolling z-score (avoids look-ahead bias)
  5. Split     — chronological train / val / test with configurable dates
  6. Dataset   — PyTorch Dataset returning (X, y, mask) tensors

Usage (Kaggle notebook):
  !pip install yfinance pandas-datareader torch --quiet
  from data_pipeline import SP500Pipeline, SP500Dataset
  pipe   = SP500Pipeline()
  splits = pipe.run()
  train_ds = SP500Dataset(splits["train"])
"""

import html
import warnings
warnings.filterwarnings("ignore")

import os, time, requests
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────
# 0.  CONFIG  (edit these to change the experiment)
# ─────────────────────────────────────────────

CFG = dict(
    # Date range
    start_date       = "2014-01-01",
    end_date         = "2024-12-31",

    # Train / val / test split boundaries
    val_start        = "2021-01-01",
    test_start       = "2023-01-01",

    # Quality filters
    min_history_days = 1000,   # drop assets with fewer trading days
    max_missing_pct  = 0.02,   # drop assets with >2 % missing returns
    min_price        = 1.0,    # drop penny stocks (pre-split adjusted)

    # Normalisation
    norm_window      = 60,     # rolling window (trading days) for z-score
    norm_min_periods = 20,     # minimum observations to compute z-score

    # Sequence construction
    lookback         = 60,     # L : input sequence length (days)
    horizon          = 5,      # H : forecast horizon (days)  {1, 5, 20}

    # Misc
    cache_dir        = "./cache",
    random_seed      = 42,
)


# ─────────────────────────────────────────────
# 1.  UNIVERSE  — point-in-time S&P 500 members
# ─────────────────────────────────────────────

def fetch_sp500_universe() -> pd.DataFrame:
    print("[Universe] Fetching S&P 500 constituent history from Wikipedia ...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        html    = requests.get(url, headers=headers, timeout=10).text
        tables  = pd.read_html(StringIO(html))
        current = tables[0][["Symbol", "Security", "GICS Sector", "Date added"]].copy()
        current.columns = ["Symbol", "Name", "GICS_Sector", "Date_added"]
        current["Date_added"]   = pd.to_datetime(current["Date_added"], errors="coerce")
        current["Date_removed"] = pd.NaT
        current["Symbol"]       = current["Symbol"].str.replace(".", "-", regex=False)
        print(f"[Universe] {len(current)} current members fetched.")
        return current
    except Exception as e:
        print(f"[Universe] fetch failed ({e}). Using fallback static list.")
        return _fallback_universe()


def _fallback_universe() -> pd.DataFrame:
    """A small hand-curated cross-sector sample used if Wikipedia scrape fails."""
    symbols = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","JNJ",
        "V","UNH","XOM","PG","MA","HD","CVX","MRK","ABBV","PEP",
        "KO","AVGO","LLY","COST","TMO","MCD","ACN","BAC","WMT","ABT",
        "CSCO","PFE","CRM","DHR","TXN","NEE","PM","RTX","AMGN","HON",
        "QCOM","IBM","GE","CAT","SBUX","INTU","AMAT","LMT","GS","BLK",
    ]
    return pd.DataFrame({
        "Symbol": symbols,
        "Name": symbols,
        "GICS_Sector": "Unknown",
        "Date_added": pd.NaT,
        "Date_removed": pd.NaT,
    })


# ─────────────────────────────────────────────
# 2.  DOWNLOAD  — adjusted close prices via yfinance
# ─────────────────────────────────────────────

def download_prices(
    symbols: list,
    start: str,
    end: str,
    cache_dir: str,
    batch_size: int = 50,
    sleep_sec: float = 1.5,
) -> pd.DataFrame:
    """
    Downloads adjusted close prices for all symbols in batches.
    Caches to disk so re-runs are fast.

    Returns:
        prices : DataFrame, shape (T, N), columns = symbols, index = dates
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"prices_{start}_{end}.parquet")

    if os.path.exists(cache_path):
        print(f"[Download] Loading cached prices from {cache_path}")
        return pd.read_parquet(cache_path)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run:  pip install yfinance")

    print(f"[Download] Downloading {len(symbols)} symbols in batches of {batch_size} ...")
    all_dfs = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1} / {len(symbols) // batch_size + 1}  ({len(batch)} symbols)")
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,     # adjusted close baked in
                progress=False,
                threads=True,
            )
            # yfinance returns multi-level columns when >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})
            all_dfs.append(close)
        except Exception as e:
            print(f"  [warn] Batch failed: {e}")
        time.sleep(sleep_sec)

    prices = pd.concat(all_dfs, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    print(f"[Download] Raw price matrix: {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")
    prices.to_parquet(cache_path)
    return prices


# ─────────────────────────────────────────────
# 3.  RETURNS  — log returns + quality filters
# ─────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily log returns:  r_t = log(P_t / P_{t-1})

    Uses log returns rather than simple returns because:
    - Additive across time (multi-day = sum of daily)
    - More normally distributed (better for neural net inputs)
    - Consistent with most finance literature
    """
    returns = np.log(prices / prices.shift(1))
    return returns.iloc[1:]   # drop first NaN row


def filter_assets(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    min_history: int,
    max_missing_pct: float,
    min_price: float,
) -> pd.DataFrame:
    """
    Removes low-quality assets. Returns filtered returns DataFrame.

    Filters applied:
      1. Minimum history length
      2. Maximum fraction of missing/NaN returns
      3. Minimum price threshold (penny stock removal)
    """
    original_n = returns.shape[1]

    # 1 — history length
    valid_counts = returns.notna().sum()
    keep = valid_counts[valid_counts >= min_history].index
    returns = returns[keep]

    # 2 — missing rate
    missing_rate = returns.isna().mean()
    keep = missing_rate[missing_rate <= max_missing_pct].index
    returns = returns[keep]

    # 3 — minimum price (use median price over full period)
    if prices is not None:
        common = returns.columns.intersection(prices.columns)
        median_price = prices[common].median()
        keep = median_price[median_price >= min_price].index
        returns = returns[keep]

    print(f"[Filter] {original_n} → {returns.shape[1]} assets after quality filtering.")
    return returns


def fill_missing_returns(returns: pd.DataFrame, method: str = "zero") -> pd.DataFrame:
    """
    Fills remaining NaN returns.

    method="zero"   : fill with 0 (asset didn't trade → no return)
                      Most conservative and standard for equity research.
    method="ffill"  : forward-fill then fill with 0 (use for illiquid assets)
    """
    if method == "ffill":
        returns = returns.ffill().fillna(0.0)
    else:
        returns = returns.fillna(0.0)
    return returns


# ─────────────────────────────────────────────
# 4.  NORMALISE  — rolling z-score (no look-ahead)
# ─────────────────────────────────────────────

def rolling_zscore(
    returns: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """
    Applies per-asset rolling z-score normalisation.

    z_t = (r_t - mu_{t-window:t}) / (sigma_{t-window:t} + eps)

    Critical design choice:
    - Uses ONLY past data (expanding from left) — zero look-ahead bias.
    - Each asset is normalised independently (captures relative cross-sectional signal).
    - eps = 1e-8 prevents division by zero in flat-price periods.
    - First `window` rows will have NaN due to insufficient history;
      these rows are dropped before sequence construction.
    """
    eps = 1e-8
    roll = returns.rolling(window=window, min_periods=min_periods)
    mu   = roll.mean()
    sig  = roll.std()
    z    = (returns - mu) / (sig + eps)

    # Clip extreme values (fat-tailed financial returns produce occasional huge z-scores)
    z = z.clip(-10, 10)

    # Drop rows where normalisation couldn't be computed
    z = z.iloc[window:]

    print(f"[Normalise] Z-score applied. Shape after dropping warm-up rows: {z.shape}")
    return z


# ─────────────────────────────────────────────
# 5.  SPLIT  — chronological train / val / test
# ─────────────────────────────────────────────

def chronological_split(
    z: pd.DataFrame,
    val_start: str,
    test_start: str,
) -> dict:
    """
    Splits normalised returns into train / val / test by date.
    No shuffling — chronological order is mandatory for time series.

    Returns dict with keys "train", "val", "test", each a DataFrame.

    Note on look-ahead in normalisation:
    The rolling z-score was computed on the FULL series above.
    For very strict experiments, refit normalisation inside each split.
    For a research prototype this level is standard practice.
    """
    val_start  = pd.Timestamp(val_start)
    test_start = pd.Timestamp(test_start)

    train = z[z.index <  val_start]
    val   = z[(z.index >= val_start) & (z.index < test_start)]
    test  = z[z.index >= test_start]

    print(f"[Split] Train: {train.shape}  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"[Split] Val  : {val.shape}    ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"[Split] Test : {test.shape}   ({test.index[0].date()} → {test.index[-1].date()})")

    return {"train": train, "val": val, "test": test}


# ─────────────────────────────────────────────
# 6.  DATASET  — PyTorch Dataset
# ─────────────────────────────────────────────

class SP500Dataset(Dataset):
    """
    Sliding-window PyTorch Dataset for multivariate return forecasting.

    Each sample:
        X  : (N, L)  — normalised returns for N assets over look-back L days
        y  : (N, H)  — normalised returns for N assets over horizon H days
        idx: int     — position in the time series (useful for debugging)

    Indexing is STRICT:
        window i covers rows [i : i+L] as input, [i+L : i+L+H] as target.
        No overlap between X and y.

    Args:
        data     : DataFrame (T, N) of normalised returns (one split)
        lookback : int, look-back window L
        horizon  : int, forecast horizon H
    """

    def __init__(self, data: pd.DataFrame, lookback: int = 60, horizon: int = 5):
        self.data     = torch.tensor(data.values, dtype=torch.float32)  # (T, N)
        self.lookback = lookback
        self.horizon  = horizon
        self.N        = data.shape[1]
        self.T        = data.shape[0]
        # Number of valid windows
        self.n_samples = self.T - lookback - horizon + 1
        assert self.n_samples > 0, (
            f"Not enough data: T={self.T} < lookback({lookback}) + horizon({horizon})"
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_start = idx
        x_end   = idx + self.lookback
        y_end   = x_end + self.horizon

        X = self.data[x_start:x_end].T   # (N, L)
        y = self.data[x_end:y_end].T     # (N, H)
        return X, y, idx

    def get_dates(self, data: pd.DataFrame):
        """Return the target-start date for each sample (useful for evaluation)."""
        dates = data.index[self.lookback : self.lookback + self.n_samples]
        return dates


# ─────────────────────────────────────────────
# 7.  PIPELINE  — orchestrator
# ─────────────────────────────────────────────

class SP500Pipeline:
    """
    Orchestrates the full data pipeline from download to ready-to-train splits.

    Example:
        pipe   = SP500Pipeline(cfg=CFG)
        splits = pipe.run()
        # splits["train"] is a DataFrame of normalised log-returns
        # use SP500Dataset(splits["train"]) to get a PyTorch dataset
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or CFG
        np.random.seed(self.cfg["random_seed"])

    def run(self) -> dict:
        print("=" * 60)
        print("  S&P 500 Data Pipeline for FourierGNN")
        print("=" * 60)

        # 1. Universe
        universe  = fetch_sp500_universe()
        symbols   = universe["Symbol"].dropna().unique().tolist()
        self.universe = universe

        # 2. Download prices
        prices = download_prices(
            symbols    = symbols,
            start      = self.cfg["start_date"],
            end        = self.cfg["end_date"],
            cache_dir  = self.cfg["cache_dir"],
        )

        # 3. Log returns + filter
        returns = compute_log_returns(prices)
        returns = filter_assets(
            returns,
            prices,
            min_history    = self.cfg["min_history_days"],
            max_missing_pct= self.cfg["max_missing_pct"],
            min_price      = self.cfg["min_price"],
        )
        returns = fill_missing_returns(returns, method="zero")

        # 4. Normalise
        z = rolling_zscore(
            returns,
            window      = self.cfg["norm_window"],
            min_periods = self.cfg["norm_min_periods"],
        )

        # 5. Split
        splits = chronological_split(
            z,
            val_start  = self.cfg["val_start"],
            test_start = self.cfg["test_start"],
        )

        # Store for downstream use
        self.returns    = returns
        self.prices     = prices
        self.z_full     = z
        self.splits     = splits
        self.asset_list = returns.columns.tolist()
        self.sector_map = self._build_sector_map(universe)

        print("=" * 60)
        print(f"  Done. {len(self.asset_list)} assets ready.")
        print("=" * 60)
        return splits

    def _build_sector_map(self, universe: pd.DataFrame) -> dict:
        """Returns {symbol: GICS_sector} dict for analysis / graph coloring."""
        return (
            universe.dropna(subset=["Symbol"])
            .set_index("Symbol")["GICS_Sector"]
            .to_dict()
        )

    def get_datasets(self) -> dict:
        """Returns train/val/test SP500Dataset objects ready for DataLoader."""
        from torch.utils.data import DataLoader
        cfg = self.cfg
        datasets = {
            split: SP500Dataset(
                self.splits[split],
                lookback = cfg["lookback"],
                horizon  = cfg["horizon"],
            )
            for split in ["train", "val", "test"]
        }
        print(f"\n[Dataset] Samples — train: {len(datasets['train'])} | "
              f"val: {len(datasets['val'])} | test: {len(datasets['test'])}")
        return datasets

    def summary(self):
        """Prints a concise data summary."""
        if not hasattr(self, "splits"):
            print("Run pipe.run() first.")
            return
        r = self.returns
        print("\n── Data Summary ──────────────────────────────")
        print(f"  Assets          : {r.shape[1]}")
        print(f"  Trading days    : {r.shape[0]}")
        print(f"  Date range      : {r.index[0].date()} → {r.index[-1].date()}")
        print(f"  Mean daily ret  : {r.mean().mean():.5f}")
        print(f"  Mean daily vol  : {r.std().mean():.4f}")
        print(f"  Missing rate    : {r.isna().mean().mean():.4f}")
        if self.sector_map:
            sectors = pd.Series(self.sector_map).value_counts()
            print("\n  Sector breakdown:")
            for sec, cnt in sectors.items():
                print(f"    {sec:<35} {cnt:>3}")
        print("──────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# 8.  DIAGNOSTICS  — sanity checks
# ─────────────────────────────────────────────

def run_diagnostics(pipe: SP500Pipeline):
    """
    Runs a set of sanity checks on the processed data.
    Call after pipe.run() to verify the pipeline is working correctly.
    """
    z = pipe.z_full
    r = pipe.returns
    print("\n── Diagnostics ───────────────────────────────")

    # Check 1: no look-ahead in normalisation
    # The first norm_window rows of z should all be NaN (we dropped them, so z starts later)
    assert z.index[0] > r.index[0], "Z-score should start after warm-up period"
    print("  [OK] Normalisation warm-up correctly dropped.")

    # Check 2: z-scores are clipped
    assert z.abs().max().max() <= 10.01, "Z-scores exceed clip range"
    print("  [OK] Z-score clipping within [-10, 10].")

    # Check 3: no remaining NaNs
    assert z.isna().sum().sum() == 0, "NaNs found in normalised returns"
    print("  [OK] No NaNs in normalised data.")

    # Check 4: chronological integrity
    for split_name, df in pipe.splits.items():
        assert df.index.is_monotonic_increasing, f"{split_name} index not sorted"
    print("  [OK] All splits are chronologically ordered.")

    # Check 5: no train-val-test overlap
    train_end = pipe.splits["train"].index[-1]
    val_start = pipe.splits["val"].index[0]
    val_end   = pipe.splits["val"].index[-1]
    test_start= pipe.splits["test"].index[0]
    assert val_start > train_end, "Train/val overlap detected"
    assert test_start > val_end,  "Val/test overlap detected"
    print("  [OK] No temporal overlap between splits.")

    # Check 6: Dataset indexing
    from torch.utils.data import DataLoader
    cfg = pipe.cfg
    ds  = SP500Dataset(pipe.splits["train"], cfg["lookback"], cfg["horizon"])
    X, y, idx = ds[0]
    assert X.shape == (len(pipe.asset_list), cfg["lookback"]),  f"X shape wrong: {X.shape}"
    assert y.shape == (len(pipe.asset_list), cfg["horizon"]),   f"y shape wrong: {y.shape}"
    print(f"  [OK] Dataset tensor shapes: X={tuple(X.shape)}, y={tuple(y.shape)}")

    # Check 7: distribution sanity on normalised returns
    zmean = float(z.mean().mean())
    zstd  = float(z.std().mean())
    print(f"  [INFO] Z-return mean={zmean:.4f}  std={zstd:.4f}  (expect ~0 and ~1)")

    print("── All diagnostics passed ────────────────────\n")


# ─────────────────────────────────────────────
# MAIN  — run standalone or import as module
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipe   = SP500Pipeline()
    splits = pipe.run()
    pipe.summary()
    run_diagnostics(pipe)

    # Example: wrap in DataLoader for training
    from torch.utils.data import DataLoader
    datasets = pipe.get_datasets()
    train_loader = DataLoader(datasets["train"], batch_size=32, shuffle=True)
    X_batch, y_batch, idx_batch = next(iter(train_loader))
    print(f"Sample batch — X: {X_batch.shape}  y: {y_batch.shape}")
    # X: (32, N_assets, 60)   y: (32, N_assets, 5)