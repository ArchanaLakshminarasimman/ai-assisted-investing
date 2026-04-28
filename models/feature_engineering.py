"""
Data + Feature Engineering pipeline (mirror of feature_engineering.ipynb).
Downloads OHLCV from yfinance, cleans, engineers per-ticker features, saves ML-ready CSV.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# --- STEP 1: Stock universe -------------------------------------------------
TICKERS = [
    "NVDA",
    "MSFT",
    "AAPL",
    "GOOGL",
    "META",
    "JNJ",
    "PFE",
    "MRK",
    "ABBV",
    "UNH",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "V",
]

START_DATE = "2014-01-01"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DATA_DIR = PROJECT_ROOT / "market_data"
OUTPUT_CSV = MARKET_DATA_DIR / "features.csv"


def download_historical_data(
    tickers: list,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download Open, High, Low, Close, Volume per ticker; return long dataframe."""
    frames: list[pd.DataFrame] = []
    for symbol in tickers:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            continue
        raw = raw.rename(columns=str.title)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        part = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        part = part.reset_index()
        date_col = "Date" if "Date" in part.columns else part.columns[0]
        part = part.rename(columns={date_col: "Date"})
        part["Date"] = pd.to_datetime(part["Date"]).dt.normalize()
        part["Ticker"] = symbol
        frames.append(part)

    if not frames:
        raise RuntimeError("No data downloaded; check tickers and network.")

    return pd.concat(frames, ignore_index=True)


def clean_combined_data(df: pd.DataFrame) -> pd.DataFrame:
    """Date column, sort, per-ticker ffill, drop remaining NaNs."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    df = df.sort_values(["Date", "Ticker"], kind="mergesort").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Ticker", "Date"], kind="mergesort")
    for col in numeric_cols:
        df[col] = df.groupby("Ticker", group_keys=False)[col].ffill()

    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI = 100 - 100/(1+RS); RS = avg gain / avg loss (rolling means)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return rsi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker features only (sorted by Ticker, Date)."""
    df = df.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(drop=True)

    out = df.copy()
    g_close = out.groupby("Ticker")["Close"]
    g_vol = out.groupby("Ticker")["Volume"]

    out["Return_5"] = g_close.pct_change(5)
    out["Return_20"] = g_close.pct_change(20)
    out["Return_60"] = g_close.pct_change(60)
    out["Return_120"] = g_close.pct_change(120)

    out["MA_20"] = g_close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    out["MA_50"] = g_close.transform(lambda s: s.rolling(50, min_periods=50).mean())
    out["MA_ratio"] = out["Close"] / out["MA_20"]

    out["Volatility"] = out.groupby("Ticker")["Close"].transform(
        lambda s: s.pct_change().rolling(20, min_periods=20).std()
    )

    out["RSI"] = out.groupby("Ticker")["Close"].transform(lambda s: compute_rsi(s, 14))

    out["Volume_change"] = g_vol.pct_change()

    return out


def build_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return_5",
        "Return_20",
        "Return_60",
        "Return_120",
        "MA_20",
        "MA_50",
        "MA_ratio",
        "Volatility",
        "RSI",
        "Volume_change",
    ]
    final_df = df[cols].copy()
    final_df = final_df.dropna(how="any").reset_index(drop=True)
    return final_df


def main() -> pd.DataFrame:
    # yfinance `end` is exclusive — add one day so today is included
    end_dt = datetime.now().date() + timedelta(days=1)
    end_str = end_dt.strftime("%Y-%m-%d")

    raw = download_historical_data(TICKERS, start=START_DATE, end=end_str)
    cleaned = clean_combined_data(raw)
    featured = engineer_features(cleaned)
    final_df = build_final_dataset(featured)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(final_df)} rows to {OUTPUT_CSV.resolve()}")
    assert final_df.isna().sum().sum() == 0
    print(final_df.head())

    return final_df


if __name__ == "__main__":
    main()
