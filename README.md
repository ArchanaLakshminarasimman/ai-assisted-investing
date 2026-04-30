# AI-Assisted Investing

This repository is a small end-to-end equity research workflow built around three steps:

1. Download and engineer market features for a fixed stock universe.
2. Rank stocks with machine learning and momentum signals.
3. Backtest monthly rebalanced portfolios and optionally generate RAG-based qualitative insights.

The project currently covers 15 large-cap tickers across Technology, Healthcare, and Finance.

## What Is In The Repo

- `models/`: feature engineering and ranking model scripts.
- `backtesting/`: portfolio construction, benchmark comparison, and CSV output generation.
- `market_data/`: engineered feature data plus saved model ranking outputs.
- `data/`: per-ticker JSON documents used by the RAG pipeline.
- `rag/`: FAISS index build and retrieval-driven insight generation.
- `rag_index/`: saved FAISS index artifacts.
- `xgboost_backtesting_outputs/`: saved outputs from the XGBoost backtest flow.
- `xgboost_randomforest_backtesting_outputs/`: saved outputs from the combined-model flow.

## Pipeline Overview

```text
yfinance OHLCV
    -> models/feature_engineering.py
    -> market_data/features.csv
    -> models/xgboost_model.py or models/xgboost_randomforest_combined.py
    -> market_data/model_rankings/*.csv
    -> backtesting/backtesting.py
    -> ranked holdings, returns, metrics, drawdowns, sector comparisons
```

Separately, the `data/` JSON files can be indexed with FAISS and used to generate narrative stock insights.

## Quick Start

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-backtesting.txt
python3 -m pip install -r requirements-rag.txt
```

If you only want the feature engineering, model, and backtesting workflow, `requirements-backtesting.txt` is enough.

### 2. Build market features

```bash
python3 models/feature_engineering.py
```

This downloads OHLCV data with `yfinance` and writes `market_data/features.csv`.

### 3. Train a ranking model

Run the single-model XGBoost pipeline:

```bash
python3 models/xgboost_model.py
```

Or run the blended XGBoost + Random Forest pipeline:

```bash
python3 models/xgboost_randomforest_combined.py
```

These scripts write ranking files to `market_data/model_rankings/`.

### 4. Run the default backtest

```bash
python3 backtesting/backtesting.py
```

By default, this reads `market_data/model_rankings/xgboost_scores.csv` and writes outputs to `xgboost_backtesting_outputs/ranking/`.

### 5. Optional: build the RAG index

```bash
python3 rag/build_faiss_index.py
```

This reads the ticker JSON files in `data/` and writes index artifacts to `rag_index/`.

## Key Outputs

- `market_data/features.csv`: model-ready feature dataset.
- `market_data/model_rankings/xgboost_scores.csv`: XGBoost ranking output.
- `market_data/model_rankings/xgboost_randomforest_combined_scores.csv`: blended-model ranking output.
- `xgboost_backtesting_outputs/ranking/ranked_holdings.csv`: monthly selected holdings with scores and signal summaries.
- `xgboost_backtesting_outputs/ranking/metrics.csv`: strategy vs. SPY performance metrics.
- `xgboost_backtesting_outputs/ranking/equity_curve.csv`: cumulative return series.
- `xgboost_backtesting_outputs/ranking/sector_metrics.csv`: backtest comparison by sector.

## Folder Guide

- Root: project overview, saved outputs, dependency files.
- [`models/README.md`](/Users/archana/Documents/ai-assisted-investing/models/README.md): model training and feature engineering details.
- [`backtesting/README.md`](/Users/archana/Documents/ai-assisted-investing/backtesting/README.md): portfolio construction, metrics, and output files.

## Notes And Assumptions

- `feature_engineering.py` downloads fresh market data, so it needs internet access.
- `backtesting.py` also downloads SPY prices from `yfinance` for benchmarking.
- The model scripts do not save serialized model artifacts; they save scored datasets instead.
- The workflows are script-based today, not packaged as a CLI.
- This is a research prototype and should not be treated as production trading infrastructure or investment advice.
