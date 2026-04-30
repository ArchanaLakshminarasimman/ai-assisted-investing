# Backtesting

This folder contains the portfolio construction and evaluation logic for the project. The main script, [`backtesting.py`](/Users/archana/Documents/ai-assisted-investing/backtesting/backtesting.py), takes scored stock rankings, selects monthly holdings, simulates an equal-weight portfolio, and compares the strategy against SPY.

## What The Script Does

`run_full_pipeline(...)` performs the full workflow:

1. Load `market_data/features.csv` and a model ranking file.
2. Map each ticker into one of the supported sectors.
3. Merge feature columns into the score file for explainability.
4. Keep the latest available score for each ticker in each month.
5. Select the top names by `Final_Score`.
6. Run an equal-weight backtest between rebalance dates.
7. Download SPY returns from `yfinance` and align benchmark dates.
8. Save holdings, returns, equity curve, drawdown, and summary metrics.
9. Save sector-specific ranked holdings and sector comparison metrics.

## Current Strategy Rules

- Rebalance frequency: monthly.
- Portfolio weighting: equal-weight across selected names.
- Default universe: 15 stocks across Technology, Healthcare, and Finance.
- Default selection size:
  - `All Sectors`: top 10.
  - Individual sector: top 5.
- Default score column: `Final_Score`.
- Default benchmark: `SPY`.

## Inputs

The script expects:

- `market_data/features.csv`
- a ranking file such as `market_data/model_rankings/xgboost_scores.csv`

The ranking file should contain at least:

- `Date`
- `Ticker`
- `Final_Score`

Additional columns such as `ML_Score`, `Momentum_Score`, `RSI`, or `Signal_Summary` are preserved when available and carried into the output holdings files.

## Run The Default Backtest

From the repository root:

```bash
python3 backtesting/backtesting.py
```

This uses:

- `features_path = market_data/features.csv`
- `scores_path = market_data/model_rankings/xgboost_scores.csv`
- `output_dir = xgboost_backtesting_outputs/ranking`
- `sector = All Sectors`
- `top_k = None` which resolves to 10 for the full universe

## Run A Different Scores File

For the combined XGBoost + Random Forest rankings:

```bash
python3 -c "from pathlib import Path; from backtesting.backtesting import run_full_pipeline; root = Path.cwd(); run_full_pipeline(features_path=root/'market_data'/'features.csv', scores_path=root/'market_data'/'model_rankings'/'xgboost_randomforest_combined_scores.csv', output_dir=root/'xgboost_randomforest_backtesting_outputs'/'rankings')"
```

## Run A Sector-Specific Backtest

Example: Finance sector with the top 3 names each month:

```bash
python3 -c "from pathlib import Path; from backtesting.backtesting import run_full_pipeline; root = Path.cwd(); run_full_pipeline(features_path=root/'market_data'/'features.csv', scores_path=root/'market_data'/'model_rankings'/'xgboost_scores.csv', output_dir=root/'xgboost_backtesting_outputs'/'finance_top3', sector='Finance', top_k=3)"
```

## Output Files

The output folder contains:

- `ranked_holdings.csv`: selected names by rebalance date, with rank, scores, and signal summaries.
- `metrics.csv`: strategy and SPY performance metrics.
- `equity_curve.csv`: cumulative return path for strategy and benchmark.
- `drawdown.csv`: strategy and SPY drawdown series.
- `returns.csv`: aligned daily return series.
- `run_summary.csv`: configuration and run-level metadata.
- `sector_metrics.csv`: comparison of All Sectors, Technology, Healthcare, and Finance backtests.
- `ranked_holdings_all_sectors.csv`
- `ranked_holdings_technology.csv`
- `ranked_holdings_healthcare.csv`
- `ranked_holdings_finance.csv`

## Important Functions

- `load_data(...)`: reads and normalizes features and scores.
- `enrich_scores_with_features(...)`: adds signal context from the feature dataset.
- `rank_stocks_monthly(...)`: keeps the latest score in each month and ranks names.
- `run_backtest(...)`: simulates post-rebalance portfolio returns.
- `get_spy_benchmark(...)`: downloads and aligns SPY returns.
- `calculate_metrics(...)`: computes total return, CAGR, volatility, Sharpe ratio, and max drawdown.
- `generate_sector_wise_metrics(...)`: reruns the backtest separately by sector.

## Practical Notes

- SPY data is pulled live from `yfinance`, so the benchmark step needs internet access.
- The script is written as an importable module plus a simple `__main__` entry point; there is no CLI parser yet.
- Output directories are created automatically if they do not already exist.
- If you change the ranking schema, keep `Date`, `Ticker`, and the selected score column intact so the backtest can still run.
