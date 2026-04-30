# Models

This folder contains the data preparation and stock ranking scripts used by the project. The workflow starts with feature engineering, then produces ranked stock scores that the backtesting pipeline can consume directly.

## Files In This Folder

- [`feature_engineering.py`](/Users/archana/Documents/ai-assisted-investing/models/feature_engineering.py): downloads OHLCV data from `yfinance`, cleans it, engineers per-ticker features, and writes `market_data/features.csv`.
- [`xgboost_model.py`](/Users/archana/Documents/ai-assisted-investing/models/xgboost_model.py): trains a single XGBoost regressor and produces `market_data/model_rankings/xgboost_scores.csv`.
- [`xgboost_randomforest_combined.py`](/Users/archana/Documents/ai-assisted-investing/models/xgboost_randomforest_combined.py): trains both XGBoost and Random Forest models and writes `market_data/model_rankings/xgboost_randomforest_combined_scores.csv`.

## End-To-End Flow

```text
feature_engineering.py
    -> market_data/features.csv
    -> xgboost_model.py or xgboost_randomforest_combined.py
    -> market_data/model_rankings/*.csv
```

## 1. Build The Feature Dataset

Run:

```bash
python3 models/feature_engineering.py
```

The script:

- downloads data for 15 predefined tickers,
- normalizes dates and numeric columns,
- forward-fills missing OHLCV values per ticker,
- engineers technical features,
- drops rows with incomplete feature windows,
- saves the final dataset to `market_data/features.csv`.

### Engineered Features

The current feature set includes:

- `Return_5`
- `Return_20`
- `Return_60`
- `Return_120`
- `MA_20`
- `MA_50`
- `MA_ratio`
- `Volatility`
- `RSI`
- `Volume_change`

## 2. Train The XGBoost Ranking Model

Run:

```bash
python3 models/xgboost_model.py
```

This script:

- loads `market_data/features.csv`,
- creates `Target_20` as the forward 20-day return,
- trains on data from `2015-01-01` through `2019-12-31`,
- evaluates on `2020-01-01` onward,
- creates a normalized `ML_Score`,
- combines ML and momentum signals into `Final_Score`,
- writes a scored dataset to `market_data/model_rankings/xgboost_scores.csv`.

### Scoring Logic

- Target: future 20-day return.
- ML model: `XGBRegressor`.
- Momentum score:
  - `0.15 * Return_5`
  - `0.20 * Return_20`
  - `0.30 * Return_60`
  - `0.35 * Return_120`
- Final score:
  - `0.60 * ML_Score_Norm`
  - `0.40 * Momentum_Score_Norm`

## 3. Train The Combined XGBoost + Random Forest Model

Run:

```bash
python3 models/xgboost_randomforest_combined.py
```

This script follows the same train/test split, but:

- trains both `RandomForestRegressor` and `XGBRegressor`,
- averages their raw predictions into `ML_Score`,
- normalizes both model outputs by date,
- adds `Sector`, `Overall_Rank`, and `Sector_Rank`,
- writes the combined ranking file to `market_data/model_rankings/xgboost_randomforest_combined_scores.csv`.

### Combined Model Logic

- Raw ML blend:
  - `0.50 * XGB_Pred`
  - `0.50 * RF_Pred`
- Final score:
  - `0.60 * ML_Score_Norm`
  - `0.40 * Momentum_Score_Norm`

## Output Schemas

### `xgboost_scores.csv`

Includes:

- `Date`, `Ticker`
- `XGB_Pred`
- `ML_Score`, `ML_Score_Norm`
- `Momentum_Score`, `Momentum_Score_Norm`
- `Final_Score`
- selected input features
- `Signal_Summary`

### `xgboost_randomforest_combined_scores.csv`

Includes:

- `Date`, `Ticker`, `Sector`
- `XGB_Pred`, `RF_Pred`
- `ML_Score`, `ML_Score_Norm`
- `XGB_Score_Norm`, `RF_Score_Norm`
- `Momentum_Score`, `Momentum_Score_Norm`
- `Final_Score`
- `Overall_Rank`, `Sector_Rank`
- selected input features
- `Signal_Summary`

## Notes

- These scripts save scored datasets, not pickled model artifacts.
- `feature_engineering.py` needs internet access because it downloads data from `yfinance`.
- The modeling scripts assume `market_data/features.csv` already exists.
- If you adjust the feature list, update the model scripts and downstream backtesting expectations together.
