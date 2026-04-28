import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DATA_DIR = PROJECT_ROOT / "market_data"
MODEL_RANKINGS_DIR = MARKET_DATA_DIR / "model_rankings"
FEATURES_PATH = MARKET_DATA_DIR / "features.csv"
OUTPUT_PATH = MODEL_RANKINGS_DIR / "xgboost_randomforest_combined_scores.csv"


# -------------------------
# Load data
# -------------------------
df = pd.read_csv(FEATURES_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

print("Data shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Tickers:", df["Ticker"].nunique())


# -------------------------
# Create target: future 20-day return
# -------------------------
df["Target_20"] = df.groupby("Ticker")["Close"].shift(-20) / df["Close"] - 1


# -------------------------
# Feature columns
# -------------------------
feature_cols = [
    "Return_5",
    "Return_20",
    "Return_60",
    "Return_120",
    "MA_ratio",
    "Volatility",
    "RSI",
    "Volume_change"
]


# -------------------------
# Drop missing values
# -------------------------
model_df = df.dropna(subset=feature_cols + ["Target_20"]).copy()

print("Model data shape:", model_df.shape)


# -------------------------
# Train/Test split
# Train: 2015-2019
# Test: 2020 onward
# -------------------------
train_df = model_df[
    (model_df["Date"] >= "2015-01-01") &
    (model_df["Date"] < "2020-01-01")
].copy()

test_df = model_df[
    model_df["Date"] >= "2020-01-01"
].copy()

X_train = train_df[feature_cols]
y_train = train_df["Target_20"]

X_test = test_df[feature_cols]
y_test = test_df["Target_20"]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train range:", train_df["Date"].min(), "to", train_df["Date"].max())
print("Test range:", test_df["Date"].min(), "to", test_df["Date"].max())
print("Test tickers:", test_df["Ticker"].nunique())


# -------------------------
# Random Forest Model
# -------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
test_df["RF_Pred"] = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, test_df["RF_Pred"]))
rf_mae = mean_absolute_error(y_test, test_df["RF_Pred"])

print("\nRandom Forest Performance")
print("RF RMSE:", rf_rmse)
print("RF MAE:", rf_mae)


# -------------------------
# XGBoost Model
# -------------------------
xgb = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)
test_df["XGB_Pred"] = xgb.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, test_df["XGB_Pred"]))
xgb_mae = mean_absolute_error(y_test, test_df["XGB_Pred"])

print("\nXGBoost Performance")
print("XGB RMSE:", xgb_rmse)
print("XGB MAE:", xgb_mae)


# -------------------------
# Combined ML Score
# Keeps downstream interfaces aligned with the existing single-model pipeline
# while still preserving both model-specific predictions.
# -------------------------
test_df["ML_Score"] = (
    0.5 * test_df["XGB_Pred"] +
    0.5 * test_df["RF_Pred"]
)


# -------------------------
# Momentum Score
# -------------------------
test_df["Momentum_Score"] = (
    0.15 * test_df["Return_5"] +
    0.20 * test_df["Return_20"] +
    0.30 * test_df["Return_60"] +
    0.35 * test_df["Return_120"]
)


# -------------------------
# Normalize scores by date
# -------------------------
def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


test_df["XGB_Score_Norm"] = test_df.groupby("Date")["XGB_Pred"].transform(zscore)
test_df["RF_Score_Norm"] = test_df.groupby("Date")["RF_Pred"].transform(zscore)
test_df["ML_Score_Norm"] = (
    0.5 * test_df["XGB_Score_Norm"] +
    0.5 * test_df["RF_Score_Norm"]
)
test_df["Momentum_Score_Norm"] = test_df.groupby("Date")["Momentum_Score"].transform(zscore)


# -------------------------
# Combined Final Score
# XGBoost = 30%
# Random Forest = 30%
# Momentum = 40%
# -------------------------
test_df["Final_Score"] = (
    0.60 * test_df["ML_Score_Norm"] +
    0.40 * test_df["Momentum_Score_Norm"]
)


# -------------------------
# Add sector/domain
# -------------------------
sector_map = {
    "NVDA": "Technology",
    "MSFT": "Technology",
    "AAPL": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",

    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "MRK": "Healthcare",
    "ABBV": "Healthcare",
    "UNH": "Healthcare",

    "JPM": "Finance",
    "BAC": "Finance",
    "GS": "Finance",
    "MS": "Finance",
    "V": "Finance"
}

test_df["Sector"] = test_df["Ticker"].map(sector_map)


# -------------------------
# Add ranking columns
# -------------------------
test_df["Overall_Rank"] = test_df.groupby("Date")["Final_Score"].rank(
    ascending=False,
    method="first"
)

test_df["Sector_Rank"] = test_df.groupby(["Date", "Sector"])["Final_Score"].rank(
    ascending=False,
    method="first"
)


# -------------------------
# Show latest top stocks overall
# -------------------------
sample_date = test_df["Date"].max()

top_overall = test_df[test_df["Date"] == sample_date][[
    "Date",
    "Ticker",
    "Sector",
    "XGB_Pred",
    "RF_Pred",
    "Momentum_Score",
    "Final_Score",
    "Overall_Rank"
]].sort_values("Final_Score", ascending=False)

print("\nTop 10 overall stocks:")
print(top_overall.head(10))


# -------------------------
# Show latest top stocks by sector
# -------------------------
top_by_sector = test_df[
    (test_df["Date"] == sample_date) &
    (test_df["Sector_Rank"] <= 5)
][[
    "Date",
    "Sector",
    "Ticker",
    "Final_Score",
    "Sector_Rank"
]].sort_values(["Sector", "Sector_Rank"])

print("\nTop stocks by sector:")
print(top_by_sector)


# -------------------------
# Final output for Person 3 / Person 5 / Frontend
# -------------------------
final_output = test_df[[
    "Date",
    "Ticker",
    "Sector",

    "XGB_Pred",
    "RF_Pred",
    "ML_Score",

    "XGB_Score_Norm",
    "RF_Score_Norm",
    "ML_Score_Norm",

    "Momentum_Score",
    "Momentum_Score_Norm",

    "Final_Score",
    "Overall_Rank",
    "Sector_Rank",

    "Return_5",
    "Return_20",
    "Return_60",
    "Return_120",
    "MA_ratio",
    "Volatility",
    "RSI",
    "Volume_change"
]].copy()


# -------------------------
# Signal summary for explainability
# -------------------------
final_output["Signal_Summary"] = (
    "XGBoost prediction: " + final_output["XGB_Pred"].round(4).astype(str) +
    ", Random Forest prediction: " + final_output["RF_Pred"].round(4).astype(str) +
    ", Combined ML score: " + final_output["ML_Score"].round(4).astype(str) +
    ", Momentum score: " + final_output["Momentum_Score"].round(4).astype(str) +
    ", 20-day return: " + final_output["Return_20"].round(4).astype(str) +
    ", 60-day return: " + final_output["Return_60"].round(4).astype(str) +
    ", 120-day return: " + final_output["Return_120"].round(4).astype(str) +
    ", MA ratio: " + final_output["MA_ratio"].round(4).astype(str) +
    ", RSI: " + final_output["RSI"].round(2).astype(str) +
    ", Volatility: " + final_output["Volatility"].round(4).astype(str) +
    ", Volume change: " + final_output["Volume_change"].round(4).astype(str)
)


# -------------------------
# Save output
# -------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_output.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved output as {OUTPUT_PATH}")
print(final_output.head())
