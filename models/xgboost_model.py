import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DATA_DIR = PROJECT_ROOT / "market_data"
MODEL_RANKINGS_DIR = MARKET_DATA_DIR / "model_rankings"
FEATURES_PATH = MARKET_DATA_DIR / "features.csv"
OUTPUT_PATH = MODEL_RANKINGS_DIR / "xgboost_scores.csv"


# -------------------------
# Load data
# -------------------------
df = pd.read_csv(FEATURES_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

print(df.shape)
print(df.columns.tolist())
print("Date range:", df["Date"].min(), "to", df["Date"].max())


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

print(feature_cols)


# -------------------------
# Drop missing values
# -------------------------
model_df = df.dropna(subset=feature_cols + ["Target_20"]).copy()

print("Original shape:", df.shape)
print("Model shape:", model_df.shape)


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
# XGBoost Model
# -------------------------
xgb = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)
print("Model trained successfully")

test_df["XGB_Pred"] = xgb.predict(X_test)


# -------------------------
# Evaluation
# -------------------------
rmse = np.sqrt(mean_squared_error(y_test, test_df["XGB_Pred"]))
mae = mean_absolute_error(y_test, test_df["XGB_Pred"])

print("RMSE:", rmse)
print("MAE:", mae)


# -------------------------
# ML Score
# -------------------------
test_df["ML_Score"] = test_df["XGB_Pred"]


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


test_df["ML_Score_Norm"] = test_df.groupby("Date")["ML_Score"].transform(zscore)
test_df["Momentum_Score_Norm"] = test_df.groupby("Date")["Momentum_Score"].transform(zscore)


# -------------------------
# Final Score
# 60% ML + 40% Momentum
# -------------------------
test_df["Final_Score"] = (
    0.6 * test_df["ML_Score_Norm"] +
    0.4 * test_df["Momentum_Score_Norm"]
)


# -------------------------
# Top 10 latest stocks
# -------------------------
sample_date = test_df["Date"].max()

top_stocks = test_df[test_df["Date"] == sample_date][
    ["Date", "Ticker", "Final_Score"]
].sort_values("Final_Score", ascending=False)

print("\nTop 10 stocks:")
print(top_stocks.head(10))


# -------------------------
# Final output for Person 3 / Person 5 / Frontend
# -------------------------
final_output = test_df[[
    "Date",
    "Ticker",
    "XGB_Pred",
    "ML_Score",
    "ML_Score_Norm",
    "Momentum_Score",
    "Momentum_Score_Norm",
    "Final_Score",
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
    "ML prediction: " + final_output["ML_Score"].round(4).astype(str) +
    ", Momentum score: " + final_output["Momentum_Score"].round(4).astype(str) +
    ", 20-day return: " + final_output["Return_20"].round(4).astype(str) +
    ", 60-day return: " + final_output["Return_60"].round(4).astype(str) +
    ", 120-day return: " + final_output["Return_120"].round(4).astype(str) +
    ", RSI: " + final_output["RSI"].round(2).astype(str) +
    ", Volatility: " + final_output["Volatility"].round(4).astype(str)
)


# -------------------------
# Save output
# -------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_output.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved successfully as {OUTPUT_PATH}")
print(final_output.head())
