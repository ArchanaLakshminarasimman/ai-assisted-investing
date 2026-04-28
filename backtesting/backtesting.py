import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIG
# =========================================================

SECTOR_MAP = {
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
    "V": "Finance",
}

VALID_SECTORS = ["All Sectors", "Technology", "Healthcare", "Finance"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DATA_DIR = PROJECT_ROOT / "market_data"
RANKINGS_DIR = MARKET_DATA_DIR / "model_rankings"
OUTPUTS_DIR = PROJECT_ROOT / "xgboost_backtesting_outputs" / "ranking"


# =========================================================
# LOAD + PREPARE DATA
# =========================================================

def load_data(features_path, scores_path):
    features = pd.read_csv(features_path)
    scores = pd.read_csv(scores_path)

    features["Date"] = pd.to_datetime(features["Date"]).dt.normalize()
    scores["Date"] = pd.to_datetime(scores["Date"]).dt.normalize()

    features["Ticker"] = features["Ticker"].astype(str).str.upper()
    scores["Ticker"] = scores["Ticker"].astype(str).str.upper()

    features["Sector"] = features["Ticker"].map(SECTOR_MAP)
    scores["Sector"] = scores["Ticker"].map(SECTOR_MAP)

    features = features.dropna(subset=["Sector"]).sort_values(["Date", "Ticker"])
    scores = scores.dropna(subset=["Sector"]).sort_values(["Date", "Ticker"])

    return features, scores


def enrich_scores_with_features(scores, features):
    """
    Adds useful feature columns from features.csv into scores file
    so ranked_holdings.csv can explain WHY a stock was selected.
    """

    feature_cols = [
        "Date",
        "Ticker",
        "Return_20",
        "Return_60",
        "RSI",
        "Volatility",
        "Volume_change",
    ]

    available_feature_cols = [col for col in feature_cols if col in features.columns]

    # Remove duplicate feature columns from scores if they already exist empty
    cols_to_remove = [
        "Return_20",
        "Return_60",
        "RSI",
        "Volatility",
        "Volume_change",
        "Signal_Summary",
    ]

    scores = scores.drop(columns=[c for c in cols_to_remove if c in scores.columns], errors="ignore")

    enriched = scores.merge(
        features[available_feature_cols],
        on=["Date", "Ticker"],
        how="left",
    )

    enriched["Signal_Summary"] = enriched.apply(create_signal_summary, axis=1)

    return enriched


def create_signal_summary(row):
    return (
        f"Final Score={safe_round(row.get('Final_Score'))}; "
        f"ML Score={safe_round(row.get('ML_Score'))}; "
        f"Momentum Score={safe_round(row.get('Momentum_Score'))}; "
        f"20D Return={safe_round(row.get('Return_20'))}; "
        f"60D Return={safe_round(row.get('Return_60'))}; "
        f"RSI={safe_round(row.get('RSI'))}; "
        f"Volatility={safe_round(row.get('Volatility'))}"
    )


def safe_round(value, digits=3):
    if pd.isna(value):
        return "N/A"
    try:
        return round(float(value), digits)
    except Exception:
        return "N/A"


# =========================================================
# SECTOR + TOP K LOGIC
# =========================================================

def filter_by_sector(df, sector):
    if sector not in VALID_SECTORS:
        raise ValueError(f"Invalid sector: {sector}. Choose from {VALID_SECTORS}")

    if sector == "All Sectors":
        return df.copy()

    filtered = df[df["Sector"] == sector].copy()

    if filtered.empty:
        raise ValueError(f"No data found for sector: {sector}")

    return filtered


def resolve_top_k(sector, requested_top_k=None):
    """
    Default:
    - All Sectors -> Top 10
    - Individual sector -> Top 3
    """

    if requested_top_k is not None:
        return int(requested_top_k)

    if sector == "All Sectors":
        return 10

    return 3


def validate_top_k(df, top_k):
    available = df["Ticker"].nunique()
    return min(top_k, available)


# =========================================================
# MONTHLY RANKING
# =========================================================

def get_monthly_scores(scores):
    """
    Uses latest available score per ticker per month.
    This creates monthly rebalance dates.
    """

    scores = scores.copy()
    scores["Month"] = scores["Date"].dt.to_period("M")

    monthly_scores = (
        scores.sort_values("Date")
        .groupby(["Month", "Ticker"], as_index=False)
        .tail(1)
        .drop(columns=["Month"])
        .reset_index(drop=True)
    )

    return monthly_scores


def rank_stocks_monthly(scores, score_col="Final_Score", top_k=10):
    required = {"Date", "Ticker", "Sector", score_col}
    missing = required - set(scores.columns)

    if missing:
        raise ValueError(f"Missing required columns in scores file: {missing}")

    monthly_scores = get_monthly_scores(scores)

    holdings = []

    for date, group in monthly_scores.groupby("Date"):
        ranked = group.sort_values(score_col, ascending=False).head(top_k)

        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            holdings.append({
                "Date": date,
                "Ticker": row["Ticker"],
                "Sector": row["Sector"],
                "Rank": rank,

                # Scores from Person 2
                "Final_Score": row.get("Final_Score"),
                "ML_Score": row.get("ML_Score"),
                "Momentum_Score": row.get("Momentum_Score"),

                # Feature explanation columns from features.csv
                "Return_20": row.get("Return_20"),
                "Return_60": row.get("Return_60"),
                "RSI": row.get("RSI"),
                "Volatility": row.get("Volatility"),
                "Volume_change": row.get("Volume_change"),

                # Ready for RAG
                "Signal_Summary": row.get("Signal_Summary"),
            })

    holdings_df = pd.DataFrame(holdings)

    if holdings_df.empty:
        raise ValueError("No holdings generated. Check scores and sector filter.")

    return holdings_df


# =========================================================
# BACKTESTING
# =========================================================

def run_backtest(features, holdings, price_col="Close"):
    if price_col not in features.columns:
        raise ValueError(f"{price_col} column not found in features.csv")

    price_matrix = (
        features[["Date", "Ticker", price_col]]
        .pivot(index="Date", columns="Ticker", values=price_col)
        .sort_index()
    )

    daily_returns = price_matrix.pct_change(fill_method=None).dropna(how="all")

    rebalance_dates = sorted(pd.to_datetime(holdings["Date"].unique()))
    portfolio_returns = []

    for i, rebalance_date in enumerate(rebalance_dates):
        selected = holdings.loc[holdings["Date"] == rebalance_date, "Ticker"].tolist()
        selected = [t for t in selected if t in daily_returns.columns]

        if not selected:
            continue

        next_date = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else daily_returns.index.max()

        period_returns = daily_returns.loc[
            (daily_returns.index > rebalance_date)
            & (daily_returns.index <= next_date),
            selected,
        ]

        if period_returns.empty:
            continue

        # Equal-weight portfolio
        portfolio_returns.append(period_returns.fillna(0).mean(axis=1))

    if not portfolio_returns:
        raise ValueError("No portfolio returns generated.")

    strategy_returns = pd.concat(portfolio_returns).sort_index()
    strategy_returns = strategy_returns[~strategy_returns.index.duplicated()]
    strategy_returns.name = "Strategy_Return"

    return strategy_returns


# =========================================================
# SPY BENCHMARK
# =========================================================

def get_spy_benchmark(strategy_returns):
    start = strategy_returns.index.min().strftime("%Y-%m-%d")
    end = (strategy_returns.index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    spy = None

    for attempt in range(3):
        try:
            spy = yf.Ticker("SPY").history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
            )

            if spy is not None and not spy.empty:
                break

        except Exception as e:
            print(f"SPY download attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    if spy is None or spy.empty:
        raise ValueError("SPY download failed after 3 attempts.")

    spy_close = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None).normalize()

    spy_returns = spy_close.pct_change(fill_method=None).dropna()
    spy_returns.name = "SPY_Return"

    common_dates = strategy_returns.index.intersection(spy_returns.index)

    strategy_returns = strategy_returns.loc[common_dates]
    spy_returns = spy_returns.loc[common_dates]

    return strategy_returns, spy_returns


# =========================================================
# EVALUATION
# =========================================================

def calculate_metrics(returns, periods_per_year=252):
    returns = returns.dropna()

    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year

    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    volatility = returns.std() * np.sqrt(periods_per_year)

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(periods_per_year)
        if returns.std() != 0
        else np.nan
    )

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    return {
        "Total Return": round(float(total_return), 4),
        "CAGR": round(float(cagr), 4),
        "Volatility": round(float(volatility), 4),
        "Sharpe Ratio": round(float(sharpe), 4),
        "Max Drawdown": round(float(drawdown.min()), 4),
    }


def calculate_drawdown(equity):
    return equity / equity.cummax() - 1


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    output_dir,
    holdings,
    metrics,
    equity_curve,
    drawdown,
    returns,
    run_summary,
):
    os.makedirs(output_dir, exist_ok=True)

    holdings.to_csv(f"{output_dir}/ranked_holdings.csv", index=False)
    metrics.to_csv(f"{output_dir}/metrics.csv", index_label="Metric")
    equity_curve.to_csv(f"{output_dir}/equity_curve.csv", index=False)
    drawdown.to_csv(f"{output_dir}/drawdown.csv", index=False)
    returns.to_csv(f"{output_dir}/returns.csv", index=False)
    run_summary.to_csv(f"{output_dir}/run_summary.csv", index=False)


def save_sector_wise_holdings(all_scores_df, output_dir, score_col="Final_Score"):
    """
    Generates sector-wise ranking CSVs for ALL sectors,
    independent of what user selected.

    This is for RAG + comparison.
    """

    os.makedirs(output_dir, exist_ok=True)

    for sector in VALID_SECTORS:
        if sector == "All Sectors":
            sector_df = all_scores_df.copy()
            top_k = 10
            filename = "all_sectors"
        else:
            sector_df = all_scores_df[all_scores_df["Sector"] == sector].copy()
            top_k = 3
            filename = sector.lower()

        if sector_df.empty:
            continue

        holdings = rank_stocks_monthly(
            scores=sector_df,
            score_col=score_col,
            top_k=min(top_k, sector_df["Ticker"].nunique())
        )

        holdings["Date"] = holdings["Date"].dt.strftime("%Y-%m-%d")

        holdings.to_csv(
            f"{output_dir}/ranked_holdings_{filename}.csv",
            index=False
        )


def generate_sector_wise_metrics(
    features,
    scores,
    output_dir,
    score_col="Final_Score",
    price_col="Close"
):
    """
    Runs backtesting separately for:
    - All Sectors
    - Technology
    - Healthcare
    - Finance

    Saves one comparison file:
    outputs/sector_metrics.csv
    """

    rows = []

    for sector in VALID_SECTORS:
        sector_scores = filter_by_sector(scores, sector)
        sector_features = filter_by_sector(features, sector)

        top_k = resolve_top_k(sector, requested_top_k=None)
        top_k = validate_top_k(sector_scores, top_k)

        holdings = rank_stocks_monthly(
            scores=sector_scores,
            score_col=score_col,
            top_k=top_k
        )

        strategy_returns = run_backtest(
            features=sector_features,
            holdings=holdings,
            price_col=price_col
        )

        strategy_returns, spy_returns = get_spy_benchmark(strategy_returns)

        strategy_metrics = calculate_metrics(strategy_returns)
        spy_metrics = calculate_metrics(spy_returns)

        rows.append({
            "Sector": sector,
            "Top_K": top_k,
            "Unique_Tickers": sector_scores["Ticker"].nunique(),
            "Monthly_Rebalance_Dates": holdings["Date"].nunique(),
            "Total_Holding_Rows": len(holdings),

            "Strategy_Total_Return": strategy_metrics["Total Return"],
            "Strategy_CAGR": strategy_metrics["CAGR"],
            "Strategy_Volatility": strategy_metrics["Volatility"],
            "Strategy_Sharpe_Ratio": strategy_metrics["Sharpe Ratio"],
            "Strategy_Max_Drawdown": strategy_metrics["Max Drawdown"],

            "SPY_Total_Return": spy_metrics["Total Return"],
            "SPY_CAGR": spy_metrics["CAGR"],
            "SPY_Volatility": spy_metrics["Volatility"],
            "SPY_Sharpe_Ratio": spy_metrics["Sharpe Ratio"],
            "SPY_Max_Drawdown": spy_metrics["Max Drawdown"],
        })

    sector_metrics_df = pd.DataFrame(rows)
    sector_metrics_df.to_csv(f"{output_dir}/sector_metrics.csv", index=False)

    return sector_metrics_df


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_full_pipeline(
    features_path=MARKET_DATA_DIR / "features.csv",
    scores_path=RANKINGS_DIR / "xgboost_scores.csv",
    output_dir=OUTPUTS_DIR,
    sector="All Sectors",
    top_k=None,
    score_col="Final_Score",
    price_col="Close",
):
    print("Loading data...")
    features, scores = load_data(features_path, scores_path)

    print("Merging scores with feature explanation columns...")
    scores = enrich_scores_with_features(scores, features)

    print(f"Selected sector: {sector}")

    requested_top_k = top_k
    resolved_top_k = resolve_top_k(sector, requested_top_k)

    filtered_scores = filter_by_sector(scores, sector)
    filtered_features = filter_by_sector(features, sector)

    used_top_k = validate_top_k(filtered_scores, resolved_top_k)

    print(f"Requested Top K: {'Auto' if requested_top_k is None else requested_top_k}")
    print(f"Used Top K: {used_top_k}")

    print("Ranking stocks monthly...")
    holdings = rank_stocks_monthly(
        scores=filtered_scores,
        score_col=score_col,
        top_k=used_top_k,
    )

    print("Running backtest...")
    strategy_returns = run_backtest(
        features=filtered_features,
        holdings=holdings,
        price_col=price_col,
    )

    print("Downloading and aligning SPY benchmark...")
    strategy_returns, spy_returns = get_spy_benchmark(strategy_returns)

    strategy_equity = (1 + strategy_returns).cumprod()
    spy_equity = (1 + spy_returns).cumprod()

    print("Calculating metrics...")
    strategy_metrics = calculate_metrics(strategy_returns)
    spy_metrics = calculate_metrics(spy_returns)

    metrics_df = pd.DataFrame({
        "Strategy": strategy_metrics,
        "SPY Benchmark": spy_metrics,
    })

    equity_curve_df = pd.DataFrame({
        "Date": strategy_equity.index,
        "Strategy": strategy_equity.values,
        "SPY_Benchmark": spy_equity.values,
    })

    drawdown_df = pd.DataFrame({
        "Date": strategy_equity.index,
        "Strategy_Drawdown": calculate_drawdown(strategy_equity).values,
        "SPY_Drawdown": calculate_drawdown(spy_equity).values,
    })

    returns_df = pd.DataFrame({
        "Date": strategy_returns.index,
        "Strategy_Return": strategy_returns.values,
        "SPY_Return": spy_returns.values,
    })

    holdings_output = holdings.copy()
    holdings_output["Date"] = holdings_output["Date"].dt.strftime("%Y-%m-%d")

    run_summary_df = pd.DataFrame([{
        "Selected_Sector": sector,
        "Requested_Top_K": "Auto" if requested_top_k is None else requested_top_k,
        "Resolved_Top_K": resolved_top_k,
        "Used_Top_K": used_top_k,
        "Score_Column": score_col,
        "Price_Column": price_col,
        "Rebalance_Frequency": "Monthly",
        "Benchmark": "SPY",
        "Unique_Tickers": filtered_scores["Ticker"].nunique(),
        "Monthly_Rebalance_Dates": holdings["Date"].nunique(),
        "Total_Holding_Rows": len(holdings),
    }])

    print("Saving outputs...")
    save_outputs(
        output_dir=output_dir,
        holdings=holdings_output,
        metrics=metrics_df,
        equity_curve=equity_curve_df,
        drawdown=drawdown_df,
        returns=returns_df,
        run_summary=run_summary_df,
    )

    save_sector_wise_holdings(
    all_scores_df=scores,
    output_dir=output_dir,
    score_col=score_col
    )

    sector_metrics_df = generate_sector_wise_metrics(
    features=features,
    scores=scores,
    output_dir=output_dir,
    score_col=score_col,
    price_col=price_col
)

    print("\nBacktest complete.")
    print("\nRun Summary:")
    print(run_summary_df.to_string(index=False))

    print("\nStrategy Metrics:")
    print(strategy_metrics)

    print("\nSPY Benchmark Metrics:")
    print(spy_metrics)

    print(f"\nOutputs saved to: {output_dir}/")

    return {
        "holdings": holdings_output,
        "metrics": metrics_df,
        "equity_curve": equity_curve_df,
        "drawdown": drawdown_df,
        "returns": returns_df,
        "run_summary": run_summary_df,
        "sector_metrics": sector_metrics_df,
    }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":
    run_full_pipeline(
        features_path=MARKET_DATA_DIR / "features.csv",
        scores_path=RANKINGS_DIR / "xgboost_scores.csv",
        output_dir=OUTPUTS_DIR,
        sector="All Sectors",   # All Sectors / Technology / Healthcare / Finance
        top_k=None,             # Auto: All=10, Sector=3
        score_col="Final_Score",
        price_col="Close",
    )
