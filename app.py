"""
AI-Powered Stock Strategy Dashboard
Team 10 — AI for Engineers

Run with:
    streamlit run app.py

Place this file at the root of ai-assisted-investing-main/
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Animated top bar ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #238636, #3fb950, #58a6ff, #3fb950, #238636);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
}
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #8b949e;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #238636, #3fb950);
}
[data-testid="stMetric"]:hover {
    border-color: #3fb950;
    transform: translateY(-2px);
}
[data-testid="stMetric"] label {
    color: #6e7681 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #3fb950 !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.65rem 1.5rem;
    width: 100%;
    transition: all 0.2s ease;
    letter-spacing: 0.03em;
    box-shadow: 0 2px 8px rgba(46,160,67,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(63,185,80,0.4);
}
.stButton > button:active { transform: translateY(0); }

/* ── Headers ── */
h1 {
    font-family: 'Space Mono', monospace;
    color: #e6edf3;
    letter-spacing: -0.02em;
}
h2 {
    color: #c9d1d9;
    font-size: 1.15rem;
    font-weight: 700;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-top: 0.5rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Divider ── */
hr { border-color: #21262d; margin: 1.5rem 0; }

/* ── Cards ── */
.section-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Stock badge ── */
.stock-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f3a5f, #1f6feb33);
    color: #58a6ff;
    border: 1px solid #1f6feb;
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    margin-right: 0.4rem;
}

/* ── Tags ── */
.tag {
    display: inline-block;
    background: #21262d;
    color: #8b949e;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    margin: 0.2rem;
    border: 1px solid #30363d;
    transition: all 0.2s;
}
.tag:hover { border-color: #3fb950; color: #3fb950; }

/* ── Insight box ── */
.insight-box {
    background: linear-gradient(135deg, #0d1f0d, #0f1f0f);
    border: 1px solid #2ea043;
    border-left: 4px solid #3fb950;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #c9d1d9;
    box-shadow: 0 0 20px rgba(63,185,80,0.08);
}

/* ── Risk box ── */
.risk-box {
    background: linear-gradient(135deg, #1a0e0e, #1f1010);
    border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #ffa198;
    box-shadow: 0 0 15px rgba(248,81,73,0.08);
}

/* ── Stat pill ── */
.stat-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    margin: 0.2rem;
}

/* ── Color helpers ── */
.pos { color: #3fb950; font-family: 'Space Mono', monospace; font-weight: 700; }
.neg { color: #f85149; font-family: 'Space Mono', monospace; font-weight: 700; }
.neutral { color: #58a6ff; font-family: 'Space Mono', monospace; }
.title-accent { color: #3fb950; }

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6e7681;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

/* ── Caption chip ── */
.date-chip {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #8b949e;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1rem;
}



/* ── Stock cards inspired by product dashboards ── */
.stock-card {
    background: linear-gradient(135deg, #161b22 0%, #10161d 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    min-height: 154px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    transition: all 0.2s ease;
}
.stock-card:hover {
    border-color: #3fb950;
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(63,185,80,0.10);
}
.card-rank {
    color: #3fb950;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.card-ticker {
    color: #e6edf3;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.25rem;
}
.card-sector {
    color: #6e7681;
    font-size: 0.78rem;
    margin-top: 0.1rem;
}
.card-score {
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 0.65rem;
}
.card-signal {
    color: #8b949e;
    font-size: 0.78rem;
    line-height: 1.45;
    margin-top: 0.55rem;
}
.subtle-note {
    color:#8b949e;
    font-size:0.82rem;
    line-height:1.55;
    margin-top:-0.4rem;
    margin-bottom:1rem;
}
.info-strip {
    background: linear-gradient(135deg, rgba(31,111,235,0.10), rgba(63,185,80,0.08));
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    color: #c9d1d9;
    font-size: 0.86rem;
    margin: 0.75rem 0 1.2rem 0;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT / "rag"
DATA_DIR = ROOT / "data"


def _find_rag_index() -> Path:
    candidates = [
        ROOT / "rag_index",
        Path.home() / "Downloads" / "ai-assisted-investing-main" / "rag_index",
        Path(__file__).resolve().parent / "rag_index",
    ]
    for c in candidates:
        if (c / "documents.faiss").exists():
            return c
    return ROOT / "rag_index"


RAG_INDEX_DIR = _find_rag_index()

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


def _rag_available() -> bool:
    required = [
        RAG_INDEX_DIR / "documents.faiss",
        RAG_INDEX_DIR / "documents.json",
        RAG_INDEX_DIR / "index_info.json",
    ]
    return all(p.exists() for p in required)


@st.cache_resource(show_spinner=False)
def _load_retriever():
    from faiss_retriever import FAISSRetriever
    return FAISSRetriever(index_dir=str(RAG_INDEX_DIR), local_files_only=False)


def get_rag_insight(ticker: str, stock_row, generator: str = "openai"):
    try:
        from rag_insights import (
            RankedStock,
            build_evidence_items,
            build_key_points,
            build_risk_points,
            build_insight_text,
        )

        retriever = _load_retriever()

        stock = RankedStock(
            ticker=ticker.upper(),
            rank=int(stock_row.get("Rank", 1)),
            score=float(stock_row.get("Final_Score", 0)),
            signal_summary=str(stock_row.get("Signal_Summary", "")) or None,
            sector=str(stock_row.get("Sector", "")) or None,
        )

        query, evidence = build_evidence_items(
            retriever=retriever,
            stock=stock,
            top_k=3,
        )

        if not evidence:
            return "No FAISS evidence found. Run `build_faiss_index.py` first.", [], []

        key_points = build_key_points(evidence)
        risk_points = build_risk_points(evidence)

        insight = build_insight_text(
            generator=generator,
            stock=stock,
            evidence=evidence,
            key_points=key_points,
            risk_points=risk_points,
        )

        return insight, key_points, risk_points

    except FileNotFoundError:
        return "FAISS index not found. Run `python rag/build_faiss_index.py` first.", [], []
    except RuntimeError as e:
        return str(e), [], []
    except Exception as e:
        return f"RAG error: {e}", [], []


STRATEGY_OUTPUTS = {
    "XGBoost": ROOT / "xgboost_backtesting_outputs" / "ranking",
    "XGBoost + Random Forest": ROOT / "xgboost_randomforest_backtesting_outputs" / "rankings",
}


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_outputs(strategy: str):
    folder = STRATEGY_OUTPUTS[strategy]
    out = {}

    for fname in [
        "metrics",
        "sector_metrics",
        "equity_curve",
        "returns",
        "ranked_holdings",
        "ranked_holdings_technology",
        "ranked_holdings_healthcare",
        "ranked_holdings_finance",
        "ranked_holdings_all_sectors",
        "run_summary",
        "drawdown",
    ]:
        path = folder / f"{fname}.csv"
        if path.exists():
            out[fname] = pd.read_csv(path)

    return out


@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str):
    ticker_dir = DATA_DIR / ticker.upper()
    if not ticker_dir.exists():
        return {}

    result = {}
    for json_file in ticker_dir.glob("*.json"):
        with open(json_file) as f:
            result[json_file.stem] = json.load(f)

    return result


def sector_file_key(sector: str) -> str:
    mapping = {
        "All Sectors": "ranked_holdings_all_sectors",
        "Technology": "ranked_holdings_technology",
        "Healthcare": "ranked_holdings_healthcare",
        "Finance": "ranked_holdings_finance",
    }
    return mapping.get(sector, "ranked_holdings")


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _compact_signal(row) -> str:
    parts = []
    r20 = _safe_float(row.get("Return_20", 0))
    r60 = _safe_float(row.get("Return_60", 0))
    rsi = _safe_float(row.get("RSI", 50))
    vol = _safe_float(row.get("Volatility", 0))

    if r20 > 0.03:
        parts.append("strong 20D momentum")
    elif r20 < -0.03:
        parts.append("weak 20D momentum")

    if r60 > 0.06:
        parts.append("positive 60D trend")
    elif r60 < -0.06:
        parts.append("negative 60D trend")

    if rsi < 30:
        parts.append("oversold RSI")
    elif rsi > 70:
        parts.append("overbought RSI")
    else:
        parts.append("neutral RSI")

    if vol > 0.025:
        parts.append("higher volatility")
    return ", ".join(parts[:3]) if parts else "balanced model and momentum signals"


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(family="DM Sans", color="#8b949e", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=10, r=10, t=35, b=10),
)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.2rem'>
        <div style='font-family: Space Mono, monospace; font-size: 1.05rem; color: #e6edf3; font-weight: 700; letter-spacing:-0.01em;'>
            📈 AI Strategy
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    sector = st.selectbox(
        "Sector / Domain",
        ["All Sectors", "Technology", "Healthcare", "Finance"],
        index=0,
    )

    top_k = st.slider(
        "Top K Stocks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    strategy_type = st.selectbox(
        "Strategy Type",
        list(STRATEGY_OUTPUTS.keys()),
        index=0,
    )

    st.markdown("**Rebalancing Frequency**")
    st.markdown(
    """
    <div style='background:#161b22; border:1px solid #30363d; border-radius:8px;
                padding:0.65rem 0.9rem; color:#e6edf3; font-weight:600;'>
        Monthly
    </div>
    """,
    unsafe_allow_html=True
    )
    
    rebalancing = "Monthly"
    st.caption("Strategy rebalances monthly based on the latest ranking signals.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e; text-transform:uppercase;
                letter-spacing:0.08em; margin-bottom:0.4rem;'>
        🤖 AI Insights (RAG)
    </div>
    """, unsafe_allow_html=True)

    rag_generator = st.selectbox(
    "LLM Generator",
    ["openai", "template"],
    index=0,
    help="'template' works offline. OpenAI needs API key in .env file.",
    )

    # Load API keys from .env file or Streamlit secrets — no manual input needed
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass  # dotenv not installed, rely on system env or st.secrets

    if rag_generator == "openai" and not os.environ.get("OPENAI_API_KEY"):
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    openai_key = ""
    gemini_key = ""

    st.markdown("---")
    run_clicked = st.button("🚀 Run Strategy")

    st.markdown("""
    <div style='margin-top: 1.5rem; font-size: 0.72rem; color: #6e7681; line-height: 1.6;'>
        ⚠️ For educational use only.<br>Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────
if "ran" not in st.session_state:
    st.session_state.ran = False

if run_clicked:
    st.session_state.ran = True
    st.session_state.selected_stock = None


# ──────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 1.8rem;'>
    <div style='font-family: Space Mono, monospace; font-size: 0.72rem; color: #3fb950;
                letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.5rem;'>
        AI for Engineers · Team 10
    </div>
    <h1 style='margin:0; font-size: 2rem; line-height: 1.2; letter-spacing:-0.02em;'>
        From Data to Decisions:
        <span class='title-accent'> AI-Assisted</span> Investing
    </h1>
    <p style='color:#8b949e; font-size:0.88rem; margin: 0.5rem 0 0;'>
        Momentum · ML Ranking · Backtesting · RAG Insights
    </p>
    <div class='info-strip'>
        Stocks are ranked using ML predictions and momentum signals, then validated through monthly backtesting against the S&amp;P 500 benchmark. AI insights combine retrieved financial evidence with model signals.
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.ran:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0d1f0d 0%, #161b22 50%, #0d1117 100%);
                border: 1px solid #2ea043; border-radius: 16px; padding: 3.5rem 2rem;
                text-align: center; margin-top: 1rem;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>📈</div>
        <div style='font-family: Space Mono, monospace; font-size: 1.3rem; color: #3fb950;
                    font-weight: 700; margin-bottom: 0.6rem;'>
            Ready to Analyze
        </div>
        <div style='font-size: 1rem; color: #c9d1d9; font-weight: 500; margin-bottom: 1.2rem;'>
            Configure your strategy in the sidebar, then click
            <strong style='color:#3fb950;'>Run Strategy</strong>
        </div>
        <div style='display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;
                    font-size: 0.82rem; color: #6e7681;'>
            <span>🏦 Select Sector</span>
            <span>🎯 Set Top K</span>
            <span>🤖 Choose ML Model</span>
            <span>🚀 Run & Explore</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
with st.spinner("Loading strategy data…"):
    data = load_outputs(strategy_type)

if not data:
    st.error("Could not find output files. Make sure you run backtesting.py first.")
    st.stop()


sector_metrics_df = data.get("sector_metrics", pd.DataFrame())
sector_row = {}

if not sector_metrics_df.empty:
    match = sector_metrics_df[sector_metrics_df["Sector"] == sector]
    if not match.empty:
        sector_row = match.iloc[0].to_dict()


file_key = sector_file_key(sector)
ranked_df = data.get(file_key, data.get("ranked_holdings", pd.DataFrame())).copy()

if not ranked_df.empty:
    ranked_df["Date"] = pd.to_datetime(ranked_df["Date"])
    latest_date = ranked_df["Date"].max()
    latest_df = (
        ranked_df[ranked_df["Date"] == latest_date]
        .sort_values("Rank")
        .head(top_k)
        .reset_index(drop=True)
    )
    st.markdown(f'<div class="date-chip">📅 Latest Rebalance: {latest_date.date()}</div>', unsafe_allow_html=True)
else:
    latest_df = pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# STRATEGY SUMMARY
# ──────────────────────────────────────────────────────────────
st.markdown("## 📋 Strategy Summary")

col1, col2, col3, col4, col5 = st.columns(5)


def _get(key, fallback="—"):
    return sector_row.get(key, fallback)


strat_return = _get("Strategy_Total_Return")
strat_cagr = _get("Strategy_CAGR")
strat_sharpe = _get("Strategy_Sharpe_Ratio")
strat_drawdown = _get("Strategy_Max_Drawdown")
spy_return = _get("SPY_Total_Return")

with col1:
    val = float(strat_return) if strat_return != "—" else 0
    if spy_return != "—":
        spy_val = float(spy_return)
        delta = f"Outperformed SPY by {(val - spy_val) * 100:.1f}%"
    else:
        delta = None
    st.metric("Total Return", f"{val*100:.1f}%", delta=delta)

with col2:
    val = float(strat_cagr) if strat_cagr != "—" else 0
    st.metric("CAGR", f"{val*100:.1f}%")

with col3:
    val = float(strat_sharpe) if strat_sharpe != "—" else 0
    st.metric(
        "Sharpe Ratio",
        f"{val:.3f}",
        delta="Good" if val > 1 else "Needs Improvement",
        delta_color="normal",
    )

with col4:
    val = float(strat_drawdown) if strat_drawdown != "—" else 0
    st.metric("Max Drawdown", f"{val*100:.1f}%", delta_color="inverse")

with col5:
    tickers = latest_df["Ticker"].tolist() if not latest_df.empty else []
    st.metric("Stocks Selected", f"{len(tickers)} / {top_k}", delta=f"{sector}")


# ──────────────────────────────────────────────────────────────
# PORTFOLIO VS SPY
# ──────────────────────────────────────────────────────────────
st.markdown("## 📈 Portfolio vs S&P 500")

equity_df = data.get("equity_curve", pd.DataFrame())

if not equity_df.empty:
    equity_df["Date"] = pd.to_datetime(equity_df["Date"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["Date"],
        y=equity_df["Strategy"],
        name=f"{strategy_type} Strategy",
        line=dict(color="#3fb950", width=2),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.06)",
    ))

    fig.add_trace(go.Scatter(
        x=equity_df["Date"],
        y=equity_df["SPY_Benchmark"],
        name="SPY Benchmark",
        line=dict(color="#58a6ff", width=2, dash="dot"),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Cumulative Growth of $1 Invested",
            font=dict(size=13, color="#c9d1d9"),
        ),
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        height=360,
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Equity curve data not found.")


# ──────────────────────────────────────────────────────────────
# TOP K RANKED STOCKS
# ──────────────────────────────────────────────────────────────
st.markdown(f"## 🏆 Top {top_k} Ranked Stocks — {sector}")

if latest_df.empty:
    st.warning("No ranked holdings data found for selected sector.")
else:
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        y=latest_df["Ticker"],
        x=latest_df["Final_Score"],
        orientation="h",
        marker=dict(
            color=latest_df["Final_Score"],
            colorscale=[[0, "#1f6feb"], [0.5, "#3fb950"], [1, "#ffa657"]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in latest_df["Final_Score"]],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11, color="#c9d1d9"),
    ))

    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="ML Composite Score by Stock",
            font=dict(size=13, color="#c9d1d9"),
        ),
        xaxis_title="Final Score",
        height=300,
    )

    fig2.update_yaxes(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickfont=dict(family="Space Mono", size=12, color="#e6edf3"),
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### ⭐ Top Stock Cards")
    card_cols = st.columns(min(3, len(latest_df)))
    for idx, (_, row) in enumerate(latest_df.head(min(top_k, 6)).iterrows()):
        with card_cols[idx % len(card_cols)]:
            rank_val = int(row.get("Rank", idx + 1))
            ticker_val = row.get("Ticker", "—")
            sector_val = row.get("Sector", sector)
            score_val = _safe_float(row.get("Final_Score", 0))
            signal_val = _compact_signal(row)
            st.markdown(f"""
            <div class='stock-card'>
                <div class='card-rank'>Rank #{rank_val}</div>
                <div class='card-ticker'>{ticker_val}</div>
                <div class='card-sector'>{sector_val}</div>
                <div class='card-score'>Final Score: {score_val:.4f}</div>
                <div class='card-signal'>{signal_val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    display_cols = [
        "Rank",
        "Ticker",
        "Final_Score",
        "Return_20",
        "Return_60",
        "RSI",
        "Volatility",
    ]

    available_cols = [c for c in display_cols if c in latest_df.columns]
    table_df = latest_df[available_cols].copy()

    if "Final_Score" in table_df:
        table_df["Final_Score"] = table_df["Final_Score"].apply(lambda x: f"{x:.4f}")
    if "Return_20" in table_df:
        table_df["Return_20"] = table_df["Return_20"].apply(lambda x: f"{x*100:.2f}%")
    if "Return_60" in table_df:
        table_df["Return_60"] = table_df["Return_60"].apply(lambda x: f"{x*100:.2f}%")
    if "RSI" in table_df:
        table_df["RSI"] = table_df["RSI"].apply(lambda x: f"{x:.1f}")
    if "Volatility" in table_df:
        table_df["Volatility"] = table_df["Volatility"].apply(lambda x: f"{x:.4f}")

    table_df.columns = [c.replace("_", " ") for c in table_df.columns]
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown("### 🔍 Dive into a Stock")

    tickers = latest_df["Ticker"].tolist()
    selected = st.selectbox(
        "Select a stock for detailed insights",
        ["— Choose —"] + tickers,
    )

    if selected != "— Choose —":
        st.session_state.selected_stock = selected


# ──────────────────────────────────────────────────────────────
# STOCK DETAIL
# ──────────────────────────────────────────────────────────────
if st.session_state.get("selected_stock") and not latest_df.empty:
    ticker = st.session_state.selected_stock
    stock_row_df = latest_df[latest_df["Ticker"] == ticker]

    if stock_row_df.empty:
        st.warning(f"No data for {ticker} in current selection.")
    else:
        stock_row = stock_row_df.iloc[0]
        stock_data = load_stock_data(ticker)

        st.markdown("---")
        st.markdown(f"""
        <h2 style='font-family: Space Mono, monospace; font-size: 1.4rem; color: #e6edf3;'>
            <span class='stock-badge'>{ticker}</span>
            {stock_data.get("overview", {}).get("company", ticker)}
            <span style='font-size:0.8rem; color:#6e7681; font-family: DM Sans;'>
                · {stock_data.get("overview", {}).get("sector", sector)}
            </span>
        </h2>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric("Final Score", f"{stock_row.get('Final_Score', 0):.4f}")

        with m2:
            st.metric("Rank", f"#{int(stock_row.get('Rank', 0))}")

        with m3:
            r20 = float(stock_row.get("Return_20", 0))
            st.metric("20D Return", f"{r20*100:.2f}%")

        with m4:
            r60 = float(stock_row.get("Return_60", 0))
            st.metric("60D Return", f"{r60*100:.2f}%")

        with m5:
            rsi = float(stock_row.get("RSI", 0))
            rsi_label = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral")
            st.metric("RSI", f"{rsi:.1f}", delta=rsi_label)

        st.markdown("#### 📉 Score Trend Over Time")

        ticker_history = (
            ranked_df[ranked_df["Ticker"] == ticker].sort_values("Date")
            if not ranked_df.empty
            else pd.DataFrame()
        )

        if not ticker_history.empty and "Final_Score" in ticker_history.columns:
            fig3 = go.Figure()

            fig3.add_trace(go.Scatter(
                x=ticker_history["Date"],
                y=ticker_history["Final_Score"],
                mode="lines+markers",
                name="Final Score",
                line=dict(color="#ffa657", width=2),
                marker=dict(size=4),
            ))

            if "Return_20" in ticker_history.columns:
                fig3.add_trace(go.Scatter(
                    x=ticker_history["Date"],
                    y=ticker_history["Return_20"],
                    mode="lines",
                    name="20D Return",
                    line=dict(color="#58a6ff", width=1.5, dash="dot"),
                    yaxis="y2",
                ))

            fig3.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text=f"{ticker} — Score History",
                    font=dict(size=13, color="#c9d1d9"),
                ),
                height=300,
                hovermode="x unified",
                yaxis2=dict(
                    title="20D Return",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    tickformat=".0%",
                    color="#8b949e",
                ),
            )

            fig3.update_yaxes(title_text="Final Score", gridcolor="#21262d")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 🤖 AI-Generated Insight (RAG + LLM)")
        st.markdown("<div class='subtle-note'>The insight below combines retrieved company evidence with the selected stock’s ranking signals, so the explanation is grounded in both fundamentals and model output.</div>", unsafe_allow_html=True)

        rag_ready = _rag_available()

        if not rag_ready:
            st.markdown("""
            <div class='risk-box'>
                ⚠️ FAISS index not built yet.<br>
                Run: <code>python rag/build_faiss_index.py</code> — then refresh.
            </div>
            """, unsafe_allow_html=True)

        else:
            if rag_generator == "openai" and not os.environ.get("OPENAI_API_KEY"):
                try:
                    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                except Exception:
                    pass


            rag_cache_key = f"rag_{strategy_type}_{sector}_{ticker}_{rag_generator}"

            col_btn, col_clear = st.columns([2, 1])

            with col_btn:
                generate_clicked = st.button(
                    f"✨ Generate insight with {rag_generator.upper()}",
                    key="gen_insight_btn",
                )

            with col_clear:
                if st.button("Clear", key="clear_insight_btn"):
                    st.session_state.pop(rag_cache_key, None)

            if generate_clicked:
                if rag_generator == "openai" and not os.environ.get("OPENAI_API_KEY"):
                    st.error("Please enter your OpenAI API key in the sidebar or configure Streamlit secrets.")
                else:
                    with st.spinner(f"Querying FAISS + calling {rag_generator.upper()}…"):
                        result = get_rag_insight(ticker, stock_row, generator=rag_generator)
                    st.session_state[rag_cache_key] = result

            if rag_cache_key in st.session_state:
                result = st.session_state[rag_cache_key]

                if isinstance(result, tuple):
                    insight_text, kp, rp = result
                else:
                    insight_text, kp, rp = result, [], []

                if (
                    insight_text.startswith("RAG error")
                    or insight_text.startswith("FAISS")
                    or insight_text.startswith("No FAISS")
                ):
                    st.markdown(f'<div class="risk-box">⚠️ {insight_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="insight-box">💡 {insight_text}</div>', unsafe_allow_html=True)

                    if kp:
                        st.markdown("**Key Points from Retrieved Docs:**")
                        for point in kp:
                            st.markdown(f"- {point}")

                    if rp:
                        st.markdown("**Risk Signals from Retrieved Docs:**")
                        risk_html = "".join(f"<div>• {r}</div>" for r in rp)
                        st.markdown(f'<div class="risk-box">{risk_html}</div>', unsafe_allow_html=True)

            else:
                signal = stock_row.get("Signal_Summary", "")
                if signal and str(signal).strip():
                    st.markdown(f"""
                    <div class='section-card' style='font-size:0.85rem; color:#8b949e;'>
                        📋 <strong>Static Signal:</strong> {signal}<br>
                        <span style='font-size:0.75rem;'>Click the button above to generate a live AI explanation.</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("#### ⚠️ Risk Indicators")

        vol = float(stock_row.get("Volatility", 0))
        rsi_val = float(stock_row.get("RSI", 50))
        r20_val = float(stock_row.get("Return_20", 0))

        risks = []

        if vol > 0.025:
            risks.append(f"High volatility detected ({vol:.4f}) — price swings may be significant")
        if rsi_val > 70:
            risks.append(f"RSI is overbought ({rsi_val:.1f}) — potential near-term pullback")
        if rsi_val < 30:
            risks.append(f"RSI is oversold ({rsi_val:.1f}) — may continue declining or signal reversal")
        if r20_val < -0.05:
            risks.append(f"Negative 20-day return ({r20_val*100:.1f}%) — near-term momentum is bearish")

        if risks:
            risk_html = "".join(f"<div>• {r}</div>" for r in risks)
            st.markdown(f'<div class="risk-box">{risk_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#0d1f0d; border:1px solid #2ea043; border-radius:8px;
                        padding:0.7rem 1rem; color:#3fb950; font-size:0.88rem;'>
                ✅ No major risk flags detected for current parameters
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🚀 Potential Catalysts")

        catalysts = []

        if r20_val > 0.05:
            catalysts.append("Strong recent price momentum (20D return > 5%)")
        if 40 < rsi_val < 65:
            catalysts.append("RSI in healthy momentum zone (40–65)")
        if float(stock_row.get("Volume_change", 0)) > 0.1:
            catalysts.append("Rising trading volume signals growing investor interest")
        if float(stock_row.get("Return_60", 0)) > 0.08:
            catalysts.append("Positive 60-day trend indicates sustained institutional buying")

        if catalysts:
            cats_html = "".join(f'<span class="tag">⚡ {c}</span>' for c in catalysts)
            st.markdown(f"<div>{cats_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="tag">No strong catalysts identified at current settings</span>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────
# DRAWDOWN CHART
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📉 Drawdown Analysis")

drawdown_df = data.get("drawdown", pd.DataFrame())

if not drawdown_df.empty and "Date" in drawdown_df.columns:
    drawdown_df["Date"] = pd.to_datetime(drawdown_df["Date"])
    val_col = [c for c in drawdown_df.columns if c != "Date"][0]

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=drawdown_df["Date"],
        y=drawdown_df[val_col],
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.12)",
        line=dict(color="#f85149", width=1.5),
        name="Drawdown",
    ))

    fig4.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Portfolio Drawdown Over Time",
            font=dict(size=13, color="#c9d1d9"),
        ),
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=280,
        hovermode="x",
    )

    st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")

st.markdown(f"""
<div style='text-align:center; padding: 1.5rem 0 0.5rem; margin-top: 1rem;
            border-top: 1px solid #21262d;'>
    <div style='font-family: Space Mono, monospace; font-size: 0.75rem; color: #3fb950;
                letter-spacing: 0.1em; margin-bottom: 0.4rem;'>
        FROM DATA TO DECISIONS
    </div>
    <div style='color:#6e7681; font-size:0.75rem;'>
        AI-Assisted Investing for Retail Investors · Team 10 · AI for Engineers
    </div>
    <div style='color:#8b949e; font-size:0.72rem; margin-top:0.35rem;'>
        Insights generated using RAG over financial documents combined with ML and momentum signals.
    </div>
    <div style='margin-top: 0.6rem; display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap;'>
        <span class='stat-pill'>Strategy: {strategy_type}</span>
        <span class='stat-pill'>Sector: {sector}</span>
        <span class='stat-pill'>Top K: {top_k}</span>
        <span class='stat-pill'>Rebalancing: {rebalancing}</span>
    </div>
</div>
""", unsafe_allow_html=True)