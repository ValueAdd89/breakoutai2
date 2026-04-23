"""
Signal — Professional Trading Analytics (Pro Edition)
======================================================
Data sources:
  • yfinance         — daily OHLCV, intraday 5-min bars, float/short data
  • Finnhub (free)   — news, company profile, earnings calendar
                       Set FINNHUB_API_KEY in Streamlit secrets for news/catalysts

Features:
  • Configurable auto-refresh (30s / 1m / 2m / 5m / 15m)
  • Daily breakout scanner with 6-factor composite scoring (100 pts base)
  • Pro-grade scoring layer adding float, short interest, news catalysts, intraday (50 pts bonus)
  • 5-minute intraday scanner with VWAP, Opening Range Breakouts, new-day-highs
  • News feed with sentiment & importance tagging
  • Options plays with 90% CI and Black-Scholes pricing
  • All operations parallelized; single batch data fetch
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

from utils.data import (
    fetch_batch_data, get_ticker_info, is_penny_stock,
    DEFAULT_TICKERS, LARGE_CAP_TICKERS, PENNY_TICKERS,
    REFRESH_INTERVALS, CACHE_TTL_MAP,
)
from utils.catalysts import (
    fetch_news, fetch_company_profile, fetch_earnings_calendar,
    _get_api_key,
)
from utils.intraday import (
    fetch_intraday_batch, compute_intraday_stats, is_market_hours,
)
from models.predictor import predict_batch_parallel, Signal
from models.pro_scorer import compute_pro_breakout
from models.options_engine import generate_options_plays

st.set_page_config(page_title="Signal — Pro Analytics", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

FINNHUB_KEY_PRESENT = bool(_get_api_key())

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◉ Signal Config")
    st.markdown("---")
    refresh_choice = st.selectbox("⏱ Refresh Interval", list(REFRESH_INTERVALS.keys()), index=2)
    st.markdown("---")
    ticker_mode = st.radio("Watchlist", ["All (Large + Penny)", "Large Cap Only", "Penny / Small Cap Only", "Custom"])
    if ticker_mode == "All (Large + Penny)":
        selected_tickers = DEFAULT_TICKERS
    elif ticker_mode == "Large Cap Only":
        selected_tickers = LARGE_CAP_TICKERS
    elif ticker_mode == "Penny / Small Cap Only":
        selected_tickers = PENNY_TICKERS
    else:
        selected_tickers = st.multiselect("Custom Tickers", options=DEFAULT_TICKERS, default=DEFAULT_TICKERS[:8])
    st.markdown("---")
    lookback = st.selectbox("Lookback (Daily)", ["3mo", "6mo", "1y", "2y"], index=1)

    st.markdown("---")
    enable_pro = st.checkbox("Enable Pro Scoring", value=True, help="Adds float, short interest, news catalysts, and intraday structure to breakout scoring. Uses more API calls.")
    enable_intraday = st.checkbox("Enable Intraday Data", value=True, help="Fetches 5-minute bars for the current session")

    st.markdown("---")
    if FINNHUB_KEY_PRESENT:
        st.success("✓ Finnhub API connected")
    else:
        st.warning("⚠ No Finnhub API key")
        with st.expander("Setup news + profile data"):
            st.markdown("""
            1. Get a free key at **finnhub.io** (60 req/min)
            2. On Streamlit Cloud: Settings → Secrets, add:
            ```
            FINNHUB_API_KEY = "your-key"
            ```
            3. Or locally: `export FINNHUB_API_KEY=your-key`
            """)

    st.markdown("---")
    st.markdown("""
    **Scoring System**

    **Base (0-100)** — technical:
    - Volume (25), Squeeze (20)
    - Momentum (15), Accumulation (15)
    - Key Levels (10), Model (15)

    **Pro (0-50)** — trader data:
    - Float Tier (15)
    - Float Turnover (10)
    - Short Squeeze (10)
    - News Catalysts (10)
    - Intraday Structure (5)

    **Total: 0-150**
    """)

if HAS_AUTOREFRESH:
    st_autorefresh(interval=REFRESH_INTERVALS[refresh_choice], limit=None, key="data_refresh")

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root { --bg-card: rgba(255,255,255,0.03); --border: rgba(255,255,255,0.06); --text-primary: #F5F5F7; --text-secondary: rgba(255,255,255,0.45); --text-tertiary: rgba(255,255,255,0.25); --font-display: 'DM Sans'; --font-mono: 'JetBrains Mono'; }
.stApp { background: linear-gradient(180deg, #000 0%, #070710 50%, #0d0d15 100%) !important; }
#MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { background: #08080e !important; border-right: 1px solid var(--border) !important; }
div[data-testid="stMetric"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 16px 18px !important; }
div[data-testid="stMetric"] label { font-family: 'DM Sans' !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; color: var(--text-tertiary) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono' !important; font-weight: 600 !important; font-size: 22px !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; border: 1px solid var(--border) !important; background: var(--bg-card) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans' !important; font-size: 14px !important; font-weight: 500 !important; color: var(--text-secondary) !important; border-radius: 8px 8px 0 0 !important; padding: 10px 18px !important; }
.stTabs [aria-selected="true"] { color: var(--text-primary) !important; background: var(--bg-card) !important; }
div[data-baseweb="select"] > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.sig-h { font-family: 'DM Sans'; font-size: 38px; font-weight: 700; letter-spacing: -0.03em; color: #F5F5F7; margin: 0; }
.sig-sub { font-family: 'DM Sans'; font-size: 13px; color: rgba(255,255,255,0.3); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; }
.badge-bullish { display:inline-block; background:rgba(52,199,89,0.12); color:#34C759; padding:4px 12px; border-radius:20px; font-family:'DM Sans'; font-size:12px; font-weight:600; }
.badge-bearish { display:inline-block; background:rgba(255,69,58,0.12); color:#FF453A; padding:4px 12px; border-radius:20px; font-family:'DM Sans'; font-size:12px; font-weight:600; }
.badge-neutral { display:inline-block; background:rgba(255,214,10,0.12); color:#FFD60A; padding:4px 12px; border-radius:20px; font-family:'DM Sans'; font-size:12px; font-weight:600; }
.sig-item { display:flex; align-items:center; gap:10px; padding:8px 0; font-family:'DM Sans'; font-size:14px; color:rgba(255,255,255,0.65); border-bottom:1px solid rgba(255,255,255,0.04); }
.sig-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.mono { font-family: 'JetBrains Mono' !important; }
.live-badge { display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono'; font-size:11px; color:rgba(255,255,255,0.4); }
.live-dot { width:6px; height:6px; border-radius:50%; background:#34C759; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.grade-badge { display:inline-flex; align-items:center; justify-content:center; width:48px; height:48px; border-radius:12px; font-family:'JetBrains Mono'; font-size:19px; font-weight:700; }
.section-label { font-family:'DM Sans'; font-size:11px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px; margin-top:20px; }
.driver-row { display:flex; align-items:flex-start; gap:10px; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.driver-tag { font-family:'JetBrains Mono'; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; padding:2px 8px; border-radius:4px; white-space:nowrap; flex-shrink:0; margin-top:2px; }
.play-type-badge { display:inline-block; padding:3px 10px; border-radius:6px; font-family:'JetBrains Mono'; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }
.news-item { padding:12px 16px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; margin-bottom:8px; border-left:3px solid rgba(255,255,255,0.1); }
.news-item.bullish { border-left-color:#34C759; }
.news-item.bearish { border-left-color:#FF453A; }
.news-headline { font-family:'DM Sans'; font-size:14px; font-weight:600; color:#F5F5F7; line-height:1.4; margin-bottom:6px; }
.news-meta { font-family:'JetBrains Mono'; font-size:10px; color:rgba(255,255,255,0.35); }
.news-summary { font-family:'DM Sans'; font-size:12px; color:rgba(255,255,255,0.5); margin-top:6px; line-height:1.5; }
.imp-badge { display:inline-block; padding:1px 6px; border-radius:4px; font-family:'JetBrains Mono'; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; margin-right:6px; }
.vwap-line { background:rgba(10,132,255,0.08); border-radius:8px; padding:10px 14px; font-family:'JetBrains Mono'; font-size:13px; margin-bottom:12px; border:1px solid rgba(10,132,255,0.2); }
.kv-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
.kv-cell { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:12px 14px; }
.kv-label { font-family:'DM Sans'; font-size:10px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }
.kv-value { font-family:'JetBrains Mono'; font-size:16px; font-weight:600; color:#F5F5F7; }
.kv-sub { font-family:'DM Sans'; font-size:10px; color:rgba(255,255,255,0.2); margin-top:2px; }
.play-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:24px; margin-bottom:18px; position:relative; overflow:hidden; }
.play-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:18px 18px 0 0; }
.play-card.bullish::before { background:linear-gradient(90deg,#34C759,#30D158); }
.play-card.bearish::before { background:linear-gradient(90deg,#FF453A,#FF6961); }
.play-card.neutral::before { background:linear-gradient(90deg,#0A84FF,#5AC8FA); }
.play-card.volatility::before { background:linear-gradient(90deg,#BF5AF2,#FF6FF1); }
.play-strategy { font-family:'DM Sans'; font-size:22px; font-weight:700; color:#F5F5F7; letter-spacing:-0.02em; }
.leg-row { display:flex; align-items:center; gap:12px; padding:10px 14px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; margin-bottom:6px; font-family:'JetBrains Mono'; font-size:13px; }
.leg-direction { font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; padding:2px 8px; border-radius:4px; }
.thesis-block { background:rgba(255,255,255,0.02); border-left:3px solid rgba(255,255,255,0.1); padding:16px 20px; border-radius:0 10px 10px 0; font-family:'DM Sans'; font-size:14px; color:rgba(255,255,255,0.6); line-height:1.7; }
.ci-bar-container { position:relative; height:40px; background:rgba(255,255,255,0.03); border-radius:8px; border:1px solid rgba(255,255,255,0.06); overflow:hidden; margin:12px 0; }
.ci-bar-fill { position:absolute; top:0; bottom:0; border-radius:8px; opacity:0.2; }
.ci-marker { position:absolute; top:50%; transform:translate(-50%,-50%); font-family:'JetBrains Mono'; font-size:10px; font-weight:600; white-space:nowrap; }
.risk-item { display:flex; align-items:flex-start; gap:8px; padding:6px 0; font-family:'DM Sans'; font-size:13px; color:rgba(255,255,255,0.55); line-height:1.5; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="rgba(255,255,255,0.5)", size=11),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
    hoverlabel=dict(bgcolor="#1c1c1e", font_size=12, font_family="JetBrains Mono"),
)

def color_for(d): return {"bullish":"#34C759","bearish":"#FF453A"}.get(d,"#FFD60A")
def icon_for(d): return {"bullish":"↑","bearish":"↓"}.get(d,"→")
def grade_color(g):
    if g.startswith("A"): return "#34C759"
    if g.startswith("B"): return "#0A84FF"
    if g.startswith("C"): return "#FFD60A"
    return "#FF453A"
def sentiment_color(s):
    return {"bullish":"#34C759","bearish":"#FF453A"}.get(s,"rgba(255,255,255,0.3)")

def to_list(df, col):
    """
    Return a plain Python list of floats for a DataFrame column.
    Plotly 6.x on Python 3.14 rejects numpy arrays and pandas Series
    for Candlestick OHLC fields in some configurations. Plain Python
    lists always pass validation.
    """
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return [float(v) for v in s.values.ravel()]


def idx_to_str(index):
    """
    Convert a DatetimeIndex to a list of ISO strings.
    Plotly 6.x on Python 3.14 can choke on timezone-aware pandas
    DatetimeIndex objects passed directly to x= parameters.
    """
    try:
        return [t.isoformat() for t in index]
    except Exception:
        return list(index)

# ─── Header ──────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<p class="sig-h">Signal</p>', unsafe_allow_html=True)
    st.markdown('<p class="sig-sub">Professional Trading Analytics · Pro Edition</p>', unsafe_allow_html=True)
with h2:
    mkt = "OPEN" if is_market_hours() else "CLOSED"
    mkt_clr = "#34C759" if mkt == "OPEN" else "rgba(255,255,255,0.3)"
    st.markdown(
        f'<div style="text-align:right; padding-top:16px;">'
        f'<span class="live-badge"><span class="live-dot" style="background:{mkt_clr};"></span> MKT {mkt} · {refresh_choice}</span>'
        f'<br><span class="mono" style="font-size:11px; color:rgba(255,255,255,0.2);">{datetime.now().strftime("%H:%M:%S")}</span></div>',
        unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── Data Pipeline ───────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice, 120), show_spinner=False)
def load_and_predict(tickers, lookback):
    data = fetch_batch_data(tickers, period=lookback)
    sigs = predict_batch_parallel(data)
    return data, sigs

@st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice, 120), show_spinner=False)
def load_intraday(tickers):
    return fetch_intraday_batch(tickers, days=5)

progress = st.progress(0, text="Fetching daily market data & running models...")
data_dict, signals = load_and_predict(tuple(selected_tickers), lookback)
progress.progress(40, text="Daily models complete.")

# Intraday (optional — skip if market closed to save API)
intraday_dict = {}
intraday_stats_dict = {}
if enable_intraday:
    progress.progress(55, text="Fetching 5-minute intraday bars...")
    intraday_dict = load_intraday(tuple(selected_tickers))
    for t, df in intraday_dict.items():
        sig = signals.get(t)
        avg_vol = 0
        if sig and t in data_dict:
            daily = data_dict[t]
            avg_vol = float(daily["Volume"].tail(20).mean())
        stats = compute_intraday_stats(t, df, avg_daily_vol=avg_vol)
        if stats:
            intraday_stats_dict[t] = stats
    progress.progress(75, text="Intraday data complete.")

# Pro scoring: merge signals with profile/news/intraday
pro_scores = {}
profiles = {}
news_by_ticker = {}

if enable_pro:
    progress.progress(80, text="Computing pro-grade breakout scores...")
    for ticker, sig in signals.items():
        profile = fetch_company_profile(ticker) if FINNHUB_KEY_PRESENT or True else None
        news = fetch_news(ticker, days_back=3, max_items=5) if FINNHUB_KEY_PRESENT else []
        profiles[ticker] = profile
        news_by_ticker[ticker] = news

        daily_vol = 0
        if ticker in data_dict:
            daily_vol = float(data_dict[ticker]["Volume"].iloc[-1])

        pro = compute_pro_breakout(
            base_score=sig.breakout_score,
            ticker=ticker,
            daily_volume=daily_vol,
            profile=profile,
            news_items=news,
            intraday_stats=intraday_stats_dict.get(ticker),
        )
        pro_scores[ticker] = pro

progress.progress(100, text="Ready.")
progress.empty()

# ─── Summary Metrics ─────────────────────────────────────────────
bull = sum(1 for s in signals.values() if s.direction == "bullish")
bear = sum(1 for s in signals.values() if s.direction == "bearish")
neut = len(signals) - bull - bear
if pro_scores:
    top_pro = sum(1 for p in pro_scores.values() if p.total_score >= 100)
    avg_pro = np.mean([p.total_score for p in pro_scores.values()])
else:
    top_pro = sum(1 for s in signals.values() if s.breakout_score >= 65)
    avg_pro = np.mean([s.breakout_score for s in signals.values()]) if signals else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Bullish", bull)
c2.metric("Bearish", bear)
c3.metric("Neutral", neut)
c4.metric("Avg Score", f"{avg_pro:.0f}")
c5.metric("High Probability", f"{top_pro}", delta="Score ≥ 100" if pro_scores else "Score ≥ 65")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────
tab_overview, tab_breakout, tab_intraday, tab_detail, tab_options, tab_heatmap = st.tabs([
    "◉ Overview", "🔥 Breakout Scanner", "⚡ Intraday (5m)",
    "◎ Detail", "⬡ Options", "◈ Heatmap",
])

# ═══ TAB 1: OVERVIEW ═══
with tab_overview:
    if not signals:
        st.warning("No signals available.")
    else:
        sort_key = (lambda s: pro_scores[s.ticker].total_score) if pro_scores else (lambda s: s.confidence)
        sorted_sigs = sorted(signals.values(), key=sort_key, reverse=True)
        cols = st.columns(3)
        for idx, sig in enumerate(sorted_sigs):
            name, tier, sector = get_ticker_info(sig.ticker)
            clr = color_for(sig.direction)
            pro = pro_scores.get(sig.ticker)
            grade = pro.total_grade if pro else sig.breakout_grade
            score = pro.total_score if pro else sig.breakout_score
            max_s = 150 if pro else 100
            gc = grade_color(grade)

            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                        f'<div><span style="font-family:\'DM Sans\'; font-size:20px; font-weight:700; color:#F5F5F7;">{sig.ticker}</span>'
                        f'<span style="font-size:10px; color:rgba(255,255,255,0.25); margin-left:8px;">{tier}</span></div>'
                        f'<span class="badge-{sig.direction}">{icon_for(sig.direction)} {sig.direction.title()}</span></div>',
                        unsafe_allow_html=True)
                    chg_c = "#34C759" if sig.change_1d >= 0 else "#FF453A"
                    st.markdown(
                        f'<div style="margin:10px 0;">'
                        f'<span class="mono" style="font-size:24px; font-weight:600; color:#F5F5F7;">${sig.price:,.2f}</span>'
                        f'<span class="mono" style="font-size:13px; color:{chg_c}; margin-left:10px;">{"+" if sig.change_1d>=0 else ""}{sig.change_1d:.2f}%</span></div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="display:flex; gap:12px; font-family:\'JetBrains Mono\'; font-size:12px; align-items:center;">'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Conf</span> <span style="color:{clr}; font-weight:600;">{sig.confidence}%</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">RVol</span> <span style="color:#F5F5F7;">{sig.volume_ratio}x</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Score</span> <span style="color:{gc}; font-weight:700;">{grade} ({score:.0f}/{max_s})</span></div>'
                        f'</div>', unsafe_allow_html=True)
                    top = sig.signals[0] if sig.signals else "—"
                    st.markdown(f'<div style="margin-top:10px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.04);"><div class="sig-item"><span class="sig-dot" style="background:{clr}; box-shadow:0 0 6px {clr}60;"></span>{top}</div></div>', unsafe_allow_html=True)

# ═══ TAB 2: BREAKOUT SCANNER ═══
with tab_breakout:
    if not signals:
        st.info("No data.")
    else:
        if pro_scores:
            st.markdown('<p class="section-label">Pro Breakout Scanner — Base (100) + Pro Factors (50) = Total (150)</p>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.4); margin-bottom:16px; line-height:1.6;">'
                'Rates every ticker on technical patterns <em>and</em> professional trader signals: float size, '
                'float turnover rate, short squeeze potential, recent news catalysts, and intraday structure.'
                '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="section-label">Breakout Scanner (Enable Pro Scoring for full analysis)</p>', unsafe_allow_html=True)

        fc1, fc2 = st.columns([1, 1])
        with fc1:
            max_score = 150 if pro_scores else 100
            default_min = 60 if pro_scores else 30
            min_score = st.slider("Min Score", 0, max_score, default_min, 5)
        with fc2:
            tier_filter = st.radio("Tier", ["All", "Penny/Small", "Large Cap"], horizontal=True)

        def get_score(t):
            return pro_scores[t].total_score if pro_scores else signals[t].breakout_score

        sorted_tickers = sorted(signals.keys(), key=get_score, reverse=True)
        filtered = [t for t in sorted_tickers if get_score(t) >= min_score]
        if tier_filter == "Penny/Small":
            filtered = [t for t in filtered if is_penny_stock(t)]
        elif tier_filter == "Large Cap":
            filtered = [t for t in filtered if not is_penny_stock(t)]

        if not filtered:
            st.info("No tickers match filters.")
        else:
            for t in filtered:
                sig = signals[t]
                pro = pro_scores.get(t)
                profile = profiles.get(t)
                news = news_by_ticker.get(t, [])
                name, tier, sector = get_ticker_info(t)
                clr = color_for(sig.direction)

                grade = pro.total_grade if pro else sig.breakout_grade
                score = pro.total_score if pro else sig.breakout_score
                max_s = 150 if pro else 100
                gc = grade_color(grade)

                with st.container(border=True):
                    # Header
                    hc1, hc2 = st.columns([4, 1])
                    with hc1:
                        profile_text = ""
                        if profile:
                            if profile.float_shares_m > 0:
                                profile_text = f" · Float: {profile.float_shares_m:.1f}M"
                            if profile.short_pct_float > 0:
                                profile_text += f" · Short: {profile.short_pct_float:.1f}%"
                        st.markdown(
                            f'<div style="display:flex; align-items:center; gap:14px;">'
                            f'<span class="grade-badge" style="background:{gc}20; color:{gc}; border:2px solid {gc}40;">{grade}</span>'
                            f'<div>'
                            f'<span style="font-family:\'DM Sans\'; font-size:22px; font-weight:700; color:#F5F5F7;">{t}</span>'
                            f'<span style="font-size:13px; color:rgba(255,255,255,0.35); margin-left:10px;">{name} · {sector}{profile_text}</span>'
                            f'<div style="display:flex; gap:8px; margin-top:4px;">'
                            f'<span class="play-type-badge" style="background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.5);">{tier}</span>'
                            f'<span class="badge-{sig.direction}">{icon_for(sig.direction)} {sig.direction.title()}</span>'
                            f'</div></div></div>', unsafe_allow_html=True)
                    with hc2:
                        if pro:
                            st.markdown(
                                f'<div style="text-align:right;">'
                                f'<div class="kv-label">Total Score</div>'
                                f'<div class="mono" style="font-size:28px; font-weight:700; color:{gc};">{score:.0f}<span style="font-size:14px; color:rgba(255,255,255,0.3);">/{max_s}</span></div>'
                                f'<div style="font-family:\'JetBrains Mono\'; font-size:10px; color:rgba(255,255,255,0.4);">Base {pro.base_score:.0f} + Pro {pro.pro_score:.0f}</div>'
                                f'<div style="height:4px; border-radius:2px; background:rgba(255,255,255,0.06); overflow:hidden; margin-top:6px;">'
                                f'<div style="height:100%; width:{(score/max_s)*100:.0f}%; background:{gc}; border-radius:2px;"></div>'
                                f'</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f'<div style="text-align:right;">'
                                f'<div class="kv-label">Score</div>'
                                f'<div class="mono" style="font-size:32px; font-weight:700; color:{gc};">{score:.0f}</div>'
                                f'</div>', unsafe_allow_html=True)

                    # Key metrics
                    pro_metrics = ""
                    if pro:
                        pro_metrics = (
                            f'<div><span style="color:rgba(255,255,255,0.3);">Turnover</span> <span style="color:#F5F5F7;">{pro.turnover_pct:.1f}%</span></div>'
                            f'<div><span style="color:rgba(255,255,255,0.3);">News</span> <span style="color:#F5F5F7;">{"+" if pro.catalyst_points > 0 else ""}{pro.catalyst_points:.0f}</span></div>'
                        )
                    intraday_stat = intraday_stats_dict.get(t)
                    intraday_txt = ""
                    if intraday_stat:
                        vw_clr = "#34C759" if intraday_stat.above_vwap else "#FF453A"
                        intraday_txt = f'<div><span style="color:rgba(255,255,255,0.3);">VWAP</span> <span style="color:{vw_clr};">{"↑" if intraday_stat.above_vwap else "↓"}</span></div>'

                    st.markdown(
                        f'<div style="display:flex; gap:18px; margin:12px 0; font-family:\'JetBrains Mono\'; font-size:12px; flex-wrap:wrap;">'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Price</span> <span style="color:#F5F5F7; font-weight:600;">${sig.price:,.2f}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">1D</span> <span style="color:{"#34C759" if sig.change_1d>=0 else "#FF453A"};">{sig.change_1d:+.2f}%</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">RVol(5d)</span> <span style="color:{"#34C759" if sig.rvol_5>1.5 else "#F5F5F7"}; font-weight:{"700" if sig.rvol_5>2 else "400"};">{sig.rvol_5:.1f}x</span></div>'
                        f'{pro_metrics}'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Squeeze</span> <span style="color:{"#BF5AF2" if sig.squeeze_on else "rgba(255,255,255,0.3)"};">{"🔥" if sig.squeeze_on else "—"}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">From High</span> <span style="color:#F5F5F7;">{sig.pct_from_high:+.1f}%</span></div>'
                        f'{intraday_txt}'
                        f'<div><span style="color:rgba(255,255,255,0.3);">RSI</span> <span style="color:#F5F5F7;">{sig.rsi:.0f}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Conf</span> <span style="color:{clr};">{sig.confidence}%</span></div>'
                        f'</div>', unsafe_allow_html=True)

                    # Factors (base + pro)
                    impact_colors = {
                        "critical":("#FF453A","rgba(255,69,58,0.15)"),
                        "strong":("#34C759","rgba(52,199,89,0.12)"),
                        "moderate":("#FFD60A","rgba(255,214,10,0.12)"),
                        "weak":("rgba(255,255,255,0.4)","rgba(255,255,255,0.06)"),
                        "negative":("#FF6961","rgba(255,105,97,0.10)"),
                        "neutral":("rgba(255,255,255,0.3)","rgba(255,255,255,0.04)"),
                        "signal":("#BF5AF2","rgba(191,90,242,0.12)"),
                    }
                    all_factors = list(sig.breakout_factors)
                    if pro:
                        all_factors.extend(pro.pro_factors)
                    for factor, desc, impact in all_factors:
                        ic, ib = impact_colors.get(impact, ("rgba(255,255,255,0.4)","rgba(255,255,255,0.06)"))
                        st.markdown(f'<div class="driver-row"><span class="driver-tag" style="background:{ib}; color:{ic};">{factor}</span><span style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.55); line-height:1.5;">{desc}</span></div>', unsafe_allow_html=True)

                    # Recent news inline
                    if news:
                        st.markdown('<p class="section-label" style="margin-top:16px;">Recent Catalysts</p>', unsafe_allow_html=True)
                        for n in news[:3]:
                            sent_c = sentiment_color(n.sentiment)
                            imp_txt = "●" * max(1, n.importance)
                            st.markdown(
                                f'<div class="news-item {n.sentiment}">'
                                f'<div class="news-headline">'
                                f'<span class="imp-badge" style="background:{sent_c}20; color:{sent_c};">{imp_txt} {n.sentiment}</span>'
                                f'{n.headline}</div>'
                                f'<div class="news-meta">{n.source} · {n.datetime_utc.strftime("%Y-%m-%d %H:%M UTC")}</div>'
                                f'</div>', unsafe_allow_html=True)

# ═══ TAB 3: INTRADAY (5m) ═══
with tab_intraday:
    if not enable_intraday:
        st.info("Intraday data is disabled in the sidebar.")
    elif not intraday_stats_dict:
        st.warning("No intraday data available. Market may be closed, or data is still loading.")
    else:
        st.markdown('<p class="section-label">5-Minute Intraday Scanner · Today\'s Session</p>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.4); margin-bottom:16px; line-height:1.6;">'
            'Real-time intraday structure — VWAP, opening range breakouts, and new-day-high detection. '
            'Data refreshes with your selected interval. Sorted by intraday % gain.'
            '</div>', unsafe_allow_html=True)

        # Sort by intraday move
        sorted_stats = sorted(intraday_stats_dict.items(), key=lambda x: x[1].day_change_pct, reverse=True)

        for t, stats in sorted_stats:
            sig = signals.get(t)
            name, tier, sector = get_ticker_info(t)
            chg_c = "#34C759" if stats.day_change_pct >= 0 else "#FF453A"
            vw_c = "#34C759" if stats.above_vwap else "#FF453A"

            with st.container(border=True):
                hc1, hc2, hc3 = st.columns([2, 2, 1])
                with hc1:
                    st.markdown(
                        f'<div>'
                        f'<span style="font-family:\'DM Sans\'; font-size:22px; font-weight:700; color:#F5F5F7;">{t}</span>'
                        f'<span style="font-size:12px; color:rgba(255,255,255,0.35); margin-left:8px;">{name}</span>'
                        f'</div>'
                        f'<div style="margin-top:6px;">'
                        f'<span class="mono" style="font-size:26px; font-weight:600; color:#F5F5F7;">${stats.last_price:,.2f}</span>'
                        f'<span class="mono" style="font-size:14px; color:{chg_c}; margin-left:10px; font-weight:600;">{stats.day_change_pct:+.2f}%</span>'
                        f'</div>', unsafe_allow_html=True)

                with hc2:
                    flags = []
                    if stats.above_vwap:
                        flags.append('<span class="play-type-badge" style="background:rgba(52,199,89,0.15); color:#34C759;">▲ Above VWAP</span>')
                    else:
                        flags.append('<span class="play-type-badge" style="background:rgba(255,69,58,0.15); color:#FF453A;">▼ Below VWAP</span>')
                    if stats.opening_range_break:
                        flags.append('<span class="play-type-badge" style="background:rgba(52,199,89,0.15); color:#34C759;">🔥 ORB Break</span>')
                    if stats.breakout_of_day:
                        flags.append('<span class="play-type-badge" style="background:rgba(191,90,242,0.15); color:#BF5AF2;">📈 New Day High</span>')
                    st.markdown(
                        f'<div style="display:flex; gap:6px; flex-wrap:wrap;">' + "".join(flags) + '</div>'
                        f'<div style="display:flex; gap:14px; margin-top:10px; font-family:\'JetBrains Mono\'; font-size:11px;">'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Open</span> <span style="color:#F5F5F7;">${stats.day_open:.2f}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Hi</span> <span style="color:#F5F5F7;">${stats.day_high:.2f}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Lo</span> <span style="color:#F5F5F7;">${stats.day_low:.2f}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">VWAP</span> <span style="color:{vw_c};">${stats.vwap:.2f}</span></div>'
                        f'</div>', unsafe_allow_html=True)

                with hc3:
                    vol_c = "#34C759" if stats.volume_vs_avg > 1.0 else "#F5F5F7"
                    st.markdown(
                        f'<div style="text-align:right;">'
                        f'<div class="kv-label">Cumul. Volume</div>'
                        f'<div class="mono" style="font-size:16px; font-weight:600; color:{vol_c};">{stats.cumulative_volume/1_000_000:.2f}M</div>'
                        f'<div class="kv-sub">{stats.volume_vs_avg:.2f}x avg</div>'
                        f'<div class="kv-sub" style="margin-top:6px;">AM Range: {stats.morning_range_pct:.1f}%</div>'
                        f'</div>', unsafe_allow_html=True)

                # Mini 5m chart
                df_5m = intraday_dict.get(t)
                if df_5m is not None:
                    df_plot = df_5m.copy()
                    df_plot.index = pd.to_datetime(df_plot.index)
                    df_plot["date"] = df_plot.index.date
                    latest = df_plot["date"].max()
                    today = df_plot[df_plot["date"] == latest].copy()
                    if len(today) >= 2:
                        # Compute VWAP line
                        typ = (today["High"] + today["Low"] + today["Close"]) / 3
                        today["vwap"] = (typ * today["Volume"]).cumsum() / today["Volume"].cumsum().replace(0, np.nan)

                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=idx_to_str(today.index),
                            open=to_list(today, "Open"),
                            high=to_list(today, "High"),
                            low=to_list(today, "Low"),
                            close=to_list(today, "Close"),
                            increasing_line_color="#34C759",
                            decreasing_line_color="#FF453A",
                            increasing_fillcolor="rgba(52,199,89,0.19)",
                            decreasing_fillcolor="rgba(255,69,58,0.19)",
                            name="Price",
                        ))
                        fig.add_trace(go.Scatter(
                            x=idx_to_str(today.index), y=to_list(today, "vwap"),
                            mode="lines", line=dict(color="#0A84FF", width=1.5, dash="dot"),
                            name="VWAP", hovertemplate="VWAP: $%{y:.2f}<extra></extra>",
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="DM Sans, sans-serif", color="rgba(255,255,255,0.5)", size=11),
                            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
                            hoverlabel=dict(bgcolor="#1c1c1e", font_size=12, font_family="JetBrains Mono"),
                            height=220, xaxis_rangeslider_visible=False, showlegend=False,
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"intraday_{t}")

# ═══ TAB 4: DETAIL VIEW ═══
with tab_detail:
    if not signals:
        st.info("No signals.")
    else:
        sel = st.selectbox("Ticker", list(signals.keys()), format_func=lambda t: f"{t} — {get_ticker_info(t)[0]}", key="detail_sel")
        sig = signals[sel]
        pro = pro_scores.get(sel)
        profile = profiles.get(sel)
        news = news_by_ticker.get(sel, [])
        name, tier, sector = get_ticker_info(sel)
        clr = color_for(sig.direction)

        st.markdown(f'<div style="display:flex; align-items:center; gap:14px;"><span style="font-family:\'DM Sans\'; font-size:34px; font-weight:700; color:#F5F5F7;">{sel}</span><span class="badge-{sig.direction}">{icon_for(sig.direction)} {sig.direction.title()}</span></div><p style="font-family:\'DM Sans\'; font-size:14px; color:rgba(255,255,255,0.35);">{name} · {tier} · {sector}</p>', unsafe_allow_html=True)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Price", f"${sig.price:,.2f}")
        m2.metric("1D", f"{sig.change_1d:+.2f}%")
        m3.metric("Confidence", f"{sig.confidence}%")
        m4.metric("RVol(5d)", f"{sig.rvol_5:.1f}x")
        if pro:
            m5.metric("Pro Score", f"{pro.total_grade} ({pro.total_score:.0f})")
        else:
            m5.metric("Score", f"{sig.breakout_grade} ({sig.breakout_score:.0f})")
        m6.metric("Model Acc.", f"{sig.accuracy}%")

        # Profile panel (new)
        if profile and (profile.float_shares_m > 0 or profile.market_cap_m > 0):
            st.markdown('<p class="section-label">Company Profile</p>', unsafe_allow_html=True)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Market Cap", f"${profile.market_cap_m:,.0f}M" if profile.market_cap_m > 0 else "—")
            p2.metric("Float", f"{profile.float_shares_m:.1f}M" if profile.float_shares_m > 0 else "—")
            p3.metric("Short % Float", f"{profile.short_pct_float:.1f}%" if profile.short_pct_float > 0 else "—")
            p4.metric("Short Ratio", f"{profile.short_ratio:.1f}" if profile.short_ratio > 0 else "—")

        # Chart + indicators
        ch_col, in_col = st.columns([2, 1])
        with ch_col:
            df = data_dict.get(sel)
            if df is not None:
                x_vals = idx_to_str(df.index)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
                fig.add_trace(go.Candlestick(
                    x=x_vals,
                    open=to_list(df, "Open"),
                    high=to_list(df, "High"),
                    low=to_list(df, "Low"),
                    close=to_list(df, "Close"),
                    increasing_line_color="#34C759", decreasing_line_color="#FF453A",
                    increasing_fillcolor="rgba(52,199,89,0.19)", decreasing_fillcolor="rgba(255,69,58,0.19)",
                ), row=1, col=1)
                close_arr = to_list(df, "Close")
                open_arr = to_list(df, "Open")
                vol_arr = to_list(df, "Volume")
                vc = ["#34C759" if c >= o else "#FF453A" for c, o in zip(close_arr, open_arr)]
                fig.add_trace(go.Bar(x=x_vals, y=vol_arr, marker_color=vc, opacity=0.4), row=2, col=1)
                fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"detail_{sel}")

        with in_col:
            st.markdown('<p class="section-label">Indicators</p>', unsafe_allow_html=True)
            for label, val in [("RSI (14)", f"{sig.rsi}"), ("RVol (5d)", f"{sig.rvol_5}x"), ("RVol (20d)", f"{sig.rvol_20}x"), ("Volatility (ATR)", f"{sig.volatility}%"), ("MACD Hist", f"{sig.macd_hist:+.4f}"), ("P(up)", f"{sig.probability:.1%}")]:
                st.markdown(f'<div style="padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.04);"><div style="display:flex; justify-content:space-between;"><span style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.5);">{label}</span><span class="mono" style="font-size:14px; font-weight:600; color:#F5F5F7;">{val}</span></div></div>', unsafe_allow_html=True)

            st.markdown('<p class="section-label">Active Signals</p>', unsafe_allow_html=True)
            for s in sig.signals:
                st.markdown(f'<div class="sig-item"><span class="sig-dot" style="background:{clr}; box-shadow:0 0 6px {clr}60;"></span>{s}</div>', unsafe_allow_html=True)

        # News feed (new)
        if news:
            st.markdown('<p class="section-label">Recent News & Catalysts</p>', unsafe_allow_html=True)
            for n in news[:8]:
                sent_c = sentiment_color(n.sentiment)
                imp_txt = "●" * max(1, n.importance)
                link = f'<a href="{n.url}" target="_blank" style="color:rgba(255,255,255,0.3); text-decoration:none; font-size:10px;">↗ open</a>' if n.url else ""
                st.markdown(
                    f'<div class="news-item {n.sentiment}">'
                    f'<div class="news-headline">'
                    f'<span class="imp-badge" style="background:{sent_c}20; color:{sent_c};">{imp_txt} {n.sentiment}</span>'
                    f'{n.headline}</div>'
                    f'<div class="news-meta">{n.source} · {n.datetime_utc.strftime("%Y-%m-%d %H:%M UTC")} · {link}</div>'
                    f'<div class="news-summary">{n.summary[:250]}{"..." if len(n.summary) > 250 else ""}</div>'
                    f'</div>', unsafe_allow_html=True)
        elif FINNHUB_KEY_PRESENT:
            st.caption("No recent news available for this ticker.")
        else:
            st.caption("⚠ Add a FINNHUB_API_KEY in Streamlit secrets to enable news feed.")

# ═══ TAB 5: OPTIONS PLAYS ═══
with tab_options:
    if not signals:
        st.info("No signals.")
    else:
        oc1, oc2, oc3 = st.columns([2, 1, 1])
        with oc1:
            ot = st.selectbox("Ticker", list(signals.keys()), format_func=lambda t: f"{t} — {signals[t].direction.title()} ({signals[t].confidence}%)", key="opt_sel")
        with oc2:
            rf = st.selectbox("Risk", ["All", "Conservative", "Moderate", "Aggressive"], key="opt_risk")
        with oc3:
            tf = st.selectbox("Type", ["All", "Directional", "Neutral", "Volatility"], key="opt_type")

        @st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice, 120), show_spinner=False)
        def get_plays(ticker, sig_hash, lb):
            s = signals.get(ticker)
            d = data_dict.get(ticker)
            return generate_options_plays(s, d) if s and d is not None else []

        sig = signals[ot]
        plays = get_plays(ot, f"{ot}_{sig.confidence}_{sig.probability}", lookback)
        if rf != "All": plays = [p for p in plays if p.risk_tier == rf.lower()]
        if tf != "All":
            if tf == "Directional": plays = [p for p in plays if "directional" in p.strategy_type]
            else: plays = [p for p in plays if p.strategy_type == tf.lower()]

        if not plays:
            st.warning(f"No plays for {ot} with current filters.")
        else:
            for pi, play in enumerate(plays):
                tc = {"directional_bullish":("#34C759","rgba(52,199,89,0.12)"),"directional_bearish":("#FF453A","rgba(255,69,58,0.12)"),"neutral":("#0A84FF","rgba(10,132,255,0.12)"),"volatility":("#BF5AF2","rgba(191,90,242,0.12)")}
                t_clr, t_bg = tc.get(play.strategy_type, ("#FFD60A","rgba(255,214,10,0.12)"))
                cc = "bullish" if "bullish" in play.strategy_type else "bearish" if "bearish" in play.strategy_type else "volatility" if play.strategy_type == "volatility" else "neutral"
                st.markdown(f'<div class="play-card {cc}">', unsafe_allow_html=True)
                pc1, pc2 = st.columns([3, 1])
                with pc1:
                    st.markdown(f'<div class="play-strategy">{play.strategy_name}</div><div style="display:flex; gap:8px; margin-top:4px;"><span class="play-type-badge" style="background:{t_bg}; color:{t_clr};">{play.strategy_type.replace("_"," ")}</span><span class="play-type-badge" style="background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.5);">{play.risk_tier}</span></div>', unsafe_allow_html=True)
                with pc2:
                    st.markdown(f'<div style="text-align:right;"><div class="kv-label">Prob of Profit</div><div class="mono" style="font-size:28px; font-weight:700; color:{t_clr};">{play.probability_of_profit}%</div></div>', unsafe_allow_html=True)

                pt = play.price_target
                fr = pt.target_high - pt.target_low
                cp = max(5, min(95, ((pt.current - pt.target_low) / fr * 100) if fr > 0 else 50))
                mp = max(5, min(95, ((pt.target_mid - pt.target_low) / fr * 100) if fr > 0 else 50))
                st.markdown(f'<p class="section-label">90% CI Target</p><div class="ci-bar-container"><div class="ci-bar-fill" style="left:10%; right:10%; background:{t_clr};"></div><div class="ci-marker" style="left:10%; color:{t_clr}; top:25%;">${pt.target_low:.0f}</div><div class="ci-marker" style="left:{cp}%; color:#F5F5F7; top:75%;">NOW ${pt.current:.0f}</div><div class="ci-marker" style="left:90%; color:{t_clr}; top:25%;">${pt.target_high:.0f}</div></div>', unsafe_allow_html=True)

                st.markdown('<p class="section-label">Legs</p>', unsafe_allow_html=True)
                for leg in play.legs:
                    dc = "#34C759" if leg.direction == "buy" else "#FF6961"
                    db = "rgba(52,199,89,0.15)" if leg.direction == "buy" else "rgba(255,105,97,0.15)"
                    st.markdown(f'<div class="leg-row"><span class="leg-direction" style="background:{db}; color:{dc};">{leg.direction}</span><span style="color:#F5F5F7; font-weight:600;">{leg.option_type.upper()}</span><span style="color:rgba(255,255,255,0.4);">K</span><span style="color:#F5F5F7; font-weight:600;">${leg.strike:.0f}</span><span style="color:rgba(255,255,255,0.4);">Prem</span><span style="color:#F5F5F7;">${leg.estimated_premium:.2f}</span><span style="color:rgba(255,255,255,0.3); margin-left:auto;">Δ{leg.delta:.2f} Γ{leg.gamma:.3f} Θ{leg.theta:.3f}</span></div>', unsafe_allow_html=True)

                st.markdown('<p class="section-label">Trade Plan</p>', unsafe_allow_html=True)
                is_cr = play.strategy_type == "neutral" or "Short" in play.strategy_name
                gi = [(("Credit" if is_cr else "Debit"), f"${play.entry_price:.2f}"), ("Max Loss", f"${play.max_loss:.0f}"), ("Max Gain", "Unlimited" if play.max_gain == -1 else f"${play.max_gain:.0f}"), ("R/R", f"{play.risk_reward_ratio:.1f}x"), ("BE", f"${play.break_even:.2f}"), ("PoP", f"{play.probability_of_profit}%")]
                gh = '<div class="kv-grid">'
                for l, v in gi:
                    gh += f'<div class="kv-cell"><div class="kv-label">{l}</div><div class="kv-value">{v}</div></div>'
                st.markdown(gh + '</div>', unsafe_allow_html=True)

                st.markdown(f'<p class="section-label">Timing</p><div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;"><div class="kv-cell"><div class="kv-label">Entry</div><div style="font-size:13px; color:rgba(255,255,255,0.6); margin-top:4px;">{play.entry_timing}</div></div><div class="kv-cell"><div class="kv-label">Exit</div><div style="font-size:13px; color:rgba(255,255,255,0.6); margin-top:4px;">{play.exit_timing}</div></div></div>', unsafe_allow_html=True)

                # ── Hold Duration & Theta ──────────────────────
                if play.hold_duration:
                    st.markdown('<p class="section-label">How Long to Hold</p>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="display:grid; grid-template-columns:1fr 2fr; gap:12px;">'
                        f'<div class="kv-cell" style="display:flex; flex-direction:column; justify-content:center; align-items:center;">'
                        f'<div class="kv-label">Recommended Hold</div>'
                        f'<div class="mono" style="font-size:22px; font-weight:700; color:{t_clr}; margin-top:6px;">{play.hold_duration}</div>'
                        f'</div>'
                        f'<div class="kv-cell">'
                        f'<div style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.6); line-height:1.6;">{play.hold_reasoning}</div>'
                        f'</div></div>', unsafe_allow_html=True)

                    # Theta warning
                    if play.theta_decay_warning:
                        st.markdown(
                            f'<div style="background:rgba(255,159,10,0.08); border:1px solid rgba(255,159,10,0.2); border-radius:10px; padding:14px 18px; margin-top:10px;">'
                            f'<div style="font-family:\'JetBrains Mono\'; font-size:10px; font-weight:600; color:#FF9F0A; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">⏳ Theta Decay Impact</div>'
                            f'<div style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.6); line-height:1.6;">{play.theta_decay_warning}</div>'
                            f'</div>', unsafe_allow_html=True)

                    # Optimal exit
                    if play.optimal_exit_scenario:
                        st.markdown(
                            f'<div style="background:rgba(52,199,89,0.06); border:1px solid rgba(52,199,89,0.15); border-radius:10px; padding:14px 18px; margin-top:10px;">'
                            f'<div style="font-family:\'JetBrains Mono\'; font-size:10px; font-weight:600; color:#34C759; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">🎯 Optimal Exit Scenario</div>'
                            f'<div style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.6); line-height:1.6;">{play.optimal_exit_scenario}</div>'
                            f'</div>', unsafe_allow_html=True)

                # ── Price Movement Insights ────────────────────
                if play.price_drivers:
                    st.markdown('<p class="section-label">What\'s Driving the Price</p>', unsafe_allow_html=True)
                    for factor, emoji, desc in play.price_drivers:
                        st.markdown(
                            f'<div class="driver-row">'
                            f'<span style="font-size:16px; flex-shrink:0;">{emoji}</span>'
                            f'<div>'
                            f'<div style="font-family:\'DM Sans\'; font-size:13px; font-weight:600; color:#F5F5F7;">{factor}</div>'
                            f'<div style="font-family:\'DM Sans\'; font-size:12px; color:rgba(255,255,255,0.5); margin-top:2px; line-height:1.5;">{desc}</div>'
                            f'</div></div>', unsafe_allow_html=True)

                st.markdown(f'<p class="section-label">Thesis</p><div class="thesis-block">{play.thesis}</div>', unsafe_allow_html=True)

                st.markdown('<p class="section-label">Technical Drivers</p>', unsafe_allow_html=True)
                dic = {"primary":("#0A84FF","rgba(10,132,255,0.12)"),"strong":("#34C759","rgba(52,199,89,0.12)"),"supportive":("#30D158","rgba(48,209,88,0.10)"),"moderate":("#FFD60A","rgba(255,214,10,0.12)"),"caution":("#FF9F0A","rgba(255,159,10,0.12)"),"neutral":("rgba(255,255,255,0.4)","rgba(255,255,255,0.06)"),"context":("#BF5AF2","rgba(191,90,242,0.12)"),"signal":("#5AC8FA","rgba(90,200,250,0.12)")}
                for f, d, i in play.drivers:
                    ic, ib = dic.get(i, ("rgba(255,255,255,0.4)","rgba(255,255,255,0.06)"))
                    st.markdown(f'<div class="driver-row"><span class="driver-tag" style="background:{ib}; color:{ic};">{f}</span><span style="font-size:13px; color:rgba(255,255,255,0.55);">{d}</span></div>', unsafe_allow_html=True)

                st.markdown('<p class="section-label">Risks</p>', unsafe_allow_html=True)
                for r in play.risks:
                    st.markdown(f'<div class="risk-item"><span style="color:#FF9F0A;">⚠</span>{r}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ═══ TAB 6: HEATMAP ═══
with tab_heatmap:
    if not signals:
        st.info("No data.")
    else:
        def get_score(t):
            return pro_scores[t].total_score if pro_scores else signals[t].breakout_score
        def get_grade(t):
            return pro_scores[t].total_grade if pro_scores else signals[t].breakout_grade

        ts = sorted(signals.keys(), key=get_score, reverse=True)
        max_y = 150 if pro_scores else 100

        st.markdown('<p class="section-label">Breakout Score Heatmap</p>', unsafe_allow_html=True)
        scores = [get_score(t) for t in ts]
        sc = [grade_color(get_grade(t)) for t in ts]
        fig = go.Figure(go.Bar(x=ts, y=scores, marker_color=sc, text=[f"{s:.0f}" for s in scores], textposition="outside", textfont=dict(family="JetBrains Mono", size=11, color="rgba(255,255,255,0.5)")))
        fig.update_layout(**{**PLOTLY_LAYOUT, "yaxis": dict(range=[0, max_y * 1.05], gridcolor="rgba(255,255,255,0.04)")}, height=350)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="heatmap_main")

        st.markdown('<p class="section-label">Full Dashboard</p>', unsafe_allow_html=True)
        td = []
        for t in ts:
            s = signals[t]
            n, tier, sec = get_ticker_info(t)
            pro = pro_scores.get(t)
            profile = profiles.get(t)
            row = {
                "Ticker": t, "Name": n, "Tier": tier, "Sector": sec,
                "Dir": s.direction.title(), "Conf": f"{s.confidence}%",
                "Price": f"${s.price:,.2f}", "1D": f"{s.change_1d:+.2f}%",
                "RVol(5)": f"{s.rvol_5:.1f}x",
                "Float": f"{profile.float_shares_m:.1f}M" if profile and profile.float_shares_m > 0 else "—",
                "Short%": f"{profile.short_pct_float:.1f}%" if profile and profile.short_pct_float > 0 else "—",
                "Squeeze": "🔥" if s.squeeze_on else "—",
                "Score": f"{get_score(t):.0f}",
                "Grade": get_grade(t),
            }
            if t in intraday_stats_dict:
                row["VWAP"] = "▲" if intraday_stats_dict[t].above_vwap else "▼"
                row["Intraday"] = f"{intraday_stats_dict[t].day_change_pct:+.2f}%"
            td.append(row)
        st.dataframe(pd.DataFrame(td), use_container_width=True, hide_index=True, height=500)

st.markdown('<div style="height:32px"></div><p style="font-family:sans-serif; font-size:11px; color:rgba(255,255,255,0.12); text-align:center; padding:24px 0; line-height:1.6;">Signal · yfinance (price/intraday/float) + Finnhub (news/profile). Educational use only. Not financial advice.</p>', unsafe_allow_html=True)
