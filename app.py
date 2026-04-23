"""
Signal — Professional Trading Analytics (Pro Edition)
======================================================
Rebuilt with card-based grid layout, always-on data (no market-hours gating),
runner pattern detection for penny stocks, and visual polish.
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
from utils.catalysts import fetch_news, fetch_company_profile, _get_api_key
from utils.intraday import fetch_intraday_batch, compute_intraday_stats, is_market_hours
from models.predictor import predict_batch_parallel, Signal
from models.pro_scorer import compute_pro_breakout
from models.options_engine import generate_options_plays

st.set_page_config(page_title="Signal Pro", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")
FINNHUB_KEY_PRESENT = bool(_get_api_key())

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◉ Signal Config")
    st.markdown("---")
    refresh_choice = st.selectbox("⏱ Refresh", list(REFRESH_INTERVALS.keys()), index=2)
    st.markdown("---")
    ticker_mode = st.radio("Watchlist", ["All", "Large Cap", "Penny/Small", "Custom"])
    if ticker_mode == "All": selected_tickers = DEFAULT_TICKERS
    elif ticker_mode == "Large Cap": selected_tickers = LARGE_CAP_TICKERS
    elif ticker_mode == "Penny/Small": selected_tickers = PENNY_TICKERS
    else: selected_tickers = st.multiselect("Custom", options=DEFAULT_TICKERS, default=DEFAULT_TICKERS[:8])
    st.markdown("---")
    lookback = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y"], index=1)
    enable_pro = st.checkbox("Pro Scoring", value=True)
    enable_intraday = st.checkbox("Intraday Data", value=True)
    st.markdown("---")
    if FINNHUB_KEY_PRESENT: st.success("✓ Finnhub connected")
    else:
        st.warning("⚠ No Finnhub key")
        with st.expander("Setup"):
            st.markdown("Get free key at **finnhub.io** → add `FINNHUB_API_KEY` in Streamlit secrets")

if HAS_AUTOREFRESH:
    st_autorefresh(interval=REFRESH_INTERVALS[refresh_choice], limit=None, key="data_refresh")

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root { --bg-card: rgba(255,255,255,0.03); --border: rgba(255,255,255,0.06); --text-primary: #F5F5F7; --text-secondary: rgba(255,255,255,0.45); --text-muted: rgba(255,255,255,0.25); --green: #34C759; --red: #FF453A; --yellow: #FFD60A; --blue: #0A84FF; --purple: #BF5AF2; }
.stApp { background: linear-gradient(180deg, #000 0%, #070710 50%, #0d0d15 100%) !important; }
#MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { background: #08080e !important; border-right: 1px solid var(--border) !important; }
div[data-testid="stMetric"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 16px 18px !important; }
div[data-testid="stMetric"] label { font-family: 'DM Sans' !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; color: var(--text-muted) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono' !important; font-weight: 600 !important; font-size: 22px !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; border: 1px solid var(--border) !important; background: var(--bg-card) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans' !important; font-size: 14px !important; font-weight: 500 !important; color: var(--text-secondary) !important; border-radius: 8px 8px 0 0 !important; padding: 10px 18px !important; }
.stTabs [aria-selected="true"] { color: var(--text-primary) !important; background: var(--bg-card) !important; }
div[data-baseweb="select"] > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.sig-h { font-family:'DM Sans'; font-size:38px; font-weight:700; letter-spacing:-0.03em; color:#F5F5F7; margin:0; }
.sig-sub { font-family:'DM Sans'; font-size:13px; color:rgba(255,255,255,0.3); letter-spacing:0.1em; text-transform:uppercase; margin-top:4px; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-family:'DM Sans'; font-size:11px; font-weight:600; }
.badge-bull { background:rgba(52,199,89,0.12); color:#34C759; }
.badge-bear { background:rgba(255,69,58,0.12); color:#FF453A; }
.badge-neut { background:rgba(255,214,10,0.12); color:#FFD60A; }
.badge-squeeze { background:rgba(191,90,242,0.15); color:#BF5AF2; }
.badge-rvol { background:rgba(52,199,89,0.12); color:#34C759; }
.badge-runner { background:linear-gradient(135deg, rgba(255,69,58,0.2), rgba(191,90,242,0.2)); color:#FF6961; border:1px solid rgba(255,69,58,0.2); }
.mono { font-family:'JetBrains Mono' !important; }
.live-badge { display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono'; font-size:11px; color:rgba(255,255,255,0.4); }
.live-dot { width:6px; height:6px; border-radius:50%; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.section-label { font-family:'DM Sans'; font-size:11px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px; margin-top:20px; }
.driver-row { display:flex; align-items:flex-start; gap:10px; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.driver-tag { font-family:'JetBrains Mono'; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; padding:2px 8px; border-radius:4px; white-space:nowrap; flex-shrink:0; margin-top:2px; }
.grade-badge { display:inline-flex; align-items:center; justify-content:center; width:44px; height:44px; border-radius:12px; font-family:'JetBrains Mono'; font-size:18px; font-weight:700; }
.news-card { padding:10px 14px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; margin-bottom:6px; }
.news-card.bullish { border-left:3px solid #34C759; }
.news-card.bearish { border-left:3px solid #FF453A; }
.kv-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
.kv-cell { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:12px 14px; }
.kv-label { font-family:'DM Sans'; font-size:10px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }
.kv-value { font-family:'JetBrains Mono'; font-size:16px; font-weight:600; color:#F5F5F7; }
.kv-sub { font-family:'DM Sans'; font-size:10px; color:rgba(255,255,255,0.2); margin-top:2px; }
.runner-score-bar { height:8px; border-radius:4px; background:rgba(255,255,255,0.06); overflow:hidden; margin-top:6px; }
.runner-score-fill { height:100%; border-radius:4px; background:linear-gradient(90deg, #FF453A, #BF5AF2, #34C759); }
.play-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:24px; margin-bottom:18px; position:relative; overflow:hidden; }
.play-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:18px 18px 0 0; }
.play-card.bullish::before { background:linear-gradient(90deg,#34C759,#30D158); }
.play-card.bearish::before { background:linear-gradient(90deg,#FF453A,#FF6961); }
.play-card.neutral::before { background:linear-gradient(90deg,#0A84FF,#5AC8FA); }
.play-card.volatility::before { background:linear-gradient(90deg,#BF5AF2,#FF6FF1); }
.thesis-block { background:rgba(255,255,255,0.02); border-left:3px solid rgba(255,255,255,0.1); padding:14px 18px; border-radius:0 10px 10px 0; font-family:'DM Sans'; font-size:13px; color:rgba(255,255,255,0.6); line-height:1.7; }
.ci-bar-container { position:relative; height:36px; background:rgba(255,255,255,0.03); border-radius:8px; border:1px solid rgba(255,255,255,0.06); overflow:hidden; margin:10px 0; }
.ci-bar-fill { position:absolute; top:0; bottom:0; border-radius:8px; opacity:0.2; }
.ci-marker { position:absolute; top:50%; transform:translate(-50%,-50%); font-family:'JetBrains Mono'; font-size:10px; font-weight:600; white-space:nowrap; }
.risk-item { display:flex; align-items:flex-start; gap:8px; padding:5px 0; font-family:'DM Sans'; font-size:12px; color:rgba(255,255,255,0.5); line-height:1.5; }
.leg-row { display:flex; align-items:center; gap:10px; padding:8px 12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:8px; margin-bottom:4px; font-family:'JetBrains Mono'; font-size:12px; }
.leg-dir { font-weight:700; font-size:10px; text-transform:uppercase; letter-spacing:0.06em; padding:2px 6px; border-radius:4px; }
</style>""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────
def color_for(d): return {"bullish":"#34C759","bearish":"#FF453A"}.get(d,"#FFD60A")
def icon_for(d): return {"bullish":"↑","bearish":"↓"}.get(d,"→")
def grade_color(g):
    if g.startswith("A"): return "#34C759"
    if g.startswith("B"): return "#0A84FF"
    if g.startswith("C"): return "#FFD60A"
    return "#FF453A"
def sentiment_color(s): return {"bullish":"#34C759","bearish":"#FF453A"}.get(s,"rgba(255,255,255,0.3)")
def to_list(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    return [float(v) for v in s.values.ravel()]
def idx_to_str(index):
    try: return [t.isoformat() for t in index]
    except: return list(index)

def compute_runner_score(sig, profile=None, news=None):
    """
    Model what drives a penny from $0.30 → $20+.
    The pattern that repeats in every multi-bagger runner:
    1. Micro/low float (supply scarcity)
    2. Volume explosion (demand surge)
    3. Catalyst (news ignition)
    4. Short squeeze fuel
    5. Technical compression (coiled spring)
    6. Momentum ignition (breakout confirmation)
    """
    pts = 0
    factors = []

    # 1. FLOAT SCARCITY (0-25) — THE #1 DRIVER
    fm = profile.float_shares_m if profile and profile.float_shares_m > 0 else 0
    if 0 < fm < 10:
        pts += 25
        factors.append(("🔥 Micro Float", f"{fm:.1f}M shares — extreme scarcity. This is the #1 ingredient in every $0.30→$20 runner. Low supply + any demand = explosive moves.", "#FF453A"))
    elif fm < 30:
        pts += 18
        factors.append(("Low Float", f"{fm:.1f}M shares — strong scarcity. Low floats account for 80% of multi-day runners.", "#FFD60A"))
    elif fm < 100:
        pts += 8
        factors.append(("Medium Float", f"{fm:.1f}M — moderate. Needs heavier volume to move significantly.", "rgba(255,255,255,0.5)"))
    elif fm > 0:
        factors.append(("Large Float", f"{fm:.0f}M — too much supply for explosive runner dynamics.", "rgba(255,255,255,0.3)"))

    # 2. VOLUME EXPLOSION (0-25) — THE IGNITION
    rv5 = sig.rvol_5
    rv20 = sig.rvol_20
    if rv5 >= 3.0:
        pts += 25
        factors.append(("🔥 Volume Explosion", f"RVol {rv5:.1f}x — this is the ignition. Every $0.30→$20 move starts with a 3x+ volume day. Smart money is entering.", "#FF453A"))
    elif rv5 >= 2.0:
        pts += 18
        factors.append(("Volume Surge", f"RVol {rv5:.1f}x — significant institutional interest. Volume precedes price in every runner.", "#34C759"))
    elif rv5 >= 1.5:
        pts += 10
        factors.append(("Volume Rising", f"RVol {rv5:.1f}x — early accumulation phase. Watch for acceleration.", "#FFD60A"))
    elif rv20 >= 1.3:
        pts += 5
        factors.append(("Volume Warming", f"20d RVol {rv20:.1f}x — slow build. Not yet runner territory.", "rgba(255,255,255,0.5)"))
    else:
        factors.append(("Dead Volume", f"RVol {rv5:.1f}x — no interest. Runners don't start without volume.", "rgba(255,255,255,0.3)"))

    # 3. CATALYST / NEWS (0-20)
    if news:
        from datetime import timedelta
        recent_bull = [n for n in news if n.sentiment == "bullish" and (datetime.utcnow() - n.datetime_utc).total_seconds() < 72*3600]
        high_imp = [n for n in recent_bull if n.importance >= 2]
        if len(high_imp) >= 2:
            pts += 20
            factors.append(("🔥 Major Catalyst", f"{len(high_imp)} high-impact bullish catalysts — this is the spark. FDA, contract, acquisition news drives 10x+ moves.", "#FF453A"))
        elif len(high_imp) == 1:
            pts += 14
            factors.append(("Catalyst Active", f"Key bullish news: {high_imp[0].headline[:70]}...", "#34C759"))
        elif len(recent_bull) >= 2:
            pts += 8
            factors.append(("Positive Flow", f"{len(recent_bull)} bullish headlines — building narrative momentum.", "#FFD60A"))
        elif recent_bull:
            pts += 4
            factors.append(("Mild Positive", "Some bullish coverage, but no high-impact catalyst yet.", "rgba(255,255,255,0.5)"))
        else:
            factors.append(("No Catalyst", "No bullish news — runners need a story to attract retail buying.", "rgba(255,255,255,0.3)"))
    else:
        factors.append(("News N/A", "Set FINNHUB_API_KEY for catalyst detection.", "rgba(255,255,255,0.25)"))

    # 4. SHORT SQUEEZE FUEL (0-15)
    sp = profile.short_pct_float if profile else 0
    if sp >= 25:
        pts += 15
        factors.append(("🔥 Squeeze Primed", f"{sp:.1f}% short — this is rocket fuel. When price moves up, shorts are forced to cover, creating a self-reinforcing loop.", "#FF453A"))
    elif sp >= 15:
        pts += 10
        factors.append(("High Short", f"{sp:.1f}% short — squeeze candidate if catalyst hits.", "#FFD60A"))
    elif sp >= 8:
        pts += 4
        factors.append(("Moderate Short", f"{sp:.1f}% short — some squeeze potential.", "rgba(255,255,255,0.5)"))
    elif sp > 0:
        factors.append(("Low Short", f"{sp:.1f}% — minimal squeeze dynamics.", "rgba(255,255,255,0.3)"))

    # 5. COMPRESSION (0-10) — COILED SPRING
    if sig.squeeze_on:
        pts += 10
        factors.append(("🔥 Squeeze Loaded", "BB inside Keltner — the spring is coiled. Every multi-bagger runner starts from a tight compression zone.", "#BF5AF2"))
    elif sig.range_compression < 8:
        pts += 6
        factors.append(("Tight Range", f"10-day range only {sig.range_compression:.1f}% — building energy for expansion.", "#FFD60A"))
    else:
        factors.append(("Wide Range", f"Range {sig.range_compression:.1f}% — not compressed. May need consolidation first.", "rgba(255,255,255,0.3)"))

    # 6. MOMENTUM IGNITION (0-10)
    if sig.direction == "bullish" and sig.momentum > 3 and sig.rsi < 70:
        pts += 10
        factors.append(("Momentum Igniting", f"ROC +{sig.momentum:.1f}%, RSI {sig.rsi:.0f} — breakout in progress with room to run. This is the confirmation phase.", "#34C759"))
    elif sig.direction == "bullish" and sig.momentum > 0:
        pts += 5
        factors.append(("Momentum Building", f"ROC +{sig.momentum:.1f}% — early stages of a potential move.", "#FFD60A"))
    elif sig.rsi < 30:
        pts += 3
        factors.append(("Oversold Bounce", f"RSI {sig.rsi:.0f} — potential snap-back rally. Some runners start from deep oversold conditions.", "#FFD60A"))
    else:
        factors.append(("No Momentum", f"ROC {sig.momentum:+.1f}%, RSI {sig.rsi:.0f} — no directional ignition.", "rgba(255,255,255,0.3)"))

    pts = min(100, pts)
    grade = "A+" if pts >= 85 else "A" if pts >= 70 else "B+" if pts >= 60 else "B" if pts >= 50 else "C" if pts >= 35 else "D" if pts >= 20 else "F"
    return pts, grade, factors

# ─── Header ──────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<p class="sig-h">Signal</p>', unsafe_allow_html=True)
    st.markdown('<p class="sig-sub">Professional Trading Analytics</p>', unsafe_allow_html=True)
with h2:
    mkt = is_market_hours()
    mkt_c = "#34C759" if mkt else "#FF9F0A"
    mkt_t = "MKT OPEN" if mkt else "AFTER HOURS"
    st.markdown(f'<div style="text-align:right; padding-top:16px;"><span class="live-badge"><span class="live-dot" style="background:{mkt_c};"></span> {mkt_t} · {refresh_choice}</span><br><span class="mono" style="font-size:11px; color:rgba(255,255,255,0.2);">{datetime.now().strftime("%H:%M:%S")}</span></div>', unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── Data Pipeline (ALWAYS runs — no market gating) ──────────────
@st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice, 120), show_spinner=False)
def load_and_predict(tickers, lookback):
    data = fetch_batch_data(tickers, period=lookback)
    sigs = predict_batch_parallel(data)
    return data, sigs

@st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice, 120), show_spinner=False)
def load_intraday(tickers):
    return fetch_intraday_batch(tickers, days=5)

progress = st.progress(0, text="Loading market data...")
data_dict, signals = load_and_predict(tuple(selected_tickers), lookback)
progress.progress(40, text="Models complete.")

intraday_dict, intraday_stats_dict = {}, {}
if enable_intraday:
    progress.progress(55, text="Fetching intraday bars...")
    intraday_dict = load_intraday(tuple(selected_tickers))
    for t, df in intraday_dict.items():
        avg_vol = float(data_dict[t]["Volume"].tail(20).mean()) if t in data_dict else 0
        stats = compute_intraday_stats(t, df, avg_daily_vol=avg_vol)
        if stats: intraday_stats_dict[t] = stats
    progress.progress(70)

pro_scores, profiles, news_by_ticker = {}, {}, {}
if enable_pro:
    progress.progress(80, text="Pro scoring...")
    for ticker, sig in signals.items():
        profile = fetch_company_profile(ticker)
        news = fetch_news(ticker, days_back=5, max_items=8) if FINNHUB_KEY_PRESENT else []
        profiles[ticker] = profile
        news_by_ticker[ticker] = news
        daily_vol = float(data_dict[ticker]["Volume"].iloc[-1]) if ticker in data_dict else 0
        pro = compute_pro_breakout(sig.breakout_score, ticker, daily_vol, profile, news, intraday_stats_dict.get(ticker))
        pro_scores[ticker] = pro

progress.progress(100, text="Ready.")
progress.empty()

# ─── Metrics ─────────────────────────────────────────────────────
bull = sum(1 for s in signals.values() if s.direction == "bullish")
bear = sum(1 for s in signals.values() if s.direction == "bearish")
neut = len(signals) - bull - bear
avg_score = np.mean([pro_scores[t].total_score for t in pro_scores]) if pro_scores else (np.mean([s.breakout_score for s in signals.values()]) if signals else 0)
penny_runners = sum(1 for t in signals if is_penny_stock(t) and compute_runner_score(signals[t], profiles.get(t), news_by_ticker.get(t, []))[0] >= 50)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Bullish", bull)
c2.metric("Bearish", bear)
c3.metric("Neutral", neut)
c4.metric("Avg Score", f"{avg_score:.0f}")
c5.metric("Runner Alerts", penny_runners, delta="Score ≥ 50")
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab_overview, tab_penny, tab_breakout, tab_detail, tab_options, tab_heatmap = st.tabs([
    "◉ Overview", "🚀 Penny Runners", "🔥 Breakout Scanner",
    "◎ Detail + News", "⬡ Options", "◈ Heatmap",
])

# ═══ TAB 1: OVERVIEW (card grid) ═══
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
            gc = grade_color(grade)
            badge_cls = "badge-bull" if sig.direction == "bullish" else "badge-bear" if sig.direction == "bearish" else "badge-neut"
            chg_c = "#34C759" if sig.change_1d >= 0 else "#FF453A"
            intra = intraday_stats_dict.get(sig.ticker)
            vwap_html = ""
            if intra:
                vw = "▲" if intra.above_vwap else "▼"
                vw_c = "#34C759" if intra.above_vwap else "#FF453A"
                vwap_html = f'<span style="color:{vw_c}; font-weight:600;">{vw}</span> '

            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
                        f'<div><span style="font-family:\'DM Sans\'; font-size:20px; font-weight:700; color:#F5F5F7;">{sig.ticker}</span>'
                        f'<span style="font-size:10px; color:rgba(255,255,255,0.2); margin-left:6px;">{tier}</span></div>'
                        f'<span class="badge {badge_cls}">{icon_for(sig.direction)} {sig.direction.title()}</span></div>'
                        f'<div style="margin:8px 0;">'
                        f'<span class="mono" style="font-size:22px; font-weight:600; color:#F5F5F7;">${sig.price:,.2f}</span>'
                        f'<span class="mono" style="font-size:12px; color:{chg_c}; margin-left:8px;">{"+" if sig.change_1d>=0 else ""}{sig.change_1d:.2f}%</span></div>'
                        f'<div style="display:flex; gap:10px; font-family:\'JetBrains Mono\'; font-size:11px;">'
                        f'{vwap_html}'
                        f'<span style="color:rgba(255,255,255,0.3);">Conf</span><span style="color:{clr}; font-weight:600;">{sig.confidence}%</span> '
                        f'<span style="color:rgba(255,255,255,0.3);">RVol</span><span style="color:#F5F5F7;">{sig.volume_ratio}x</span> '
                        f'<span style="color:{gc}; font-weight:700;">{grade}</span></div>',
                        unsafe_allow_html=True)
                    top = sig.signals[0] if sig.signals else "—"
                    st.markdown(f'<div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.04); font-family:\'DM Sans\'; font-size:12px; color:rgba(255,255,255,0.5);">{top}</div>', unsafe_allow_html=True)

# ═══ TAB 2: PENNY RUNNERS (card grid with runner pattern analysis) ═══
with tab_penny:
    penny_sigs = {t: s for t, s in signals.items() if is_penny_stock(t)}
    if not penny_sigs:
        st.warning("No penny stocks in watchlist. Select 'All' or 'Penny/Small' in sidebar.")
    else:
        mkt_label = "Live Market" if mkt else "After-Hours Prep"
        st.markdown(f'<p class="section-label">🚀 {mkt_label} · Penny Runner Scanner</p>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'DM Sans\'; font-size:13px; color:rgba(255,255,255,0.4); margin-bottom:16px; line-height:1.7;">'
            'Models what drives small stocks from <span style="color:#FF453A; font-weight:600;">$0.30 → $20+</span>: '
            '<strong style="color:rgba(255,255,255,0.6);">float scarcity</strong> (supply), '
            '<strong style="color:rgba(255,255,255,0.6);">volume explosion</strong> (demand), '
            '<strong style="color:rgba(255,255,255,0.6);">news catalyst</strong> (ignition), '
            '<strong style="color:rgba(255,255,255,0.6);">short squeeze fuel</strong> (forced buying), '
            '<strong style="color:rgba(255,255,255,0.6);">compression</strong> (coiled spring), '
            '<strong style="color:rgba(255,255,255,0.6);">momentum ignition</strong> (confirmation). '
            'Scanned continuously — even after hours.'
            '</div>', unsafe_allow_html=True)

        # Compute runner scores
        runner_data = {}
        for t, s in penny_sigs.items():
            pts, grade, factors = compute_runner_score(s, profiles.get(t), news_by_ticker.get(t, []))
            runner_data[t] = (pts, grade, factors)

        sorted_pennies = sorted(runner_data.keys(), key=lambda t: runner_data[t][0], reverse=True)

        # Summary metrics
        pc1, pc2, pc3, pc4 = st.columns(4)
        high_runners = sum(1 for t in runner_data if runner_data[t][0] >= 60)
        squeezes = sum(1 for t in penny_sigs.values() if t.squeeze_on)
        high_rvol = sum(1 for t in penny_sigs.values() if t.rvol_5 > 2)
        best = sorted_pennies[0] if sorted_pennies else "—"
        pc1.metric("Runner Setups", high_runners, delta="Score ≥ 60")
        pc2.metric("Squeezes", squeezes)
        pc3.metric("Vol Explosions", high_rvol, delta="RVol > 2x")
        pc4.metric("Top Pick", best)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # CARD GRID — 2 columns side by side
        cols = st.columns(2)
        for idx, t in enumerate(sorted_pennies):
            sig = penny_sigs[t]
            pts, grade, factors = runner_data[t]
            profile = profiles.get(t)
            news = news_by_ticker.get(t, [])
            intra = intraday_stats_dict.get(t)
            name, tier, sector = get_ticker_info(t)
            clr = color_for(sig.direction)
            gc = grade_color(grade)
            chg_c = "#34C759" if sig.change_1d >= 0 else "#FF453A"

            with cols[idx % 2]:
                with st.container(border=True):
                    # Header with grade + price
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
                        f'<div style="display:flex; align-items:center; gap:10px;">'
                        f'<span class="grade-badge" style="background:{gc}18; color:{gc}; border:2px solid {gc}35;">{grade}</span>'
                        f'<div>'
                        f'<span style="font-family:\'DM Sans\'; font-size:20px; font-weight:700; color:#F5F5F7;">{t}</span>'
                        f'<span style="font-size:11px; color:rgba(255,255,255,0.3); margin-left:8px;">{name}</span>'
                        f'<div style="display:flex; gap:4px; margin-top:3px; flex-wrap:wrap;">'
                        f'<span class="badge {"badge-bull" if sig.direction == "bullish" else "badge-bear" if sig.direction == "bearish" else "badge-neut"}">{icon_for(sig.direction)} {sig.direction.title()}</span>'
                        + (' <span class="badge badge-squeeze">🔥 Squeeze</span>' if sig.squeeze_on else '')
                        + (f' <span class="badge badge-rvol">⚡ {sig.rvol_5:.1f}x Vol</span>' if sig.rvol_5 > 1.5 else '')
                        + (' <span class="badge badge-runner">🚀 Runner Setup</span>' if pts >= 60 else '')
                        + f'</div></div></div>'
                        f'<div style="text-align:right;">'
                        f'<div class="mono" style="font-size:22px; font-weight:600; color:#F5F5F7;">${sig.price:,.2f}</div>'
                        f'<div class="mono" style="font-size:12px; color:{chg_c};">{sig.change_1d:+.2f}%</div>'
                        f'</div></div>', unsafe_allow_html=True)

                    # Runner Score Bar
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">'
                        f'<span style="font-family:\'DM Sans\'; font-size:10px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.06em;">Runner Score</span>'
                        f'<span class="mono" style="font-size:14px; font-weight:700; color:{gc};">{pts}/100</span></div>'
                        f'<div class="runner-score-bar"><div class="runner-score-fill" style="width:{pts}%;"></div></div>',
                        unsafe_allow_html=True)

                    # Key metrics row
                    float_txt = f"{profile.float_shares_m:.0f}M" if profile and profile.float_shares_m > 0 else "—"
                    short_txt = f"{profile.short_pct_float:.1f}%" if profile and profile.short_pct_float > 0 else "—"
                    st.markdown(
                        f'<div style="display:flex; gap:12px; margin:10px 0; font-family:\'JetBrains Mono\'; font-size:11px; flex-wrap:wrap;">'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Float</span> <span style="color:#F5F5F7;">{float_txt}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Short</span> <span style="color:#F5F5F7;">{short_txt}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">RSI</span> <span style="color:#F5F5F7;">{sig.rsi:.0f}</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">5D</span> <span style="color:{"#34C759" if sig.change_5d>=0 else "#FF453A"};">{sig.change_5d:+.1f}%</span></div>'
                        f'<div><span style="color:rgba(255,255,255,0.3);">Conf</span> <span style="color:{clr};">{sig.confidence}%</span></div>'
                        f'</div>', unsafe_allow_html=True)

                    # Runner factors (the core value prop)
                    for fname, fdesc, fcolor in factors[:4]:
                        st.markdown(
                            f'<div style="display:flex; align-items:flex-start; gap:8px; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.03);">'
                            f'<span style="font-family:\'JetBrains Mono\'; font-size:10px; font-weight:600; color:{fcolor}; white-space:nowrap; margin-top:2px;">{fname}</span>'
                            f'<span style="font-family:\'DM Sans\'; font-size:11px; color:rgba(255,255,255,0.5); line-height:1.5;">{fdesc}</span>'
                            f'</div>', unsafe_allow_html=True)

                    # Inline news (top 2 headlines)
                    if news:
                        bull_news = [n for n in news if n.sentiment == "bullish"][:2]
                        for n in bull_news:
                            hrs = (datetime.utcnow() - n.datetime_utc).total_seconds() / 3600
                            fresh = "🔴" if hrs < 12 else "🟡" if hrs < 48 else ""
                            st.markdown(
                                f'<div class="news-card bullish" style="margin-top:6px;">'
                                f'<div style="font-family:\'DM Sans\'; font-size:12px; font-weight:600; color:#F5F5F7;">{fresh} {n.headline[:80]}{"..." if len(n.headline)>80 else ""}</div>'
                                f'<div style="font-family:\'JetBrains Mono\'; font-size:9px; color:rgba(255,255,255,0.3);">{n.source} · {hrs:.0f}h ago</div>'
                                f'</div>', unsafe_allow_html=True)

        # Summary table
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Penny Runner Ranking</p>', unsafe_allow_html=True)
        tbl = []
        for t in sorted_pennies:
            s = penny_sigs[t]
            pts, grade, _ = runner_data[t]
            pf = profiles.get(t)
            nw = news_by_ticker.get(t, [])
            tbl.append({"Ticker":t, "Price":f"${s.price:,.2f}", "1D":f"{s.change_1d:+.2f}%", "5D":f"{s.change_5d:+.2f}%",
                "RVol":f"{s.rvol_5:.1f}x", "Squeeze":"🔥" if s.squeeze_on else "—",
                "Float":f"{pf.float_shares_m:.0f}M" if pf and pf.float_shares_m>0 else "—",
                "Short%":f"{pf.short_pct_float:.1f}%" if pf and pf.short_pct_float>0 else "—",
                "News":f"{sum(1 for n in nw if n.sentiment=='bullish')}↑" if nw else "—",
                "Runner":f"{pts}", "Grade":grade})
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True, height=min(400, 40+len(tbl)*38))

# ═══ TAB 3: BREAKOUT SCANNER ═══
with tab_breakout:
    if not signals:
        st.info("No data.")
    else:
        st.markdown('<p class="section-label">Breakout Scanner — All Tickers</p>', unsafe_allow_html=True)
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            max_sc = 150 if pro_scores else 100
            min_score = st.slider("Min Score", 0, max_sc, 30, 5)
        with fc2:
            tf = st.radio("Tier", ["All", "Penny", "Large Cap"], horizontal=True)

        def get_score(t):
            return pro_scores[t].total_score if t in pro_scores else signals[t].breakout_score
        sorted_t = sorted(signals.keys(), key=get_score, reverse=True)
        filtered = [t for t in sorted_t if get_score(t) >= min_score]
        if tf == "Penny": filtered = [t for t in filtered if is_penny_stock(t)]
        elif tf == "Large Cap": filtered = [t for t in filtered if not is_penny_stock(t)]

        if not filtered:
            st.info("No tickers match.")
        else:
            cols = st.columns(2)
            for idx, t in enumerate(filtered):
                sig = signals[t]
                pro = pro_scores.get(t)
                name, _, sector = get_ticker_info(t)
                clr = color_for(sig.direction)
                grade = pro.total_grade if pro else sig.breakout_grade
                score = get_score(t)
                gc = grade_color(grade)
                with cols[idx % 2]:
                    with st.container(border=True):
                        st.markdown(
                            f'<div style="display:flex; justify-content:space-between;">'
                            f'<div style="display:flex; align-items:center; gap:10px;">'
                            f'<span class="grade-badge" style="background:{gc}18; color:{gc}; border:2px solid {gc}35;">{grade}</span>'
                            f'<div><span style="font-family:\'DM Sans\'; font-size:18px; font-weight:700; color:#F5F5F7;">{t}</span>'
                            f'<span style="font-size:11px; color:rgba(255,255,255,0.3); margin-left:8px;">{name}</span></div></div>'
                            f'<div style="text-align:right;">'
                            f'<div class="mono" style="font-size:20px; font-weight:600; color:#F5F5F7;">${sig.price:,.2f}</div>'
                            f'<div class="mono" style="font-size:11px; color:{"#34C759" if sig.change_1d>=0 else "#FF453A"};">{sig.change_1d:+.2f}%</div></div></div>'
                            f'<div style="display:flex; gap:10px; margin-top:8px; font-family:\'JetBrains Mono\'; font-size:11px;">'
                            f'<span style="color:rgba(255,255,255,0.3);">Score</span><span style="color:{gc}; font-weight:700;">{score:.0f}</span> '
                            f'<span style="color:rgba(255,255,255,0.3);">RVol</span><span>{sig.rvol_5:.1f}x</span> '
                            f'<span style="color:rgba(255,255,255,0.3);">RSI</span><span>{sig.rsi:.0f}</span> '
                            f'<span style="color:rgba(255,255,255,0.3);">Conf</span><span style="color:{clr};">{sig.confidence}%</span></div>',
                            unsafe_allow_html=True)
                        top = sig.signals[0] if sig.signals else "—"
                        st.markdown(f'<div style="margin-top:6px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.04); font-family:\'DM Sans\'; font-size:11px; color:rgba(255,255,255,0.45);">{top}</div>', unsafe_allow_html=True)

# ═══ TAB 4: DETAIL + NEWS ═══
with tab_detail:
    if not signals:
        st.info("No signals.")
    else:
        sel = st.selectbox("Ticker", list(signals.keys()), format_func=lambda t: f"{t} — {get_ticker_info(t)[0]}", key="detail_sel")
        sig = signals[sel]
        profile = profiles.get(sel)
        news = news_by_ticker.get(sel, [])
        name, tier, sector = get_ticker_info(sel)
        clr = color_for(sig.direction)
        pro = pro_scores.get(sel)

        st.markdown(f'<div style="display:flex; align-items:center; gap:14px;"><span style="font-family:\'DM Sans\'; font-size:32px; font-weight:700; color:#F5F5F7;">{sel}</span><span class="badge {"badge-bull" if sig.direction=="bullish" else "badge-bear" if sig.direction=="bearish" else "badge-neut"}">{icon_for(sig.direction)} {sig.direction.title()}</span></div><p style="font-family:\'DM Sans\'; font-size:14px; color:rgba(255,255,255,0.35);">{name} · {tier} · {sector}</p>', unsafe_allow_html=True)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Price", f"${sig.price:,.2f}")
        m2.metric("1D", f"{sig.change_1d:+.2f}%")
        m3.metric("Confidence", f"{sig.confidence}%")
        m4.metric("RVol", f"{sig.rvol_5:.1f}x")
        grade = pro.total_grade if pro else sig.breakout_grade
        score = pro.total_score if pro else sig.breakout_score
        m5.metric("Score", f"{grade} ({score:.0f})")
        m6.metric("Accuracy", f"{sig.accuracy}%")

        if profile and profile.float_shares_m > 0:
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Float", f"{profile.float_shares_m:.1f}M")
            p2.metric("Mkt Cap", f"${profile.market_cap_m:,.0f}M")
            p3.metric("Short%", f"{profile.short_pct_float:.1f}%")
            p4.metric("Short Ratio", f"{profile.short_ratio:.1f}")

        chart_col, info_col = st.columns([2, 1])
        with chart_col:
            df = data_dict.get(sel)
            if df is not None:
                x_vals = idx_to_str(df.index)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
                fig.add_trace(go.Candlestick(x=x_vals, open=to_list(df,"Open"), high=to_list(df,"High"), low=to_list(df,"Low"), close=to_list(df,"Close"), increasing_line_color="#34C759", decreasing_line_color="#FF453A", increasing_fillcolor="rgba(52,199,89,0.19)", decreasing_fillcolor="rgba(255,69,58,0.19)"), row=1, col=1)
                c_arr, o_arr, v_arr = to_list(df,"Close"), to_list(df,"Open"), to_list(df,"Volume")
                vc = ["#34C759" if c>=o else "#FF453A" for c,o in zip(c_arr, o_arr)]
                fig.add_trace(go.Bar(x=x_vals, y=v_arr, marker_color=vc, opacity=0.4), row=2, col=1)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="rgba(255,255,255,0.5)", size=11), margin=dict(l=0,r=0,t=30,b=0), xaxis=dict(gridcolor="rgba(255,255,255,0.04)"), yaxis=dict(gridcolor="rgba(255,255,255,0.04)"), height=400, showlegend=False, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"detail_chart_{sel}")

        with info_col:
            for label, val in [("RSI", f"{sig.rsi}"), ("RVol (5d)", f"{sig.rvol_5}x"), ("Volatility", f"{sig.volatility}%"), ("MACD Hist", f"{sig.macd_hist:+.4f}"), ("P(up)", f"{sig.probability:.1%}")]:
                st.markdown(f'<div style="padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04); display:flex; justify-content:space-between;"><span style="font-family:\'DM Sans\'; font-size:12px; color:rgba(255,255,255,0.5);">{label}</span><span class="mono" style="font-size:13px; font-weight:600; color:#F5F5F7;">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Signals</p>', unsafe_allow_html=True)
            for s in sig.signals:
                st.markdown(f'<div style="display:flex; align-items:center; gap:8px; padding:4px 0; font-family:\'DM Sans\'; font-size:12px; color:rgba(255,255,255,0.6);"><span style="width:5px; height:5px; border-radius:50%; background:{clr}; box-shadow:0 0 4px {clr}60; flex-shrink:0;"></span>{s}</div>', unsafe_allow_html=True)

        # News section
        if news:
            st.markdown('<p class="section-label">News & Catalysts</p>', unsafe_allow_html=True)
            ncols = st.columns(2)
            for ni, n in enumerate(news[:6]):
                sc = sentiment_color(n.sentiment)
                hrs = (datetime.utcnow() - n.datetime_utc).total_seconds() / 3600
                fresh = "🔴" if hrs < 12 else "🟡" if hrs < 48 else ""
                link = f' <a href="{n.url}" target="_blank" style="color:rgba(255,255,255,0.25); text-decoration:none; font-size:9px;">↗</a>' if n.url else ""
                with ncols[ni % 2]:
                    st.markdown(
                        f'<div class="news-card {n.sentiment}">'
                        f'<div style="font-family:\'DM Sans\'; font-size:12px; font-weight:600; color:#F5F5F7; line-height:1.4;">'
                        f'<span style="font-family:\'JetBrains Mono\'; font-size:9px; padding:1px 5px; border-radius:3px; background:{sc}18; color:{sc}; margin-right:6px;">{n.sentiment}</span>'
                        f'{fresh} {n.headline[:90]}{"..." if len(n.headline)>90 else ""}{link}</div>'
                        f'<div style="font-family:\'JetBrains Mono\'; font-size:9px; color:rgba(255,255,255,0.25); margin-top:4px;">{n.source} · {hrs:.0f}h ago</div>'
                        f'</div>', unsafe_allow_html=True)
        elif not FINNHUB_KEY_PRESENT:
            st.caption("Set FINNHUB_API_KEY for news.")

# ═══ TAB 5: OPTIONS ═══
with tab_options:
    if not signals:
        st.info("No signals.")
    else:
        oc1,oc2 = st.columns([2,1])
        with oc1:
            ot = st.selectbox("Ticker", list(signals.keys()), format_func=lambda t: f"{t} — {signals[t].direction.title()} ({signals[t].confidence}%)", key="opt_sel")
        with oc2:
            rf = st.selectbox("Risk", ["All","Conservative","Moderate","Aggressive"], key="opt_risk")

        @st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice, 120), show_spinner=False)
        def get_plays(ticker, sig_hash, lb):
            s, d = signals.get(ticker), data_dict.get(ticker)
            return generate_options_plays(s, d) if s and d is not None else []

        sig = signals[ot]
        plays = get_plays(ot, f"{ot}_{sig.confidence}_{sig.probability}", lookback)
        if rf != "All": plays = [p for p in plays if p.risk_tier == rf.lower()]

        if not plays:
            st.warning(f"No plays for {ot}.")
        else:
            for pi, play in enumerate(plays):
                tc = {"directional_bullish":("#34C759","rgba(52,199,89,0.12)"),"directional_bearish":("#FF453A","rgba(255,69,58,0.12)"),"neutral":("#0A84FF","rgba(10,132,255,0.12)"),"volatility":("#BF5AF2","rgba(191,90,242,0.12)")}
                t_clr,t_bg = tc.get(play.strategy_type,("#FFD60A","rgba(255,214,10,0.12)"))
                cc = "bullish" if "bullish" in play.strategy_type else "bearish" if "bearish" in play.strategy_type else "volatility" if play.strategy_type=="volatility" else "neutral"
                st.markdown(f'<div class="play-card {cc}">', unsafe_allow_html=True)
                pc1,pc2 = st.columns([3,1])
                with pc1:
                    st.markdown(f'<div style="font-family:\'DM Sans\'; font-size:22px; font-weight:700; color:#F5F5F7;">{play.strategy_name}</div><div style="display:flex; gap:6px; margin-top:4px;"><span class="badge" style="background:{t_bg}; color:{t_clr}; font-family:\'JetBrains Mono\'; font-size:10px;">{play.strategy_type.replace("_"," ")}</span><span class="badge" style="background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.5); font-family:\'JetBrains Mono\'; font-size:10px;">{play.risk_tier}</span></div>', unsafe_allow_html=True)
                with pc2:
                    st.markdown(f'<div style="text-align:right;"><div class="kv-label">PoP</div><div class="mono" style="font-size:26px; font-weight:700; color:{t_clr};">{play.probability_of_profit}%</div></div>', unsafe_allow_html=True)

                # CI bar
                pt = play.price_target
                fr = pt.target_high - pt.target_low
                cp = max(5, min(95, ((pt.current-pt.target_low)/fr*100) if fr>0 else 50))
                st.markdown(f'<div class="ci-bar-container"><div class="ci-bar-fill" style="left:10%; right:10%; background:{t_clr};"></div><div class="ci-marker" style="left:10%; color:{t_clr}; top:25%;">${pt.target_low:.0f}</div><div class="ci-marker" style="left:{cp}%; color:#F5F5F7; top:75%;">NOW ${pt.current:.0f}</div><div class="ci-marker" style="left:90%; color:{t_clr}; top:25%;">${pt.target_high:.0f}</div></div>', unsafe_allow_html=True)

                # Legs
                for leg in play.legs:
                    dc = "#34C759" if leg.direction=="buy" else "#FF6961"
                    db = "rgba(52,199,89,0.15)" if leg.direction=="buy" else "rgba(255,105,97,0.15)"
                    st.markdown(f'<div class="leg-row"><span class="leg-dir" style="background:{db}; color:{dc};">{leg.direction}</span><span style="color:#F5F5F7; font-weight:600;">{leg.option_type.upper()}</span><span style="color:rgba(255,255,255,0.4);">K</span><span style="color:#F5F5F7;">${leg.strike:.0f}</span><span style="color:rgba(255,255,255,0.4);">Prem</span><span style="color:#F5F5F7;">${leg.estimated_premium:.2f}</span><span style="color:rgba(255,255,255,0.3); margin-left:auto;">Δ{leg.delta:.2f} Θ{leg.theta:.3f}</span></div>', unsafe_allow_html=True)

                # Trade plan grid
                is_cr = play.strategy_type=="neutral" or "Short" in play.strategy_name
                gi = [(("Credit" if is_cr else "Debit"),f"${play.entry_price:.2f}"),("Max Loss",f"${play.max_loss:.0f}"),("Max Gain","∞" if play.max_gain==-1 else f"${play.max_gain:.0f}"),("R/R",f"{play.risk_reward_ratio:.1f}x"),("BE",f"${play.break_even:.2f}"),("PoP",f"{play.probability_of_profit}%")]
                gh = '<div class="kv-grid">'
                for l,v in gi: gh += f'<div class="kv-cell"><div class="kv-label">{l}</div><div class="kv-value">{v}</div></div>'
                st.markdown(gh+'</div>', unsafe_allow_html=True)

                # Hold duration
                if play.hold_duration:
                    st.markdown(
                        f'<div style="display:grid; grid-template-columns:auto 1fr; gap:12px; margin-top:12px;">'
                        f'<div class="kv-cell" style="text-align:center;"><div class="kv-label">Hold</div><div class="mono" style="font-size:18px; font-weight:700; color:{t_clr};">{play.hold_duration}</div></div>'
                        f'<div class="kv-cell"><div style="font-family:\'DM Sans\'; font-size:12px; color:rgba(255,255,255,0.55); line-height:1.6;">{play.hold_reasoning}</div></div></div>', unsafe_allow_html=True)

                if play.theta_decay_warning:
                    st.markdown(f'<div style="background:rgba(255,159,10,0.06); border:1px solid rgba(255,159,10,0.15); border-radius:8px; padding:10px 14px; margin-top:8px;"><span style="font-family:\'JetBrains Mono\'; font-size:9px; font-weight:600; color:#FF9F0A; text-transform:uppercase;">⏳ Theta</span> <span style="font-family:\'DM Sans\'; font-size:12px; color:rgba(255,255,255,0.55);">{play.theta_decay_warning}</span></div>', unsafe_allow_html=True)

                if play.price_drivers:
                    st.markdown('<p class="section-label">Price Drivers</p>', unsafe_allow_html=True)
                    for factor, emoji, desc in play.price_drivers[:4]:
                        st.markdown(f'<div class="driver-row"><span style="font-size:14px;">{emoji}</span><div><div style="font-family:\'DM Sans\'; font-size:12px; font-weight:600; color:#F5F5F7;">{factor}</div><div style="font-family:\'DM Sans\'; font-size:11px; color:rgba(255,255,255,0.45); margin-top:1px;">{desc}</div></div></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="thesis-block" style="margin-top:10px;">{play.thesis}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ═══ TAB 6: HEATMAP ═══
with tab_heatmap:
    if not signals:
        st.info("No data.")
    else:
        def gs(t): return pro_scores[t].total_score if t in pro_scores else signals[t].breakout_score
        def gg(t): return pro_scores[t].total_grade if t in pro_scores else signals[t].breakout_grade
        ts = sorted(signals.keys(), key=gs, reverse=True)
        max_y = 150 if pro_scores else 100

        scores = [gs(t) for t in ts]
        sc = [grade_color(gg(t)) for t in ts]
        fig = go.Figure(go.Bar(x=ts, y=scores, marker_color=sc, text=[f"{s:.0f}" for s in scores], textposition="outside", textfont=dict(family="JetBrains Mono", size=11, color="rgba(255,255,255,0.5)")))
        fig.update_layout(**{**dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="rgba(255,255,255,0.5)", size=11), margin=dict(l=0,r=0,t=30,b=0), xaxis=dict(gridcolor="rgba(255,255,255,0.04)")), "yaxis": dict(range=[0,max_y*1.05], gridcolor="rgba(255,255,255,0.04)")}, height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="heatmap_bar")

        td = []
        for t in ts:
            s = signals[t]
            n, tier, sec = get_ticker_info(t)
            pf = profiles.get(t)
            intra = intraday_stats_dict.get(t)
            row = {"Ticker":t, "Name":n, "Dir":s.direction.title(), "Price":f"${s.price:,.2f}",
                "1D":f"{s.change_1d:+.2f}%", "RVol":f"{s.rvol_5:.1f}x",
                "Float":f"{pf.float_shares_m:.0f}M" if pf and pf.float_shares_m>0 else "—",
                "Short%":f"{pf.short_pct_float:.1f}%" if pf and pf.short_pct_float>0 else "—",
                "Score":f"{gs(t):.0f}", "Grade":gg(t), "RSI":f"{s.rsi:.0f}"}
            if intra: row["VWAP"] = "▲" if intra.above_vwap else "▼"
            td.append(row)
        st.dataframe(pd.DataFrame(td), use_container_width=True, hide_index=True, height=450)

st.markdown('<div style="height:24px"></div><p style="font-family:sans-serif; font-size:10px; color:rgba(255,255,255,0.1); text-align:center; padding:16px 0;">Signal · yfinance + Finnhub · Educational use only. Not financial advice.</p>', unsafe_allow_html=True)
