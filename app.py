"""
Signal — Professional Trading Analytics (Final Build)
• Live penny stock screener (scans ~80 seed tickers, filters by price each session)
• Runner = stock under $1 for 12+ months
• King node heatmap (volume-at-price profile with HVN detection)
• Card grid layout, always-on, Plotly 6.x + Python 3.14 safe
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
    HAS_AR = True
except ImportError:
    HAS_AR = False

from utils.data import (
    fetch_batch_data, get_ticker_info, classify_stock,
    LARGE_CAP_TICKERS, PENNY_TICKERS_FALLBACK, REFRESH_INTERVALS, CACHE_TTL_MAP,
)
from utils.screener import get_live_penny_tickers, get_scan_results, SEED_UNIVERSE
from utils.catalysts import fetch_news, fetch_company_profile, _get_api_key
from utils.intraday import fetch_intraday_batch, compute_intraday_stats, is_market_hours
from models.predictor import predict_batch_parallel
from models.pro_scorer import compute_pro_breakout
from models.options_engine import generate_options_plays
from models.king_nodes import compute_volume_profile

st.set_page_config(page_title="Signal Pro", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
FK = bool(_get_api_key())

# ═══ SIDEBAR ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◉ Signal Config")
    st.markdown("---")
    refresh_choice = st.selectbox("⏱ Refresh", list(REFRESH_INTERVALS.keys()), index=2)
    trader_profile = st.selectbox("Trader Profile", ["Balanced", "Conservative", "Aggressive"], index=0)
    st.markdown("---")

    # Live penny scanner runs automatically
    st.markdown("**Penny Stock Scanner**")
    max_penny_price = st.slider("Max penny price ($)", 1.0, 10.0, 5.0, 0.5)
    live_pennies = get_live_penny_tickers(max_price=max_penny_price)
    scan_data = get_scan_results()
    st.caption(f"Found {len(live_pennies)} stocks under ${max_penny_price:.0f} from {len(SEED_UNIVERSE)} scanned")

    st.markdown("---")
    ticker_mode = st.radio("Watchlist", ["All (Large + Live Pennies)", "Large Cap Only", "Live Pennies Only", "Custom"])
    if ticker_mode == "All (Large + Live Pennies)":
        selected_tickers = list(set(LARGE_CAP_TICKERS + live_pennies))
    elif ticker_mode == "Large Cap Only":
        selected_tickers = LARGE_CAP_TICKERS
    elif ticker_mode == "Live Pennies Only":
        selected_tickers = live_pennies if live_pennies else PENNY_TICKERS_FALLBACK
    else:
        all_avail = list(set(LARGE_CAP_TICKERS + live_pennies + PENNY_TICKERS_FALLBACK))
        selected_tickers = st.multiselect("Custom", all_avail, all_avail[:12])
    st.markdown("---")
    lookback = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y"], index=1)
    enable_pro = st.checkbox("Pro Scoring", True)
    enable_intraday = st.checkbox("Intraday Data", True)
    compact_mode = st.checkbox("Compact Cards", True)
    st.markdown("---")
    if FK: st.success("✓ Finnhub connected")
    else:
        st.warning("⚠ No Finnhub key")
        with st.expander("Setup"): st.markdown("Get free key at **finnhub.io** → add `FINNHUB_API_KEY` in Streamlit secrets")
    st.markdown("---")
    st.markdown("**Definitions**")
    st.markdown("• **Penny stock** = price < $5")
    st.markdown("• **Runner** = under $1 for 12+ months")
    st.markdown("• **King node** = price level with abnormally high traded volume (HVN)")
    st.caption("Profile changes reliability thresholds and execution strictness.")

if HAS_AR:
    st_autorefresh(interval=REFRESH_INTERVALS[refresh_choice], limit=None, key="ar")

# ═══ CSS ══════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root{--card:rgba(255,255,255,0.03);--bdr:rgba(255,255,255,0.06);--t1:#F5F5F7;--t2:rgba(255,255,255,0.45);--t3:rgba(255,255,255,0.25);--g:#34C759;--r:#FF453A;--y:#FFD60A;--bl:#0A84FF;--p:#BF5AF2;}
.stApp{background:linear-gradient(180deg,#000 0%,#070710 50%,#0d0d15 100%)!important;}
#MainMenu,footer,header,.stDeployButton,div[data-testid="stToolbar"],div[data-testid="stDecoration"]{display:none!important;}
section[data-testid="stSidebar"]{background:#08080e!important;border-right:1px solid var(--bdr)!important;}
div[data-testid="stMetric"]{background:var(--card)!important;border:1px solid var(--bdr)!important;border-radius:12px!important;padding:12px 14px!important;}
div[data-testid="stMetric"] label{font-family:'DM Sans'!important;font-size:10px!important;text-transform:uppercase!important;letter-spacing:0.08em!important;color:var(--t3)!important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{font-family:'JetBrains Mono'!important;font-weight:600!important;font-size:20px!important;}
div[data-testid="stHorizontalBlock"]>div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px!important;border:1px solid var(--bdr)!important;background:var(--card)!important;}
.stTabs [data-baseweb="tab-list"]{gap:3px!important;border-bottom:1px solid var(--bdr)!important;}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans'!important;font-size:13px!important;font-weight:500!important;color:var(--t2)!important;border-radius:8px 8px 0 0!important;padding:8px 14px!important;}
.stTabs [aria-selected="true"]{color:var(--t1)!important;background:var(--card)!important;}
div[data-baseweb="select"]>div{background:var(--card)!important;border:1px solid var(--bdr)!important;border-radius:8px!important;}
.sh{font-family:'DM Sans';font-size:34px;font-weight:700;letter-spacing:-0.03em;color:#F5F5F7;margin:0;}
.ss{font-family:'DM Sans';font-size:11px;color:rgba(255,255,255,0.3);letter-spacing:0.1em;text-transform:uppercase;margin-top:3px;}
.b{display:inline-block;padding:2px 8px;border-radius:14px;font-family:'DM Sans';font-size:10px;font-weight:600;}
.bg{background:rgba(52,199,89,0.12);color:#34C759;}.br{background:rgba(255,69,58,0.12);color:#FF453A;}.bn{background:rgba(255,214,10,0.12);color:#FFD60A;}
.bsq{background:rgba(191,90,242,0.15);color:#BF5AF2;}.bvol{background:rgba(52,199,89,0.12);color:#34C759;}
.brun{background:linear-gradient(135deg,rgba(255,69,58,0.2),rgba(191,90,242,0.2));color:#FF6961;border:1px solid rgba(255,69,58,0.15);}
.bsub{background:rgba(255,159,10,0.15);color:#FF9F0A;}
.bking{background:rgba(10,132,255,0.15);color:#0A84FF;border:1px solid rgba(10,132,255,0.2);}
.mono{font-family:'JetBrains Mono'!important;}
.lb{display:inline-flex;align-items:center;gap:6px;font-family:'JetBrains Mono';font-size:10px;color:rgba(255,255,255,0.4);}
.ld{width:6px;height:6px;border-radius:50%;animation:p 2s infinite;}
@keyframes p{0%,100%{opacity:1;}50%{opacity:0.4;}}
.sl{font-family:'DM Sans';font-size:10px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;margin-top:16px;}
.dr{display:flex;align-items:flex-start;gap:7px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);}
.dt{font-family:'JetBrains Mono';font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;padding:2px 6px;border-radius:4px;white-space:nowrap;flex-shrink:0;margin-top:1px;}
.gb{display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:10px;font-family:'JetBrains Mono';font-size:15px;font-weight:700;}
.nc{padding:7px 10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:7px;margin-bottom:4px;}
.nc.bullish{border-left:3px solid #34C759;}.nc.bearish{border-left:3px solid #FF453A;}
.sb{height:6px;border-radius:3px;background:rgba(255,255,255,0.06);overflow:hidden;margin-top:4px;}
.sf{height:100%;border-radius:3px;background:linear-gradient(90deg,#FF453A,#BF5AF2,#34C759);}
.kg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;}
.kc{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:7px;padding:8px 10px;}
.kl{font-family:'DM Sans';font-size:9px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px;}
.kv{font-family:'JetBrains Mono';font-size:14px;font-weight:600;color:#F5F5F7;}
.pc{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:18px;margin-bottom:12px;position:relative;overflow:hidden;}
.pc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;}
.pc.bullish::before{background:linear-gradient(90deg,#34C759,#30D158);}
.pc.bearish::before{background:linear-gradient(90deg,#FF453A,#FF6961);}
.pc.neutral::before{background:linear-gradient(90deg,#0A84FF,#5AC8FA);}
.pc.volatility::before{background:linear-gradient(90deg,#BF5AF2,#FF6FF1);}
.tb{background:rgba(255,255,255,0.02);border-left:3px solid rgba(255,255,255,0.1);padding:10px 14px;border-radius:0 7px 7px 0;font-family:'DM Sans';font-size:11px;color:rgba(255,255,255,0.6);line-height:1.5;}
.lr{display:flex;align-items:center;gap:7px;padding:6px 9px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:6px;margin-bottom:3px;font-family:'JetBrains Mono';font-size:10px;}
</style>""", unsafe_allow_html=True)

# ═══ HELPERS ══════════════════════════════════════════════════════
def clr(d): return {"bullish":"#34C759","bearish":"#FF453A"}.get(d,"#FFD60A")
def ico(d): return {"bullish":"↑","bearish":"↓"}.get(d,"→")
def gc(g):
    if g.startswith("A"): return "#34C759"
    if g.startswith("B"): return "#0A84FF"
    if g.startswith("C"): return "#FFD60A"
    return "#FF453A"
def sc(s): return {"bullish":"#34C759","bearish":"#FF453A"}.get(s,"rgba(255,255,255,0.3)")
def to_list(df,c):
    s=df[c]
    if isinstance(s,pd.DataFrame):s=s.iloc[:,0]
    return [float(v) for v in s.values.ravel()]
def idx_str(ix):
    try: return [t.isoformat() for t in ix]
    except: return list(ix)

PROFILE_CFG = {
    "Conservative": {"reliability_shift": 8, "distance_penalty": 5.0, "near_node_pct": 2.0},
    "Balanced": {"reliability_shift": 0, "distance_penalty": 6.0, "near_node_pct": 3.0},
    "Aggressive": {"reliability_shift": -6, "distance_penalty": 7.5, "near_node_pct": 4.0},
}
profile_cfg = PROFILE_CFG.get(trader_profile, PROFILE_CFG["Balanced"])

def runner_score(sig, cl, pf=None, nw=None):
    pts=0;facs=[]
    mu1=cl.get("months_under_1",0);cp=cl.get("current_price",999)
    if cp<1 and mu1>=12: pts+=20;facs.append(("🔥 True Runner",f"Under $1 for {mu1}mo at ${cp:.3f} — classic runner base","#FF453A"))
    elif cp<1 and mu1>=6: pts+=14;facs.append(("Runner Building",f"Under $1 for {mu1}mo at ${cp:.3f}","#FFD60A"))
    elif cp<3 and mu1>=6: pts+=8;facs.append(("Former Sub-$1",f"Was sub-$1 for {mu1}mo, now ${cp:.2f}","#FFD60A"))
    elif cp<5: pts+=4;facs.append(("Penny Stock",f"${cp:.2f} — penny dynamics","rgba(255,255,255,0.5)"))
    else: facs.append(("Not Penny",f"${cp:.2f}","rgba(255,255,255,0.3)"))
    fm=pf.float_shares_m if pf and pf.float_shares_m>0 else 0
    if 0<fm<15: pts+=20;facs.append(("🔥 Micro Float",f"{fm:.1f}M — extreme scarcity","#FF453A"))
    elif fm<50: pts+=12;facs.append(("Low Float",f"{fm:.1f}M","#FFD60A"))
    elif fm<200: pts+=5;facs.append(("Med Float",f"{fm:.0f}M","rgba(255,255,255,0.5)"))
    rv=sig.rvol_5
    if rv>=3: pts+=20;facs.append(("🔥 Vol Explosion",f"RVol {rv:.1f}x","#FF453A"))
    elif rv>=2: pts+=14;facs.append(("Vol Surge",f"RVol {rv:.1f}x","#34C759"))
    elif rv>=1.5: pts+=8;facs.append(("Vol Rising",f"RVol {rv:.1f}x","#FFD60A"))
    if nw:
        from datetime import timedelta
        rb=[n for n in nw if n.sentiment=="bullish" and (datetime.utcnow()-n.datetime_utc).total_seconds()<72*3600]
        hi=[n for n in rb if n.importance>=2]
        if len(hi)>=2: pts+=15;facs.append(("🔥 Catalyst",f"{len(hi)} major bullish news","#FF453A"))
        elif hi: pts+=10;facs.append(("Catalyst",hi[0].headline[:60]+"...","#34C759"))
        elif rb: pts+=5;facs.append(("Positive Flow",f"{len(rb)} bullish headlines","#FFD60A"))
    sp=pf.short_pct_float if pf else 0
    if sp>=25: pts+=10;facs.append(("🔥 Squeeze Fuel",f"{sp:.1f}% short","#FF453A"))
    elif sp>=15: pts+=7;facs.append(("High Short",f"{sp:.1f}%","#FFD60A"))
    if sig.squeeze_on: pts+=8;facs.append(("🔥 Squeeze",f"BB inside Keltner","#BF5AF2"))
    if sig.direction=="bullish" and sig.momentum>2: pts+=7;facs.append(("Momentum",f"ROC +{sig.momentum:.1f}%","#34C759"))
    pts=min(100,pts)
    gr="A+" if pts>=85 else "A" if pts>=70 else "B+" if pts>=60 else "B" if pts>=50 else "C" if pts>=35 else "D" if pts>=20 else "F"
    return pts,gr,facs

# ═══ HEADER ═══════════════════════════════════════════════════════
h1,h2=st.columns([3,1])
with h1:
    st.markdown('<p class="sh">Signal</p>',unsafe_allow_html=True)
    st.markdown('<p class="ss">Professional Trading Analytics</p>',unsafe_allow_html=True)
with h2:
    mo=is_market_hours();mc="#34C759" if mo else "#FF9F0A";mt="MKT OPEN" if mo else "AFTER HOURS"
    st.markdown(f'<div style="text-align:right;padding-top:12px;"><span class="lb"><span class="ld" style="background:{mc};"></span>{mt} · {refresh_choice}</span><br><span class="mono" style="font-size:9px;color:rgba(255,255,255,0.2);">{datetime.now().strftime("%H:%M:%S")}</span></div>',unsafe_allow_html=True)

# ═══ DATA PIPELINE ════════════════════════════════════════════════
@st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice,120),show_spinner=False)
def load_and_predict(tickers,lookback):
    data=fetch_batch_data(tickers,period=lookback)
    sigs=predict_batch_parallel(data)
    return data,sigs

progress=st.progress(0,text="Loading...")
data_dict,signals=load_and_predict(tuple(selected_tickers),lookback)
progress.progress(40)

if not signals:
    progress.empty()
    st.error(f"⚠ No data for {len(selected_tickers)} tickers. yfinance may be temporarily unavailable.")
    st.caption(f"Attempted: {', '.join(selected_tickers[:15])}...")
    st.button("🔄 Retry",on_click=st.cache_data.clear)
    st.stop()

# Dynamic classification
cls={}
for t in signals:
    if t in data_dict: cls[t]=classify_stock(t,data_dict[t])

# Intraday
intra_stats={}
if enable_intraday:
    progress.progress(55)
    idict=fetch_intraday_batch(tuple(selected_tickers),days=5)
    for t,df in idict.items():
        av=float(data_dict[t]["Volume"].tail(20).mean()) if t in data_dict else 0
        s=compute_intraday_stats(t,df,avg_daily_vol=av)
        if s: intra_stats[t]=s

# Pro scoring
pro_scores,profiles,news_map={},{},{}
if enable_pro:
    progress.progress(75)
    for t,sig in signals.items():
        pf=fetch_company_profile(t)
        nw=fetch_news(t,days_back=5,max_items=8) if FK else []
        profiles[t]=pf;news_map[t]=nw
        dv=float(data_dict[t]["Volume"].iloc[-1]) if t in data_dict else 0
        pro_scores[t]=compute_pro_breakout(sig.breakout_score,t,dv,pf,nw,intra_stats.get(t))

# King nodes
king_profiles={}
for t in signals:
    if t in data_dict:
        vp=compute_volume_profile(data_dict[t])
        if vp: vp.ticker=t; king_profiles[t]=vp

progress.progress(100);progress.empty()

# ═══ METRICS ══════════════════════════════════════════════════════
bull=sum(1 for s in signals.values() if s.direction=="bullish")
bear=sum(1 for s in signals.values() if s.direction=="bearish")
pennies=sum(1 for c in cls.values() if c["is_penny"])
runners=sum(1 for c in cls.values() if c["is_runner_candidate"])
sub_d=sum(1 for c in cls.values() if c["is_sub_dollar"])
king_count=sum(1 for vp in king_profiles.values() for kn in vp.king_nodes if kn.strength=="king")

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("Bullish",bull)
c2.metric("Bearish",bear)
c3.metric("Pennies",pennies)
c4.metric("Sub-$1",sub_d)
c5.metric("Runners",runners)
c6.metric("King Nodes",king_count)
st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)

# Desk snapshot for fast decisioning
high_conf = sum(1 for s in signals.values() if s.confidence >= 70)
high_rvol = sum(1 for s in signals.values() if s.rvol_5 >= 1.8)
setup_ready = sum(1 for s in signals.values() if s.confidence >= 65 and s.rvol_5 >= 1.5 and s.direction != "neutral")
breadth = bull - bear
if breadth >= max(3, int(0.2 * len(signals))):
    regime = "Risk-On Trend Day"
    regime_color = "#34C759"
elif breadth <= -max(3, int(0.2 * len(signals))):
    regime = "Risk-Off Tape"
    regime_color = "#FF453A"
else:
    regime = "Mixed / Rotation"
    regime_color = "#FFD60A"
ds1, ds2, ds3, ds4 = st.columns(4)
ds1.metric("Desk Regime", regime)
ds2.metric("Setup Ready", setup_ready)
ds3.metric("High Conviction", high_conf)
ds4.metric("High RVol", high_rvol)
st.markdown(
    f'<div class="kc" style="margin-top:4px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span class="mono" style="font-size:11px;color:{regime_color};font-weight:700;">{regime}</span><span class="mono" style="font-size:10px;color:rgba(255,255,255,0.4);">Breadth {breadth:+d} · Profile {trader_profile}</span></div><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.35);margin-top:5px;">Prioritize names where direction, confidence, and participation agree. In {trader_profile.lower()} mode, entries favor structural levels within ±{profile_cfg["near_node_pct"]:.1f}% of spot.</div></div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)

# ═══ TABS ═════════════════════════════════════════════════════════
tab_ov,tab_run,tab_brk,tab_det,tab_opt,tab_heat=st.tabs(["◉ Overview","🚀 Penny Runners","🔥 Breakout","◎ Detail","⬡ Options","👑 King Nodes"])

# ═══ TAB 1: OVERVIEW ═════════════════════════════════════════════
with tab_ov:
    # Priority watchlist for at-a-glance execution ranking.
    watch_rows = []
    for t, s in signals.items():
        kp = king_profiles.get(t)
        nearest_node = None
        if kp and kp.king_nodes:
            nearest_node = min(kp.king_nodes, key=lambda n: abs(n.distance_pct))
        node_dist = abs(nearest_node.distance_pct) if nearest_node else 99.0
        setup_score = (
            0.45 * s.confidence
            + 0.30 * min(100, s.rvol_5 * 30)
            + 0.25 * max(0, 100 - node_dist * 20)
        )
        watch_rows.append({
            "Ticker": t,
            "Dir": s.direction[0].upper(),
            "Score": round(setup_score, 1),
            "Conf": f"{s.confidence:.0f}%",
            "RVol": f"{s.rvol_5:.1f}x",
            "Nearest Node": f"{node_dist:.1f}%" if nearest_node else "—",
            "Node": nearest_node.strength.title() if nearest_node else "—",
            "Last": f"${s.price:.2f}",
        })
    watch_df = pd.DataFrame(sorted(watch_rows, key=lambda r: r["Score"], reverse=True)[:8])
    st.markdown('<p class="sl">Priority Watchlist</p>', unsafe_allow_html=True)
    st.dataframe(watch_df, use_container_width=True, hide_index=True, height=320)

    sk=(lambda s:pro_scores[s.ticker].total_score) if pro_scores else (lambda s:s.confidence)
    ss_list=sorted(signals.values(),key=sk,reverse=True)
    cols=st.columns(3)
    for i,sig in enumerate(ss_list):
        nm,sect=get_ticker_info(sig.ticker);c=clr(sig.direction);cl=cls.get(sig.ticker,{})
        tier=cl.get("tier","?");pro=pro_scores.get(sig.ticker)
        grade=pro.total_grade if pro else sig.breakout_grade;g=gc(grade)
        bc="bg" if sig.direction=="bullish" else "br" if sig.direction=="bearish" else "bn"
        cc="#34C759" if sig.change_1d>=0 else "#FF453A"
        badges=""
        if cl.get("is_runner_candidate"): badges+=' <span class="b brun">🚀 Runner</span>'
        if cl.get("is_sub_dollar"): badges+=' <span class="b bsub">Sub-$1</span>'
        # King node badge
        kp=king_profiles.get(sig.ticker)
        if kp:
            kings=[kn for kn in kp.king_nodes if kn.strength=="king"]
            if kings: badges+=f' <span class="b bking">👑 {len(kings)} King</span>'
        with cols[i%3]:
            with st.container(border=True):
                card_pad = "10px" if compact_mode else "14px"
                st.markdown(f'<div style="padding:{card_pad};"><div style="display:flex;justify-content:space-between;"><div><span style="font-family:\'DM Sans\';font-size:17px;font-weight:700;color:#F5F5F7;">{sig.ticker}</span><span style="font-size:9px;color:rgba(255,255,255,0.2);margin-left:5px;">{tier}</span>{badges}</div><span class="b {bc}">{ico(sig.direction)} {sig.direction.title()}</span></div><div style="margin:5px 0;"><span class="mono" style="font-size:18px;font-weight:600;color:#F5F5F7;">${sig.price:,.2f}</span><span class="mono" style="font-size:10px;color:{cc};margin-left:6px;">{"+" if sig.change_1d>=0 else ""}{sig.change_1d:.2f}%</span></div><div style="display:flex;gap:7px;font-family:\'JetBrains Mono\';font-size:9px;"><span style="color:rgba(255,255,255,0.3);">Conf</span><span style="color:{c};font-weight:600;">{sig.confidence}%</span> <span style="color:rgba(255,255,255,0.3);">RVol</span><span>{sig.volume_ratio}x</span> <span style="color:{g};font-weight:700;">{grade}</span></div></div>',unsafe_allow_html=True)
                t0=sig.signals[0] if sig.signals else "—"
                st.markdown(f'<div style="margin-top:5px;padding-top:5px;border-top:1px solid rgba(255,255,255,0.04);font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.4);">{t0}</div>',unsafe_allow_html=True)

# ═══ TAB 2: PENNY RUNNERS ════════════════════════════════════════
with tab_run:
    p_sigs={t:s for t,s in signals.items() if cls.get(t,{}).get("is_penny") or cls.get(t,{}).get("is_runner_candidate")}
    if not p_sigs:
        p_sigs=dict(signals)
        st.caption("No sub-$5 stocks found. Scoring all tickers.")

    st.markdown(f'<p class="sl">🚀 {"Live" if mo else "After-Hours"} Penny Runner Scanner · {len(p_sigs)} stocks · Live-scanned from {len(SEED_UNIVERSE)} universe</p>',unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Sans\';font-size:11px;color:rgba(255,255,255,0.35);margin-bottom:12px;line-height:1.5;">A <strong style="color:#FF453A;">runner</strong> = stock under $1 for 12+ months that could break to $5+. Penny tickers are discovered live each session — not hardcoded.</div>',unsafe_allow_html=True)

    rd={}
    for t,s in p_sigs.items():
        pts,gr,fcs=runner_score(s,cls.get(t,{}),profiles.get(t),news_map.get(t,[]))
        rd[t]=(pts,gr,fcs)
    sp=sorted(rd.keys(),key=lambda t:rd[t][0],reverse=True)

    pc1,pc2,pc3,pc4=st.columns(4)
    pc1.metric("Runner Setups (≥50)",sum(1 for t in rd if rd[t][0]>=50))
    pc2.metric("Sub-Dollar",sum(1 for t in p_sigs if cls.get(t,{}).get("is_sub_dollar")))
    pc3.metric("Squeezes",sum(1 for s in p_sigs.values() if s.squeeze_on))
    pc4.metric("Top Pick",sp[0] if sp else "—")

    cols=st.columns(2)
    for i,t in enumerate(sp):
        sig=p_sigs[t];pts,grade,factors=rd[t];pf=profiles.get(t);nw=news_map.get(t,[])
        nm,sect=get_ticker_info(t);cl=cls.get(t,{});g=gc(grade)
        cc="#34C759" if sig.change_1d>=0 else "#FF453A";mu1=cl.get("months_under_1",0)
        badges=f'<span class="b {"bg" if sig.direction=="bullish" else "br" if sig.direction=="bearish" else "bn"}">{ico(sig.direction)}</span>'
        if cl.get("is_runner_candidate"): badges+=' <span class="b brun">🚀</span>'
        if cl.get("is_sub_dollar"): badges+=' <span class="b bsub"><$1</span>'
        if sig.squeeze_on: badges+=' <span class="b bsq">🔥</span>'
        if sig.rvol_5>1.5: badges+=f' <span class="b bvol">⚡{sig.rvol_5:.1f}x</span>'
        ft=f" · Float:{pf.float_shares_m:.0f}M" if pf and pf.float_shares_m>0 else ""
        sh=f" · Short:{pf.short_pct_float:.1f}%" if pf and pf.short_pct_float>0 else ""
        with cols[i%2]:
            with st.container(border=True):
                st.markdown(f'<div style="display:flex;justify-content:space-between;"><div style="display:flex;align-items:center;gap:7px;"><span class="gb" style="background:{g}18;color:{g};border:2px solid {g}35;">{grade}</span><div><span style="font-family:\'DM Sans\';font-size:16px;font-weight:700;color:#F5F5F7;">{t}</span><span style="font-size:9px;color:rgba(255,255,255,0.2);margin-left:5px;">{nm}{ft}{sh}</span><div style="display:flex;gap:2px;margin-top:2px;">{badges}</div></div></div><div style="text-align:right;"><div class="mono" style="font-size:18px;font-weight:600;color:#F5F5F7;">${sig.price:,.4f}</div><div class="mono" style="font-size:10px;color:{cc};">{sig.change_1d:+.2f}%</div></div></div>',unsafe_allow_html=True)
                st.markdown(f'<div style="display:flex;justify-content:space-between;margin-top:6px;"><span style="font-family:\'DM Sans\';font-size:8px;color:rgba(255,255,255,0.3);text-transform:uppercase;">Runner Score{" · "+str(mu1)+"mo under $1" if mu1>0 else ""}</span><span class="mono" style="font-size:11px;font-weight:700;color:{g};">{pts}/100</span></div><div class="sb"><div class="sf" style="width:{pts}%;"></div></div>',unsafe_allow_html=True)
                st.markdown(f'<div style="display:flex;gap:8px;margin:6px 0;font-family:\'JetBrains Mono\';font-size:9px;"><span style="color:rgba(255,255,255,0.3);">RVol</span><span>{sig.rvol_5:.1f}x</span><span style="color:rgba(255,255,255,0.3);">RSI</span><span>{sig.rsi:.0f}</span><span style="color:rgba(255,255,255,0.3);">5D</span><span style="color:{"#34C759" if sig.change_5d>=0 else "#FF453A"};">{sig.change_5d:+.1f}%</span><span style="color:rgba(255,255,255,0.3);">Conf</span><span style="color:{clr(sig.direction)};">{sig.confidence}%</span></div>',unsafe_allow_html=True)
                for fn,fd,fc in factors[:3]:
                    st.markdown(f'<div class="dr"><span class="dt" style="background:{fc}18;color:{fc};">{fn}</span><span style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.4);line-height:1.3;">{fd}</span></div>',unsafe_allow_html=True)
                if nw:
                    bn=[n for n in nw if n.sentiment=="bullish"][:1]
                    for n in bn:
                        hrs=(datetime.utcnow()-n.datetime_utc).total_seconds()/3600
                        st.markdown(f'<div class="nc bullish" style="margin-top:3px;"><div style="font-family:\'DM Sans\';font-size:9px;font-weight:600;color:#F5F5F7;">{"🔴" if hrs<12 else "🟡" if hrs<48 else ""} {n.headline[:70]}...</div><div style="font-family:\'JetBrains Mono\';font-size:7px;color:rgba(255,255,255,0.2);">{n.source} · {hrs:.0f}h</div></div>',unsafe_allow_html=True)

    st.markdown('<p class="sl">Ranking Table</p>',unsafe_allow_html=True)
    tbl=[{"Ticker":t,"Price":f"${p_sigs[t].price:,.4f}","Tier":cls.get(t,{}).get("tier","?"),"Mo<$1":cls.get(t,{}).get("months_under_1",0),"Runner":"🚀" if cls.get(t,{}).get("is_runner_candidate") else "—","1D%":f"{p_sigs[t].change_1d:+.2f}%","RVol":f"{p_sigs[t].rvol_5:.1f}x","Squeeze":"🔥" if p_sigs[t].squeeze_on else "—","Score":rd[t][0],"Grade":rd[t][1]} for t in sp]
    st.dataframe(pd.DataFrame(tbl),use_container_width=True,hide_index=True,height=min(380,40+len(tbl)*34))

# ═══ TAB 3: BREAKOUT ═════════════════════════════════════════════
with tab_brk:
    fc1,fc2=st.columns([1,1])
    with fc1: ms=st.slider("Min Score",0,150 if pro_scores else 100,30,5)
    with fc2: tf=st.radio("Tier",["All","Penny","Large Cap"],horizontal=True)
    def gs(t): return pro_scores[t].total_score if t in pro_scores else signals[t].breakout_score
    fl=sorted([t for t in signals if gs(t)>=ms],key=gs,reverse=True)
    if tf=="Penny":fl=[t for t in fl if cls.get(t,{}).get("is_penny")]
    elif tf=="Large Cap":fl=[t for t in fl if not cls.get(t,{}).get("is_penny")]
    if not fl: st.info("No tickers match.")
    else:
        cols=st.columns(2)
        for i,t in enumerate(fl):
            sig=signals[t];nm,sect=get_ticker_info(t);pro=pro_scores.get(t)
            grade=pro.total_grade if pro else sig.breakout_grade;score=gs(t);g=gc(grade)
            cc="#34C759" if sig.change_1d>=0 else "#FF453A"
            with cols[i%2]:
                with st.container(border=True):
                    st.markdown(f'<div style="display:flex;justify-content:space-between;"><div style="display:flex;align-items:center;gap:7px;"><span class="gb" style="background:{g}18;color:{g};border:2px solid {g}35;">{grade}</span><div><span style="font-family:\'DM Sans\';font-size:15px;font-weight:700;color:#F5F5F7;">{t}</span><span style="font-size:9px;color:rgba(255,255,255,0.2);margin-left:5px;">{nm}</span></div></div><div style="text-align:right;"><div class="mono" style="font-size:16px;font-weight:600;">${sig.price:,.2f}</div><div class="mono" style="font-size:9px;color:{cc};">{sig.change_1d:+.2f}%</div></div></div><div style="display:flex;gap:7px;margin-top:5px;font-family:\'JetBrains Mono\';font-size:9px;"><span style="color:rgba(255,255,255,0.3);">Score</span><span style="color:{g};font-weight:700;">{score:.0f}</span> <span style="color:rgba(255,255,255,0.3);">RVol</span><span>{sig.rvol_5:.1f}x</span> <span style="color:rgba(255,255,255,0.3);">RSI</span><span>{sig.rsi:.0f}</span></div>',unsafe_allow_html=True)
                    st.markdown(f'<div style="margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.04);font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.35);">{sig.signals[0] if sig.signals else "—"}</div>',unsafe_allow_html=True)

# ═══ TAB 4: DETAIL ═══════════════════════════════════════════════
with tab_det:
    sel=st.selectbox("Ticker",list(signals.keys()),format_func=lambda t:f"{t} — {get_ticker_info(t)[0]}",key="d_s")
    sig=signals[sel];nm,sect=get_ticker_info(sel);c=clr(sig.direction);cl=cls.get(sel,{})
    pro=pro_scores.get(sel);pf=profiles.get(sel);nw=news_map.get(sel,[])
    grade=pro.total_grade if pro else sig.breakout_grade;score=pro.total_score if pro else sig.breakout_score
    bc="bg" if sig.direction=="bullish" else "br" if sig.direction=="bearish" else "bn"
    st.markdown(f'<div style="display:flex;align-items:center;gap:10px;"><span style="font-family:\'DM Sans\';font-size:28px;font-weight:700;color:#F5F5F7;">{sel}</span><span class="b {bc}">{ico(sig.direction)} {sig.direction.title()}</span></div><p style="font-family:\'DM Sans\';font-size:12px;color:rgba(255,255,255,0.3);">{nm} · {cl.get("tier","?")} · {sect}</p>',unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6=st.columns(6)
    m1.metric("Price",f"${sig.price:,.4f}" if sig.price<1 else f"${sig.price:,.2f}")
    m2.metric("1D",f"{sig.change_1d:+.2f}%");m3.metric("Conf",f"{sig.confidence}%")
    m4.metric("RVol",f"{sig.rvol_5:.1f}x");m5.metric("Score",f"{grade} ({score:.0f})")
    m6.metric("Acc",f"{sig.accuracy}%")
    if pf and pf.float_shares_m>0:
        p1,p2,p3,p4=st.columns(4)
        p1.metric("Float",f"{pf.float_shares_m:.1f}M");p2.metric("MktCap",f"${pf.market_cap_m:,.0f}M")
        p3.metric("Short%",f"{pf.short_pct_float:.1f}%");p4.metric("ShortRatio",f"{pf.short_ratio:.1f}")
    # Chart
    df=data_dict.get(sel)
    if df is not None:
        xv=idx_str(df.index)
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.75,.25],vertical_spacing=.03)
        fig.add_trace(go.Candlestick(x=xv,open=to_list(df,"Open"),high=to_list(df,"High"),low=to_list(df,"Low"),close=to_list(df,"Close"),increasing_line_color="#34C759",decreasing_line_color="#FF453A",increasing_fillcolor="rgba(52,199,89,0.19)",decreasing_fillcolor="rgba(255,69,58,0.19)"),row=1,col=1)
        ca,oa,va=to_list(df,"Close"),to_list(df,"Open"),to_list(df,"Volume")
        vc=["#34C759" if c2>=o2 else "#FF453A" for c2,o2 in zip(ca,oa)]
        fig.add_trace(go.Bar(x=xv,y=va,marker_color=vc,opacity=0.4),row=2,col=1)
        # King node lines
        kp=king_profiles.get(sel)
        if kp:
            for kn in kp.king_nodes[:5]:
                lc="#0A84FF" if kn.strength=="king" else "#BF5AF2" if kn.strength=="major" else "rgba(255,255,255,0.15)"
                lw=2 if kn.strength=="king" else 1
                fig.add_hline(y=kn.price_level,line_dash="dot",line_color=lc,line_width=lw,annotation_text=f"{'👑' if kn.strength=='king' else '◆'} ${kn.price_level:.2f} ({kn.volume_pct:.0f}%vol)",annotation_font_size=9,annotation_font_color=lc,row=1,col=1)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans",color="rgba(255,255,255,0.5)",size=10),margin=dict(l=0,r=0,t=20,b=0),xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),height=380,showlegend=False,xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key=f"dc_{sel}")
    # Signals + News
    sc1,sc2=st.columns([1,1])
    with sc1:
        st.markdown('<p class="sl">Signals</p>',unsafe_allow_html=True)
        for s in sig.signals:
            st.markdown(f'<div style="display:flex;align-items:center;gap:5px;padding:3px 0;font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.5);"><span style="width:4px;height:4px;border-radius:50%;background:{c};flex-shrink:0;"></span>{s}</div>',unsafe_allow_html=True)
    with sc2:
        if nw:
            st.markdown('<p class="sl">News</p>',unsafe_allow_html=True)
            for n in nw[:4]:
                hrs=(datetime.utcnow()-n.datetime_utc).total_seconds()/3600
                st.markdown(f'<div class="nc {n.sentiment}"><div style="font-family:\'DM Sans\';font-size:10px;font-weight:600;color:#F5F5F7;">{"🔴" if hrs<12 else "🟡" if hrs<48 else ""} {n.headline[:80]}</div><div style="font-family:\'JetBrains Mono\';font-size:7px;color:rgba(255,255,255,0.2);">{n.source} · {hrs:.0f}h</div></div>',unsafe_allow_html=True)

# ═══ TAB 5: OPTIONS ══════════════════════════════════════════════
with tab_opt:
    ot=st.selectbox("Ticker",list(signals.keys()),format_func=lambda t:f"{t} — {signals[t].direction.title()}",key="o_s")
    @st.cache_data(ttl=CACHE_TTL_MAP.get(refresh_choice,120),show_spinner=False)
    def gp(tk,sh,lb):
        s,d=signals.get(tk),data_dict.get(tk)
        return generate_options_plays(s,d) if s and d is not None else []
    sig=signals[ot];plays=gp(ot,f"{ot}_{sig.confidence}",lookback)
    if not plays: st.warning(f"No plays for {ot}.")
    else:
        for play in plays:
            tc={"directional_bullish":("#34C759","rgba(52,199,89,0.12)"),"directional_bearish":("#FF453A","rgba(255,69,58,0.12)"),"neutral":("#0A84FF","rgba(10,132,255,0.12)"),"volatility":("#BF5AF2","rgba(191,90,242,0.12)")}
            t_c,t_b=tc.get(play.strategy_type,("#FFD60A","rgba(255,214,10,0.12)"))
            ccc="bullish" if "bullish" in play.strategy_type else "bearish" if "bearish" in play.strategy_type else "volatility" if play.strategy_type=="volatility" else "neutral"
            st.markdown(f'<div class="pc {ccc}">',unsafe_allow_html=True)
            pc1,pc2=st.columns([3,1])
            with pc1: st.markdown(f'<div style="font-family:\'DM Sans\';font-size:18px;font-weight:700;color:#F5F5F7;">{play.strategy_name}</div><div style="display:flex;gap:4px;margin-top:2px;"><span class="b" style="background:{t_b};color:{t_c};font-family:\'JetBrains Mono\';font-size:8px;">{play.strategy_type.replace("_"," ")}</span><span class="b" style="background:rgba(255,255,255,0.06);color:rgba(255,255,255,0.5);font-family:\'JetBrains Mono\';font-size:8px;">{play.risk_tier}</span></div>',unsafe_allow_html=True)
            with pc2: st.markdown(f'<div style="text-align:right;"><div class="kl">PoP</div><div class="mono" style="font-size:22px;font-weight:700;color:{t_c};">{play.probability_of_profit}%</div></div>',unsafe_allow_html=True)
            for leg in play.legs:
                dc="#34C759" if leg.direction=="buy" else "#FF6961"
                db="rgba(52,199,89,0.15)" if leg.direction=="buy" else "rgba(255,105,97,0.15)"
                st.markdown(f'<div class="lr"><span style="background:{db};color:{dc};font-weight:700;font-size:8px;padding:1px 4px;border-radius:3px;text-transform:uppercase;">{leg.direction}</span><span style="color:#F5F5F7;font-weight:600;">{leg.option_type.upper()}</span><span style="color:rgba(255,255,255,0.4);">K</span>${leg.strike:.0f}<span style="color:rgba(255,255,255,0.4);">Prem</span>${leg.estimated_premium:.2f}<span style="color:rgba(255,255,255,0.3);margin-left:auto;">Δ{leg.delta:.2f}</span></div>',unsafe_allow_html=True)
            icr=play.strategy_type=="neutral" or "Short" in play.strategy_name
            gi=[(("Credit" if icr else "Debit"),f"${play.entry_price:.2f}"),("MaxLoss",f"${play.max_loss:.0f}"),("MaxGain","∞" if play.max_gain==-1 else f"${play.max_gain:.0f}"),("R/R",f"{play.risk_reward_ratio:.1f}x"),("BE",f"${play.break_even:.2f}"),("PoP",f"{play.probability_of_profit}%")]
            gh='<div class="kg">'
            for l,v in gi: gh+=f'<div class="kc"><div class="kl">{l}</div><div class="kv">{v}</div></div>'
            st.markdown(gh+'</div>',unsafe_allow_html=True)
            st.markdown(
                f'<div class="kc" style="margin-top:6px;"><div class="kl">Execution Checklist</div><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.45);line-height:1.45;">'
                f'1) Trigger: {play.entry_timing}<br>'
                f'2) Target discipline: {play.exit_timing}<br>'
                f'3) Risk: stop near {play.stop_loss:.2f} premium-equivalent, do not average losers.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            if play.hold_duration:
                st.markdown(f'<div style="display:grid;grid-template-columns:auto 1fr;gap:8px;margin-top:8px;"><div class="kc" style="text-align:center;"><div class="kl">Hold</div><div class="mono" style="font-size:14px;font-weight:700;color:{t_c};">{play.hold_duration}</div></div><div class="kc"><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.45);line-height:1.4;">{play.hold_reasoning}</div></div></div>',unsafe_allow_html=True)
            if play.theta_decay_warning:
                st.markdown(f'<div class="kc" style="margin-top:6px;border-left:2px solid #FF9F0A;"><div class="kl">Theta Risk</div><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.42);line-height:1.4;">{play.theta_decay_warning}</div></div>',unsafe_allow_html=True)
            if play.optimal_exit_scenario:
                st.markdown(f'<div class="kc" style="margin-top:6px;border-left:2px solid #34C759;"><div class="kl">Optimal Exit</div><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.42);line-height:1.4;">{play.optimal_exit_scenario}</div></div>',unsafe_allow_html=True)
            if play.price_drivers:
                for f,e,d in play.price_drivers[:3]:
                    st.markdown(f'<div class="dr"><span style="font-size:12px;">{e}</span><div><div style="font-family:\'DM Sans\';font-size:10px;font-weight:600;color:#F5F5F7;">{f}</div><div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.35);">{d}</div></div></div>',unsafe_allow_html=True)
            if play.risks:
                st.markdown('<p class="sl" style="margin-top:8px;">Risk Flags</p>',unsafe_allow_html=True)
                for rk in play.risks[:2]:
                    st.markdown(f'<div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,120,120,0.78);line-height:1.35;padding:2px 0;">• {rk}</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="tb" style="margin-top:6px;">{play.thesis}</div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

# ═══ TAB 6: KING NODES HEATMAP ═══════════════════════════════════
with tab_heat:
    st.markdown('<p class="sl">👑 King Nodes — Volume-at-Price Heatmap</p>',unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Sans\';font-size:11px;color:rgba(255,255,255,0.35);margin-bottom:14px;line-height:1.5;">King nodes are price levels where abnormally high volume was traded — institutional footprints. When price is <strong style="color:#34C759;">above</strong> a king node it acts as <strong style="color:#34C759;">support</strong>; <strong style="color:#FF453A;">below</strong> it acts as <strong style="color:#FF453A;">resistance</strong>. The <span style="color:#0A84FF;">POC</span> (Point of Control) is the single highest-volume level — where the market found fair value.</div>',unsafe_allow_html=True)

    kn_sel=st.selectbox("Ticker",list(king_profiles.keys()),format_func=lambda t:f"{t} — {get_ticker_info(t)[0]}",key="kn_s")
    kp=king_profiles.get(kn_sel)
    if kp:
        sig=signals[kn_sel]
        node_by_price = {round(n.price_level, 2): n for n in kp.king_nodes}
        above_nodes = sorted([n for n in kp.king_nodes if n.price_level > kp.current_price], key=lambda n: n.price_level)
        below_nodes = sorted([n for n in kp.king_nodes if n.price_level < kp.current_price], key=lambda n: n.price_level, reverse=True)
        top_king = next((n for n in kp.king_nodes if n.strength == "king"), kp.king_nodes[0] if kp.king_nodes else None)
        nearest_above = min(above_nodes, key=lambda n: n.price_level - kp.current_price) if above_nodes else None
        nearest_below = min(below_nodes, key=lambda n: kp.current_price - n.price_level) if below_nodes else None

        st.markdown('<p class="sl">Flow Heatmap (Options-Style Layout)</p>', unsafe_allow_html=True)
        ht1, ht2, ht3, ht4 = st.columns([1.1, 1.1, 1.2, 1.6])
        with ht1:
            strike_rows = st.selectbox("Rows", [24, 32, 40], index=1, key="hm_rows")
        with ht2:
            horizon = st.selectbox("Horizon", ["1w", "1m", "2m"], index=1, key="hm_horizon")
        with ht3:
            flow_mode = st.selectbox("Metric", ["Net Flow", "Call Pressure", "Put Pressure"], index=0, key="hm_metric")
        with ht4:
            st.markdown(
                f'<div class="kc" style="height:70px;"><div class="kl">Board</div><div style="display:flex;justify-content:space-between;align-items:center;"><span class="b" style="background:rgba(52,199,89,0.2);color:#34C759;">LIVE</span><span class="mono" style="font-size:10px;color:rgba(255,255,255,0.55);">{datetime.now().strftime("%H:%M:%S")}</span></div><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.32);margin-top:5px;">{kn_sel} spot ${kp.current_price:.2f} · POC ${kp.poc:.2f}</div></div>',
                unsafe_allow_html=True,
            )

        # Construct an options-style strike x expiry heatmap from node structure.
        center_idx = min(range(len(kp.levels)), key=lambda i: abs(kp.levels[i] - kp.current_price))
        half_window = max(10, strike_rows // 2)
        i0 = max(0, center_idx - half_window)
        i1 = min(len(kp.levels), center_idx + half_window)
        strike_slice = kp.levels[i0:i1]
        strike_prices = sorted([round(x, 2) for x in strike_slice], reverse=True)
        if len(strike_prices) > strike_rows:
            step = max(1, len(strike_prices) // strike_rows)
            strike_prices = strike_prices[::step][:strike_rows]

        if horizon == "1w":
            expiry_steps = [0, 3, 7, 10]
        elif horizon == "2m":
            expiry_steps = [0, 14, 30, 45]
        else:
            expiry_steps = [0, 7, 14, 28]
        expiry_labels = [(pd.Timestamp.today() + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in expiry_steps]

        z_vals = []
        text_vals = []
        sigma = max(0.25, (max(kp.levels) - min(kp.levels)) / 20)
        direction_sign = 1.0 if sig.direction == "bullish" else -1.0 if sig.direction == "bearish" else 0.0
        for s_px in strike_prices:
            row = []
            row_text = []
            rel_side = 1.0 if s_px < kp.current_price else -1.0
            for j, d in enumerate(expiry_steps):
                node_influence = 0.0
                for kn in kp.king_nodes:
                    dist = abs(s_px - kn.price_level)
                    falloff = np.exp(-(dist / sigma) ** 2)
                    node_sign = 1.0 if kn.node_type in ("support", "poc") else -1.0
                    node_weight = 1.0 if kn.strength == "king" else 0.7 if kn.strength == "major" else 0.45
                    node_influence += node_sign * node_weight * kn.volume_pct * falloff
                tenor_decay = max(0.35, 1.0 - j * 0.18)
                directional = direction_sign * 22.0 + rel_side * 12.0
                seasonal = np.sin((s_px * 0.05) + (j * 0.8)) * 4.0
                raw = (node_influence * 1.8 + directional + seasonal) * tenor_decay
                if flow_mode == "Call Pressure":
                    raw = max(raw, 0.0)
                elif flow_mode == "Put Pressure":
                    raw = min(raw, 0.0)
                row.append(raw)
                row_text.append(f"${raw:,.1f}K")
            z_vals.append(row)
            text_vals.append(row_text)

        heat = go.Figure(
            data=[
                go.Heatmap(
                    z=z_vals,
                    x=expiry_labels,
                    y=strike_prices,
                    text=text_vals,
                    texttemplate="%{text}",
                    textfont={"size": 10, "color": "rgba(245,245,247,0.92)", "family": "JetBrains Mono"},
                    colorscale=[
                        [0.0, "#7A1D2A"],
                        [0.25, "#4C286B"],
                        [0.5, "#332455"],
                        [0.7, "#2E5B88"],
                        [0.85, "#2EA7B8"],
                        [1.0, "#FFD166"],
                    ],
                    zmid=0,
                    showscale=False,
                    hovertemplate="Strike %{y}<br>Expiry %{x}<br>Flow %{z:,.1f}K<extra></extra>",
                )
            ]
        )
        heat.add_hline(
            y=kp.current_price,
            line_color="rgba(245,245,247,0.6)",
            line_width=1.5,
            line_dash="dot",
        )
        heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=8, b=0),
            height=560,
            font=dict(family="DM Sans", size=10, color="rgba(255,255,255,0.65)"),
            xaxis=dict(side="top", showgrid=False, zeroline=False, title="Expiry"),
            yaxis=dict(showgrid=False, zeroline=False, title="Strike", tickformat=".1f"),
        )
        st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False}, key=f"flow_heat_{kn_sel}")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            # Volume profile chart (horizontal bar chart of volume at each price level)
            fig = go.Figure()
            bar_colors = []
            for lvl, vol in zip(kp.levels, kp.volumes):
                kn = node_by_price.get(round(lvl, 2))
                if kn and kn.strength == "king":
                    bar_colors.append("#0A84FF")
                elif kn and kn.strength == "major":
                    bar_colors.append("#BF5AF2")
                elif lvl < kp.current_price:
                    bar_colors.append("rgba(52,199,89,0.4)")
                else:
                    bar_colors.append("rgba(255,69,58,0.4)")

            fig.add_trace(go.Bar(y=[round(l, 2) for l in kp.levels], x=kp.volumes, orientation='h', marker_color=bar_colors, marker_line_width=0, hovertemplate="Price: $%{y:.2f}<br>Volume: %{x:,.0f}<extra></extra>"))
            fig.add_hline(y=kp.current_price, line_color="#F5F5F7", line_width=2, annotation_text=f"Current ${kp.current_price:.2f}", annotation_font_size=10, annotation_font_color="#F5F5F7")
            fig.add_hline(y=kp.poc, line_color="#0A84FF", line_width=2, line_dash="dot", annotation_text=f"POC ${kp.poc:.2f}", annotation_font_size=10, annotation_font_color="#0A84FF", annotation_position="bottom right")
            fig.add_hrect(y0=kp.val, y1=kp.vah, fillcolor="rgba(10,132,255,0.06)", line_width=0, annotation_text="Value Area (70%)", annotation_font_size=9, annotation_font_color="rgba(10,132,255,0.5)")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="rgba(255,255,255,0.5)", size=10), margin=dict(l=0, r=0, t=20, b=0), xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Volume"), yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Price"), height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"vp_{kn_sel}")

        with chart_cols[1]:
            ladder = go.Figure()
            max_vol = max(kp.volumes) if kp.volumes else 1.0
            xs, ys, sizes, colors, labels, hover = [], [], [], [], [], []
            for kn in kp.king_nodes:
                side_x = 1 if kn.price_level > kp.current_price else -1 if kn.price_level < kp.current_price else 0
                base_color = "#0A84FF" if kn.strength == "king" else "#BF5AF2" if kn.strength == "major" else "rgba(255,255,255,0.45)"
                xs.append(side_x)
                ys.append(kn.price_level)
                sizes.append(10 + (kn.volume / max_vol) * 22)
                colors.append(base_color)
                labels.append("👑" if kn.strength == "king" else "◆")
                hover.append(f"${kn.price_level:.2f}<br>{kn.node_type.title()}<br>{kn.volume_pct:.1f}% volume<br>{kn.distance_pct:+.2f}% from current")

            ladder.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=labels,
                textposition="middle center",
                marker=dict(size=sizes, color=colors, line=dict(color="rgba(255,255,255,0.3)", width=1)),
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            ))
            ladder.add_hline(y=kp.current_price, line_color="#F5F5F7", line_width=2, annotation_text=f"Current ${kp.current_price:.2f}", annotation_font_color="#F5F5F7", annotation_font_size=10)
            ladder.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_dash="dot")
            if top_king:
                ladder.add_hline(y=top_king.price_level, line_color="#0A84FF", line_width=2, line_dash="dot", annotation_text=f"King ${top_king.price_level:.2f}", annotation_font_color="#0A84FF", annotation_font_size=10)

            ladder.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="rgba(255,255,255,0.5)", size=10),
                margin=dict(l=0, r=0, t=20, b=0),
                height=450,
                xaxis=dict(
                    range=[-1.5, 1.5],
                    tickmode="array",
                    tickvals=[-1, 0, 1],
                    ticktext=["Below Price", "Current", "Above Price"],
                    gridcolor="rgba(255,255,255,0.04)",
                    zeroline=False,
                    title="Vertical Price Ladder",
                ),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Price"),
                showlegend=False,
            )
            st.plotly_chart(ladder, use_container_width=True, config={"displayModeBar": False}, key=f"ladder_{kn_sel}")

        # King node details
        st.markdown('<p class="sl">Key Levels</p>',unsafe_allow_html=True)
        kn_cols=st.columns(4)
        kn_cols[0].metric("POC (Fair Value)",f"${kp.poc:.2f}")
        kn_cols[1].metric("Value Area High",f"${kp.vah:.2f}")
        kn_cols[2].metric("Value Area Low",f"${kp.val:.2f}")
        kn_cols[3].metric("Dominant King", f"${top_king.price_level:.2f}" if top_king else "—")

        def node_strength_weight(node):
            if not node:
                return 0.0
            return 1.0 if node.strength == "king" else 0.75 if node.strength == "major" else 0.45

        def distance_weight(node):
            if not node:
                return 0.0
            d = abs(node.distance_pct)
            if d <= 1.0:
                return 1.0
            if d <= 2.5:
                return 0.85
            if d <= 4.0:
                return 0.65
            if d <= 6.0:
                return 0.4
            return 0.2

        def node_quality_score(node):
            if not node:
                return 0.0
            volume_component = min(1.0, node.volume_pct / 12.0)
            strength_component = node_strength_weight(node)
            proximity_component = distance_weight(node)
            return (0.45 * strength_component) + (0.35 * volume_component) + (0.20 * proximity_component)

        trend_bull = 1 if sig.direction == "bullish" else 0
        trend_bear = 1 if sig.direction == "bearish" else 0
        conf_component = min(1.0, sig.confidence / 100.0)
        vol_confirm = min(1.0, sig.volume_ratio / 2.2)
        momentum_bull = 1.0 if sig.momentum > 0 and sig.macd_hist > 0 else 0.55 if sig.momentum > 0 else 0.25
        momentum_bear = 1.0 if sig.momentum < 0 and sig.macd_hist < 0 else 0.55 if sig.momentum < 0 else 0.25

        support_quality = node_quality_score(nearest_below)
        resistance_quality = node_quality_score(nearest_above)
        king_location_bull = 1.0 if top_king and top_king.price_level < kp.current_price else 0.45
        king_location_bear = 1.0 if top_king and top_king.price_level > kp.current_price else 0.45

        call_score_raw = (
            100
            * (
                0.34 * support_quality
                + 0.16 * king_location_bull
                + 0.18 * conf_component
                + 0.14 * vol_confirm
                + 0.10 * momentum_bull
                + 0.08 * trend_bull
            )
        )
        put_score_raw = (
            100
            * (
                0.34 * resistance_quality
                + 0.16 * king_location_bear
                + 0.18 * conf_component
                + 0.14 * vol_confirm
                + 0.10 * momentum_bear
                + 0.08 * trend_bear
            )
        )

        # Penalize setups where nearest structural level is too far to define risk.
        if nearest_below and abs(nearest_below.distance_pct) > profile_cfg["distance_penalty"]:
            call_score_raw *= 0.75
        if nearest_above and abs(nearest_above.distance_pct) > profile_cfg["distance_penalty"]:
            put_score_raw *= 0.75

        call_conf = min(99, round(call_score_raw, 1))
        put_conf = min(99, round(put_score_raw, 1))
        bias = "CALL bias" if call_conf > put_conf else "PUT bias" if put_conf > call_conf else "NEUTRAL"
        bias_color = "#34C759" if "CALL" in bias else "#FF453A" if "PUT" in bias else "#FFD60A"

        bias_cols = st.columns(3)
        bias_cols[0].metric("Nearest Support", f"${nearest_below.price_level:.2f}" if nearest_below else "—", f"{nearest_below.distance_pct:+.1f}%" if nearest_below else None)
        bias_cols[1].metric("Nearest Resistance", f"${nearest_above.price_level:.2f}" if nearest_above else "—", f"{nearest_above.distance_pct:+.1f}%" if nearest_above else None)
        bias_cols[2].metric("Options Reliability", f"{max(call_conf, put_conf):.0f}%")

        def reliability_label(score):
            adj_score = score + profile_cfg["reliability_shift"]
            if adj_score >= 78:
                return "Institutional Grade"
            if adj_score >= 65:
                return "Tradable"
            if adj_score >= 52:
                return "Tactical Only"
            return "Low Edge"

        lead_score = max(call_conf, put_conf)
        rel_label = reliability_label(lead_score)
        st.markdown(
            f'<div class="kc" style="margin-top:8px;"><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Directional Read from King Nodes</div><div style="display:flex;justify-content:space-between;align-items:center;"><span class="mono" style="font-size:13px;font-weight:700;color:{bias_color};">{bias}</span><span class="mono" style="font-size:11px;color:rgba(255,255,255,0.55);">Call {call_conf:.0f}% · Put {put_conf:.0f}%</span></div><div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;"><span class="b" style="background:rgba(255,255,255,0.07);color:#F5F5F7;font-family:\'JetBrains Mono\';font-size:9px;">{rel_label}</span><span class="mono" style="font-size:10px;color:rgba(255,255,255,0.45);">Model {sig.confidence:.0f}% · RVol {sig.volume_ratio:.1f}x</span></div><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.35);margin-top:6px;line-height:1.4;">Reliability combines node strength, distance from spot, volume concentration, trend/momentum alignment, and model conviction. In {trader_profile.lower()} mode, prefer nearest structural levels within ±{profile_cfg["near_node_pct"]:.1f}% and king/major nodes.</div></div>',
            unsafe_allow_html=True,
        )

        # Professional trader execution framing from structure.
        if bias == "CALL bias" and nearest_below and nearest_above:
            entry_zone = f"${(nearest_below.price_level + kp.current_price) / 2:.2f}–${kp.current_price:.2f}"
            invalidation = f"Daily close below ${nearest_below.price_level * 0.995:.2f}"
            target_1 = f"${nearest_above.price_level:.2f}"
            target_2 = f"${(nearest_above.price_level + (nearest_above.price_level - nearest_below.price_level) * 0.6):.2f}"
        elif bias == "PUT bias" and nearest_below and nearest_above:
            entry_zone = f"${kp.current_price:.2f}–${(nearest_above.price_level + kp.current_price) / 2:.2f}"
            invalidation = f"Daily close above ${nearest_above.price_level * 1.005:.2f}"
            target_1 = f"${nearest_below.price_level:.2f}"
            target_2 = f"${(nearest_below.price_level - (nearest_above.price_level - nearest_below.price_level) * 0.6):.2f}"
        else:
            entry_zone = "Wait for cleaner structure"
            invalidation = "N/A"
            target_1 = "N/A"
            target_2 = "N/A"

        size_note = "0.5x size unless Institutional Grade" if trader_profile == "Conservative" else "normal size if Tradable+" if trader_profile == "Balanced" else "can scale faster, respect invalidation strictly"
        st.markdown(
            f'<div class="kc" style="margin-top:8px;"><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Execution Plan (Professional Framing)</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;"><div class="kc"><div class="kl">Entry Zone</div><div class="mono" style="font-size:12px;color:#F5F5F7;">{entry_zone}</div></div><div class="kc"><div class="kl">Invalidation</div><div class="mono" style="font-size:12px;color:#FF9F0A;">{invalidation}</div></div><div class="kc"><div class="kl">Target 1</div><div class="mono" style="font-size:12px;color:#34C759;">{target_1}</div></div><div class="kc"><div class="kl">Target 2</div><div class="mono" style="font-size:12px;color:#34C759;">{target_2}</div></div></div><div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.35);margin-top:6px;">Profile rule ({trader_profile}): {size_note}. Reduce size if nearest node is beyond {profile_cfg["distance_penalty"]:.1f}% from spot.</div></div>',
            unsafe_allow_html=True,
        )

        # King node list
        if kp.king_nodes:
            st.markdown('<p class="sl">Detected Nodes</p>',unsafe_allow_html=True)
            cols=st.columns(2)
            for i,kn in enumerate(kp.king_nodes[:8]):
                nc="#0A84FF" if kn.strength=="king" else "#BF5AF2" if kn.strength=="major" else "rgba(255,255,255,0.4)"
                icon="👑" if kn.strength=="king" else "◆" if kn.strength=="major" else "·"
                type_c="#34C759" if kn.node_type=="support" else "#FF453A" if kn.node_type=="resistance" else "#0A84FF"
                with cols[i%2]:
                    st.markdown(f'<div class="kc" style="margin-bottom:6px;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><span style="font-size:14px;">{icon}</span> <span class="mono" style="font-size:14px;font-weight:600;color:{nc};">${kn.price_level:.2f}</span><span class="b" style="background:{type_c}18;color:{type_c};margin-left:6px;font-family:\'JetBrains Mono\';font-size:8px;">{kn.node_type}</span></div><span class="mono" style="font-size:11px;color:rgba(255,255,255,0.4);">{kn.volume_pct:.1f}% vol</span></div><div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.3);margin-top:3px;">{kn.distance_pct:+.1f}% from current · {kn.strength.title()} node</div></div>',unsafe_allow_html=True)

        # All-tickers king node summary
        st.markdown('<p class="sl">King Node Summary — All Tickers</p>',unsafe_allow_html=True)
        kn_tbl=[]
        for t in sorted(king_profiles.keys()):
            vp=king_profiles[t];s=signals[t]
            kings=[kn for kn in vp.king_nodes if kn.strength=="king"]
            nearest=min(vp.king_nodes,key=lambda n:abs(n.distance_pct)) if vp.king_nodes else None
            kn_tbl.append({"Ticker":t,"Price":f"${vp.current_price:.2f}","POC":f"${vp.poc:.2f}","VAH":f"${vp.vah:.2f}","VAL":f"${vp.val:.2f}","King Nodes":len(kings),"Nearest":f"${nearest.price_level:.2f} ({nearest.distance_pct:+.1f}%)" if nearest else "—","Type":nearest.node_type if nearest else "—"})
        st.dataframe(pd.DataFrame(kn_tbl),use_container_width=True,hide_index=True,height=min(380,40+len(kn_tbl)*34))
    else:
        st.warning("No volume profile data available for this ticker.")

st.markdown('<div style="height:16px"></div><p style="font-size:8px;color:rgba(255,255,255,0.08);text-align:center;padding:10px 0;">Signal · yfinance + Finnhub · Educational use only. Not financial advice.</p>',unsafe_allow_html=True)
