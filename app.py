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
from models.zero_dte import compute_0dte_analysis, get_0dte_tickers, get_current_session_window, get_all_session_windows
from utils.live_quotes import fetch_live_quotes_batch, has_live_data, LiveQuote
from utils.unusual_whales import (
    fetch_flow_alerts, fetch_dark_pool, fetch_congress_trades,
    fetch_uw_news, fetch_stock_screener, has_uw_key,
)

st.set_page_config(page_title="Signal Pro", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
FK = bool(_get_api_key())
UW = has_uw_key()

# ═══ SIDEBAR ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◉ Signal Config")
    st.markdown("---")
    refresh_choice = st.selectbox("⏱ Refresh", list(REFRESH_INTERVALS.keys()), index=2)
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
    # Always include 0DTE tickers
    dte_must = ["SPY", "QQQ", "IWM"]
    selected_tickers = list(set(selected_tickers + dte_must))
    st.markdown("---")
    lookback = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y"], index=1)
    enable_pro = st.checkbox("Pro Scoring", True)
    enable_intraday = st.checkbox("Intraday Data", True)
    st.markdown("---")
    if FK: st.success("✓ Finnhub connected")
    else:
        st.warning("⚠ No Finnhub key")
        with st.expander("Setup"): st.markdown("Get free key at **finnhub.io** → add `FINNHUB_API_KEY` in Streamlit secrets")
    if UW: st.success("✓ Unusual Whales connected")
    else:
        st.info("🐋 Unusual Whales (optional)")
        with st.expander("Setup UW"):
            st.markdown("""
**Get API key:**
1. Sign up at [unusualwhales.com](https://unusualwhales.com)
2. Go to **Settings → API Dashboard**
   (`unusualwhales.com/settings/api-dashboard`)
3. Generate your API key

**Add to Streamlit secrets:**
```
UW_API_KEY = "your_key_here"
```

Also works as `UNUSUAL_WHALES_API_KEY` (official MCP convention).

Unlocks: options flow, dark pool, congressional trades, stock screener.
""")
    st.markdown("---")
    st.markdown("**Definitions**")
    st.markdown("• **Penny stock** = price < $5")
    st.markdown("• **Runner** = under $1 for 12+ months")
    st.markdown("• **King node** = price level with abnormally high traded volume (HVN)")

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
king_count=sum(1 for vp in king_profiles.values() for kn in vp.all_nodes if kn.node_type=="king")

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("Bullish",bull)
c2.metric("Bearish",bear)
c3.metric("Pennies",pennies)
c4.metric("Sub-$1",sub_d)
c5.metric("Runners",runners)
c6.metric("King Nodes",king_count)
st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)

# ═══ TABS ═════════════════════════════════════════════════════════
tab_ov,tab_0dte,tab_run,tab_flow,tab_brk,tab_det,tab_opt,tab_heat=st.tabs(["◉ Overview","⚡ 0DTE","🚀 Penny Runners","🐋 Flow","🔥 Breakout","◎ Detail","⬡ Options","👑 King Nodes"])

# ═══ TAB 1: OVERVIEW ═════════════════════════════════════════════
with tab_ov:
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
            kings=[kn for kn in kp.all_nodes if kn.node_type=="king"]
            if kings: badges+=f' <span class="b bking">👑 {len(kings)} King</span>'
        with cols[i%3]:
            with st.container(border=True):
                st.markdown(f'<div style="display:flex;justify-content:space-between;"><div><span style="font-family:\'DM Sans\';font-size:17px;font-weight:700;color:#F5F5F7;">{sig.ticker}</span><span style="font-size:9px;color:rgba(255,255,255,0.2);margin-left:5px;">{tier}</span>{badges}</div><span class="b {bc}">{ico(sig.direction)} {sig.direction.title()}</span></div><div style="margin:5px 0;"><span class="mono" style="font-size:18px;font-weight:600;color:#F5F5F7;">${sig.price:,.2f}</span><span class="mono" style="font-size:10px;color:{cc};margin-left:6px;">{"+" if sig.change_1d>=0 else ""}{sig.change_1d:.2f}%</span></div><div style="display:flex;gap:7px;font-family:\'JetBrains Mono\';font-size:9px;"><span style="color:rgba(255,255,255,0.3);">Conf</span><span style="color:{c};font-weight:600;">{sig.confidence}%</span> <span style="color:rgba(255,255,255,0.3);">RVol</span><span>{sig.volume_ratio}x</span> <span style="color:{g};font-weight:700;">{grade}</span></div>',unsafe_allow_html=True)
                t0=sig.signals[0] if sig.signals else "—"
                st.markdown(f'<div style="margin-top:5px;padding-top:5px;border-top:1px solid rgba(255,255,255,0.04);font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.4);">{t0}</div>',unsafe_allow_html=True)

# ═══ TAB 2: 0DTE ══════════════════════════════════════════════════
with tab_0dte:
    st.markdown('<p class="sl">⚡ 0DTE Command Center — SPY · QQQ · IWM</p>',unsafe_allow_html=True)

    # Session window banner
    cw=get_current_session_window()
    if cw:
        st.markdown(f'<div style="padding:12px 16px;background:{cw["color"]}12;border:1px solid {cw["color"]}30;border-radius:10px;margin-bottom:14px;"><div style="display:flex;align-items:center;gap:10px;"><span class="b" style="background:{cw["color"]}25;color:{cw["color"]};font-family:\'JetBrains Mono\';font-size:10px;">{cw["name"].upper()}</span><span style="font-family:\'DM Sans\';font-size:12px;color:rgba(255,255,255,0.6);">{cw["action"]}</span></div></div>',unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:12px 16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;margin-bottom:14px;"><div style="font-family:\'DM Sans\';font-size:12px;color:rgba(255,255,255,0.5);">Market closed. Use this time to identify key levels and plan entries for the next session. Review the session windows below.</div></div>',unsafe_allow_html=True)

    # Session windows timeline
    with st.expander("📅 0DTE Session Windows (all times ET)", expanded=False):
        for w in get_all_session_windows():
            st.markdown(f'<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span class="mono" style="font-size:10px;color:{w["color"]};font-weight:600;width:90px;">{w["start"]}–{w["end"]}</span><span style="font-family:\'DM Sans\';font-size:11px;color:rgba(255,255,255,0.5);"><strong style="color:{w["color"]};">{w["name"]}</strong> — {w["action"]}</span></div>',unsafe_allow_html=True)

    # 0DTE analysis for each key ticker
    dte_tickers=get_0dte_tickers()
    # Make sure we have data for these
    dte_avail=[t for t in dte_tickers if t in data_dict]
    if not dte_avail:
        st.warning("No 0DTE ticker data available. Make sure SPY, QQQ, or IWM are in your watchlist.")
    else:
        # Fetch live quotes for 0DTE tickers (15s cache)
        live_q = fetch_live_quotes_batch(tuple(dte_avail)) if has_live_data() else {}
        if live_q:
            lq_time = next(iter(live_q.values())).updated_at if live_q else "N/A"
            st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;"><span class="ld" style="background:#34C759;width:5px;height:5px;"></span><span class="mono" style="font-size:9px;color:rgba(255,255,255,0.3);">LIVE QUOTES · Updated {lq_time} · 15s refresh</span></div>',unsafe_allow_html=True)

        cols=st.columns(len(dte_avail))
        for i,t in enumerate(dte_avail):
            # Get intraday data if available
            idict_local={}
            if enable_intraday:
                from utils.intraday import fetch_intraday_5m
                idf=fetch_intraday_5m(t, days=5)
                if idf is not None: idict_local[t]=idf

            analysis=compute_0dte_analysis(t, data_dict[t], idict_local.get(t))
            if not analysis:
                with cols[i]:
                    st.warning(f"No analysis for {t}")
                continue

            sig=signals.get(t)
            kp=king_profiles.get(t)
            a=analysis

            with cols[i]:
                with st.container(border=True):
                    # Header
                    orb_c="#34C759" if a.orb_broken=="above" else "#FF453A" if a.orb_broken=="below" else "#FFD60A"
                    orb_t="▲ ABOVE ORB" if a.orb_broken=="above" else "▼ BELOW ORB" if a.orb_broken=="below" else "◆ INSIDE ORB"
                    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;"><span style="font-family:\'DM Sans\';font-size:22px;font-weight:700;color:#F5F5F7;">{t}</span><span class="b" style="background:{orb_c}18;color:{orb_c};font-family:\'JetBrains Mono\';font-size:9px;">{orb_t}</span></div>',unsafe_allow_html=True)

                    # Price — use live quote if available, else analysis price
                    lq = live_q.get(t)
                    display_price = lq.current if lq else a.current_price
                    display_change = lq.change_pct if lq else 0
                    price_color = "#34C759" if display_change >= 0 else "#FF453A"
                    live_badge = f'<span class="mono" style="font-size:8px;color:rgba(52,199,89,0.6);">● LIVE</span>' if lq else '<span class="mono" style="font-size:8px;color:rgba(255,255,255,0.2);">● DELAYED</span>'

                    st.markdown(f'<div style="display:flex;align-items:baseline;gap:8px;margin:6px 0;"><span class="mono" style="font-size:24px;font-weight:600;color:#F5F5F7;">${display_price:,.2f}</span><span class="mono" style="font-size:12px;color:{price_color};">{display_change:+.2f}%</span>{live_badge}</div>',unsafe_allow_html=True)
                    st.markdown(f'<div style="display:flex;gap:8px;font-family:\'JetBrains Mono\';font-size:10px;margin-bottom:8px;"><span style="color:rgba(255,255,255,0.3);">Expected Move</span><span style="color:#F5F5F7;">±${a.expected_move_1sd:.2f} ({a.expected_move_pct:.2f}%)</span></div>',unsafe_allow_html=True)

                    # Expected range bar
                    st.markdown(f'<div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;color:rgba(255,255,255,0.3);"><span>${a.expected_range_low:.2f}</span><span style="color:#F5F5F7;">← 1σ Range →</span><span>${a.expected_range_high:.2f}</span></div><div class="sb" style="margin-top:3px;"><div style="height:100%;width:50%;background:linear-gradient(90deg,#FF453A,rgba(255,255,255,0.1),#34C759);border-radius:3px;margin:0 auto;"></div></div>',unsafe_allow_html=True)

                    # Key levels
                    st.markdown(f'<div style="margin-top:10px;"><div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="color:rgba(255,255,255,0.3);">VWAP</span><span style="color:#0A84FF;font-weight:600;">${a.intraday_vwap:.2f}</span></div><div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="color:rgba(255,255,255,0.3);">ORB High</span><span style="color:#34C759;">${a.opening_range_high:.2f}</span></div><div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="color:rgba(255,255,255,0.3);">ORB Low</span><span style="color:#FF453A;">${a.opening_range_low:.2f}</span></div><div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="color:rgba(255,255,255,0.3);">Day High</span><span>${a.intraday_high:.2f}</span></div><div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;padding:3px 0;"><span style="color:rgba(255,255,255,0.3);">Day Low</span><span>${a.intraday_low:.2f}</span></div></div>',unsafe_allow_html=True)

                    # King nodes for this ticker
                    if kp and kp.all_nodes:
                        kings=[n for n in kp.all_nodes if n.node_type in ("king","gatekeeper")][:3]
                        if kings:
                            st.markdown('<div style="margin-top:8px;">',unsafe_allow_html=True)
                            for kn in kings:
                                kc="#0A84FF" if kn.node_type=="king" else "#FF9F0A"
                                ki="👑" if kn.node_type=="king" else "🛡"
                                st.markdown(f'<div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;padding:2px 0;"><span style="color:{kc};">{ki} {kn.node_type.title()}</span><span style="color:{kc};font-weight:600;">${kn.price_level:.2f}</span><span style="color:rgba(255,255,255,0.25);">{kn.distance_pct:+.1f}%</span></div>',unsafe_allow_html=True)
                            st.markdown('</div>',unsafe_allow_html=True)

                    # Regime
                    rc={"trending":"#34C759","range_bound":"#FFD60A","volatile":"#FF453A"}.get(a.regime,"#F5F5F7")
                    st.markdown(f'<div style="margin-top:10px;padding:8px 10px;background:{rc}08;border:1px solid {rc}20;border-radius:7px;"><span class="b" style="background:{rc}20;color:{rc};font-family:\'JetBrains Mono\';font-size:8px;">{a.regime.replace("_"," ").upper()}</span><div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.45);margin-top:4px;line-height:1.4;">{a.regime_strategy}</div></div>',unsafe_allow_html=True)

                    # Theta
                    st.markdown(f'<div style="margin-top:8px;display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;"><span style="color:rgba(255,255,255,0.3);">⏳ Theta/hr</span><span style="color:#FF9F0A;">${a.theta_per_hour:.2f}</span></div><div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\';font-size:9px;"><span style="color:rgba(255,255,255,0.3);">Time left</span><span>{a.minutes_to_close}min</span></div>',unsafe_allow_html=True)

        # Strategies section (full width)
        st.markdown('<p class="sl">0DTE Strategy Recommendations</p>',unsafe_allow_html=True)
        # Use the first available ticker's analysis
        if dte_avail:
            main_a = compute_0dte_analysis(dte_avail[0], data_dict[dte_avail[0]],
                                           idict_local.get(dte_avail[0]) if 'idict_local' in dir() else None)
            if main_a and main_a.strategies:
                scols=st.columns(min(3,len(main_a.strategies)))
                for si,(sname,sdesc,srisk) in enumerate(main_a.strategies[:3]):
                    risk_c="#34C759" if srisk=="conservative" else "#FFD60A" if srisk=="moderate" else "#FF453A" if srisk=="aggressive" else "rgba(255,255,255,0.4)"
                    with scols[si%3]:
                        st.markdown(f'<div class="kc"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;"><span style="font-family:\'DM Sans\';font-size:13px;font-weight:700;color:#F5F5F7;">{sname}</span><span class="b" style="background:{risk_c}18;color:{risk_c};font-family:\'JetBrains Mono\';font-size:8px;">{srisk}</span></div><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.5);line-height:1.5;">{sdesc}</div></div>',unsafe_allow_html=True)

        # Risk management
        st.markdown('<p class="sl">⚠ 0DTE Risk Rules</p>',unsafe_allow_html=True)
        if main_a:
            r1,r2,r3=st.columns(3)
            r1.markdown(f'<div class="kc"><div class="kl">Max Position Size</div><div class="kv">{main_a.max_position_size_pct}%</div><div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.3);margin-top:2px;">of account per trade</div></div>',unsafe_allow_html=True)
            r2.markdown(f'<div class="kc"><div class="kl">Stop Loss Rule</div><div style="font-family:\'DM Sans\';font-size:11px;color:rgba(255,255,255,0.5);margin-top:4px;">{main_a.suggested_stop}</div></div>',unsafe_allow_html=True)
            r3.markdown(f'<div class="kc"><div class="kl">Time Stop</div><div style="font-family:\'DM Sans\';font-size:11px;color:#FF9F0A;margin-top:4px;">{main_a.time_stop}</div></div>',unsafe_allow_html=True)

# ═══ TAB 3: PENNY RUNNERS ════════════════════════════════════════
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

# ═══ TAB 3: FLOW (Unusual Whales) ═════════════════════════════════
with tab_flow:
    if not UW:
        st.markdown('<p class="sl">🐋 Unusual Whales — Live Options Flow, Dark Pool & Congressional Trades</p>',unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'DM Sans\';font-size:12px;color:rgba(255,255,255,0.4);margin-bottom:14px;line-height:1.6;">Connect Unusual Whales to see real-time institutional options flow, dark pool activity, and what Congress is trading. This is the raw data professional traders use to front-run institutional moves.</div>',unsafe_allow_html=True)
        st.info("🐋 Add `UW_API_KEY` in Streamlit secrets to enable this tab. Get your key at **unusualwhales.com/settings/api-dashboard**")
    else:
        st.markdown('<p class="sl">🐋 Live Market Intelligence — Unusual Whales</p>',unsafe_allow_html=True)

        fl_col1,fl_col2,fl_col3=st.columns(3)

        # ── Options Flow Alerts ────────────────────────────
        with fl_col1:
            st.markdown('<p class="sl" style="margin-top:0;">⚡ Options Flow Alerts</p>',unsafe_allow_html=True)
            flow_alerts=fetch_flow_alerts(limit=15)
            if flow_alerts:
                for fa in flow_alerts[:10]:
                    sc_c="#34C759" if fa.sentiment=="bullish" else "#FF453A" if fa.sentiment=="bearish" else "rgba(255,255,255,0.4)"
                    prem_fmt=f"${fa.premium:,.0f}" if fa.premium>=1000 else f"${fa.premium:.0f}"
                    st.markdown(
                        f'<div class="nc {"bullish" if fa.sentiment=="bullish" else "bearish" if fa.sentiment=="bearish" else ""}" style="padding:8px 10px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div><span style="font-family:\'DM Sans\';font-size:14px;font-weight:700;color:#F5F5F7;">{fa.ticker}</span>'
                        f' <span class="b" style="background:{sc_c}18;color:{sc_c};font-family:\'JetBrains Mono\';font-size:8px;">{fa.option_type.upper()} {fa.sentiment}</span></div>'
                        f'<span class="mono" style="font-size:12px;font-weight:600;color:{sc_c};">{prem_fmt}</span></div>'
                        f'<div style="font-family:\'JetBrains Mono\';font-size:9px;color:rgba(255,255,255,0.35);margin-top:3px;">'
                        f'${fa.strike:.0f} {fa.option_type.upper()} exp {fa.expiry[:10]} · vol {fa.volume:,} · OI {fa.open_interest:,}</div>'
                        f'<div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.25);margin-top:2px;">{fa.alert_rule}</div>'
                        f'</div>',unsafe_allow_html=True)
            else:
                st.caption("No flow alerts available. May be outside market hours or API rate limited.")

        # ── Dark Pool ─────────────────────────────────────
        with fl_col2:
            st.markdown('<p class="sl" style="margin-top:0;">🌑 Dark Pool Activity</p>',unsafe_allow_html=True)
            dp_trades=fetch_dark_pool(limit=12)
            if dp_trades:
                for dp in dp_trades[:10]:
                    not_fmt=f"${dp.notional/1_000_000:.1f}M" if dp.notional>=1_000_000 else f"${dp.notional/1000:.0f}K"
                    st.markdown(
                        f'<div class="kc" style="margin-bottom:4px;padding:7px 10px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span style="font-family:\'DM Sans\';font-size:13px;font-weight:700;color:#F5F5F7;">{dp.ticker}</span>'
                        f'<span class="mono" style="font-size:11px;color:#0A84FF;font-weight:600;">{not_fmt}</span></div>'
                        f'<div style="font-family:\'JetBrains Mono\';font-size:9px;color:rgba(255,255,255,0.3);margin-top:2px;">'
                        f'{dp.size:,} shares @ ${dp.price:.2f}</div>'
                        f'</div>',unsafe_allow_html=True)
            else:
                st.caption("No dark pool data available.")

        # ── Congressional Trades ──────────────────────────
        with fl_col3:
            st.markdown('<p class="sl" style="margin-top:0;">🏛️ Congressional Trades</p>',unsafe_allow_html=True)
            congress=fetch_congress_trades(limit=12)
            if congress:
                for ct in congress[:10]:
                    tx_c="#34C759" if "purchase" in ct.transaction_type.lower() else "#FF453A"
                    tx_icon="🟢" if "purchase" in ct.transaction_type.lower() else "🔴"
                    st.markdown(
                        f'<div class="kc" style="margin-bottom:4px;padding:7px 10px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div><span style="font-size:11px;">{tx_icon}</span> <span style="font-family:\'DM Sans\';font-size:12px;font-weight:600;color:#F5F5F7;">{ct.ticker}</span></div>'
                        f'<span class="mono" style="font-size:10px;color:{tx_c};">{ct.transaction_type.title()}</span></div>'
                        f'<div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.35);margin-top:2px;">{ct.politician} · {ct.amount} · {ct.date[:10]}</div>'
                        f'</div>',unsafe_allow_html=True)
            else:
                st.caption("No congressional trade data available.")

        # ── UW Stock Screener for Penny Stocks ────────────
        st.markdown('<p class="sl">🔍 UW Penny Stock Screener (Volume Leaders Under $5)</p>',unsafe_allow_html=True)
        uw_pennies=fetch_stock_screener(min_volume=200000,max_price=5.0)
        if uw_pennies:
            cols=st.columns(3)
            for i,res in enumerate(uw_pennies[:12]):
                cc="#34C759" if res.change_pct>=0 else "#FF453A"
                with cols[i%3]:
                    st.markdown(
                        f'<div class="kc" style="margin-bottom:4px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span style="font-family:\'DM Sans\';font-size:14px;font-weight:700;color:#F5F5F7;">{res.ticker}</span>'
                        f'<span class="mono" style="font-size:12px;font-weight:600;color:{cc};">{res.change_pct:+.1f}%</span></div>'
                        f'<div style="font-family:\'JetBrains Mono\';font-size:9px;color:rgba(255,255,255,0.3);margin-top:2px;">'
                        f'${res.price:.3f} · Vol {res.volume/1_000_000:.1f}M · MCap ${res.market_cap/1_000_000:.0f}M</div>'
                        f'</div>',unsafe_allow_html=True)
        else:
            st.caption("No screener results. UW screener may not be available on free tier.")

# ═══ TAB 4: BREAKOUT ═════════════════════════════════════════════
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
            for kn in kp.all_nodes[:5]:
                lc="#0A84FF" if kn.node_type=="king" else "#FFD60A" if kn.polarity=="pika" else "#BF5AF2" if kn.polarity=="barney" else "rgba(255,255,255,0.15)"
                lw=2 if kn.node_type=="king" else 1
                icon="👑" if kn.node_type=="king" else "🟡" if kn.polarity=="pika" else "🟣" if kn.polarity=="barney" else "🛡"
                fig.add_hline(y=kn.price_level,line_dash="dot",line_color=lc,line_width=lw,annotation_text=f"{icon} ${kn.price_level:.2f} ({kn.volume_pct:.0f}%)",annotation_font_size=9,annotation_font_color=lc,row=1,col=1)
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
            if play.hold_duration:
                st.markdown(f'<div style="display:grid;grid-template-columns:auto 1fr;gap:8px;margin-top:8px;"><div class="kc" style="text-align:center;"><div class="kl">Hold</div><div class="mono" style="font-size:14px;font-weight:700;color:{t_c};">{play.hold_duration}</div></div><div class="kc"><div style="font-family:\'DM Sans\';font-size:10px;color:rgba(255,255,255,0.45);line-height:1.4;">{play.hold_reasoning}</div></div></div>',unsafe_allow_html=True)
            if play.price_drivers:
                for f,e,d in play.price_drivers[:3]:
                    st.markdown(f'<div class="dr"><span style="font-size:12px;">{e}</span><div><div style="font-family:\'DM Sans\';font-size:10px;font-weight:600;color:#F5F5F7;">{f}</div><div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.35);">{d}</div></div></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="tb" style="margin-top:6px;">{play.thesis}</div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

# ═══ TAB 6: SKYLIT-STYLE HEATMAP ═════════════════════════════════
with tab_heat:
    st.markdown('<p class="sl">👑 Heatseeker-Style Volume Profile · King Nodes · Gatekeepers · Air Pockets</p>',unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Sans\';font-size:11px;color:rgba(255,255,255,0.35);margin-bottom:12px;line-height:1.5;">Inspired by Skylit\'s Heatseeker framework. <span style="color:#FFD60A;">🟡 Pika zones</span> absorb price (stabilizing — magnetic pillow). <span style="color:#BF5AF2;">🟣 Barney zones</span> amplify price (volatile — gasoline on fire). <span style="color:#0A84FF;">👑 King Nodes</span> = highest volume — price gravitates here. <span style="color:#FF9F0A;">🛡 Gatekeepers</span> = barriers between price and King. <span style="color:rgba(255,255,255,0.5);">💨 Air Pockets</span> = low volume — price flies through.</div>',unsafe_allow_html=True)

    kn_sel=st.selectbox("Ticker",list(king_profiles.keys()),format_func=lambda t:f"{t} — {get_ticker_info(t)[0]}",key="kn_s")
    kp=king_profiles.get(kn_sel)
    if kp:
        sig=signals[kn_sel]

        # Regime badge
        regime_colors={"range_bound":("#FFD60A","rgba(255,214,10,0.12)"),"volatile":("#BF5AF2","rgba(191,90,242,0.12)"),"trending":("#0A84FF","rgba(10,132,255,0.12)")}
        rc,rb=regime_colors.get(kp.regime,("#F5F5F7","rgba(255,255,255,0.06)"))
        st.markdown(f'<div style="padding:10px 14px;background:{rb};border:1px solid {rc}30;border-radius:10px;margin-bottom:14px;"><div style="display:flex;align-items:center;gap:8px;"><span class="b" style="background:{rc}25;color:{rc};font-family:\'JetBrains Mono\';font-size:9px;">{kp.regime.replace("_"," ").upper()}</span><span style="font-family:\'DM Sans\';font-size:12px;color:rgba(255,255,255,0.6);">{kp.regime_description}</span></div></div>',unsafe_allow_html=True)

        # Volume profile chart with Skylit-style coloring
        fig=go.Figure()
        bar_colors=[]
        mean_v=np.mean(kp.volumes);std_v=np.std(kp.volumes)
        for lvl,vol in zip(kp.levels,kp.volumes):
            if vol>=mean_v+2*std_v:
                bar_colors.append("#0A84FF")  # King node
            elif vol>=mean_v+1.5*std_v:
                # Check if Pika or Barney
                node=next((n for n in kp.all_nodes if abs(n.price_level-lvl)<0.01),None)
                if node and node.polarity=="pika":
                    bar_colors.append("#FFD60A")  # Pika — stabilizing
                elif node and node.polarity=="barney":
                    bar_colors.append("#BF5AF2")  # Barney — volatile
                else:
                    bar_colors.append("#FF9F0A")  # Gatekeeper
            elif vol>=mean_v+std_v:
                bar_colors.append("rgba(255,159,10,0.5)")  # Gatekeeper/cluster
            elif vol<mean_v-0.5*std_v:
                bar_colors.append("rgba(255,255,255,0.08)")  # Air pocket
            elif lvl<kp.current_price:
                bar_colors.append("rgba(52,199,89,0.3)")
            else:
                bar_colors.append("rgba(255,69,58,0.3)")

        fig.add_trace(go.Bar(y=[round(l,2) for l in kp.levels],x=kp.volumes,orientation='h',marker_color=bar_colors,marker_line_width=0,hovertemplate="$%{y:.2f} — Vol: %{x:,.0f}<extra></extra>"))
        fig.add_hline(y=kp.current_price,line_color="#F5F5F7",line_width=2,annotation_text=f"NOW ${kp.current_price:.2f}",annotation_font_size=10,annotation_font_color="#F5F5F7")
        fig.add_hline(y=kp.poc,line_color="#0A84FF",line_width=2,line_dash="dot",annotation_text=f"POC ${kp.poc:.2f}",annotation_font_size=10,annotation_font_color="#0A84FF",annotation_position="bottom right")
        fig.add_hrect(y0=kp.val,y1=kp.vah,fillcolor="rgba(10,132,255,0.05)",line_width=0)
        # Mark air pockets
        for ap in kp.air_pockets[:3]:
            fig.add_hline(y=ap.price_level,line_color="rgba(255,255,255,0.15)",line_width=1,line_dash="dash")
        # Mark king node
        if kp.king_node:
            fig.add_hline(y=kp.king_node.price_level,line_color="#0A84FF",line_width=3,annotation_text=f"👑 KING ${kp.king_node.price_level:.2f}",annotation_font_size=11,annotation_font_color="#0A84FF")

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans",color="rgba(255,255,255,0.5)",size=10),margin=dict(l=0,r=0,t=20,b=0),xaxis=dict(gridcolor="rgba(255,255,255,0.04)",title="Volume"),yaxis=dict(gridcolor="rgba(255,255,255,0.04)",title="Price"),height=450,showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key=f"vp_{kn_sel}")

        # Key levels
        kc1,kc2,kc3,kc4=st.columns(4)
        kc1.metric("👑 King Node",f"${kp.king_node.price_level:.2f}" if kp.king_node else "—")
        kc2.metric("POC",f"${kp.poc:.2f}")
        kc3.metric("Value Area High",f"${kp.vah:.2f}")
        kc4.metric("Value Area Low",f"${kp.val:.2f}")

        # Node cards — 2-column grid
        node_sections=[
            ("👑 King & Gatekeepers",[n for n in kp.all_nodes if n.node_type in ("king","gatekeeper")][:6]),
            ("🟡 Pika Zones (Stabilizing)",kp.pika_zones[:4]),
            ("🟣 Barney Zones (Volatile)",kp.barney_zones[:4]),
            ("💨 Air Pockets (Fast Move)",kp.air_pockets[:4]),
        ]
        for section_title,nodes in node_sections:
            if not nodes: continue
            st.markdown(f'<p class="sl">{section_title}</p>',unsafe_allow_html=True)
            cols=st.columns(2)
            for i,n in enumerate(nodes):
                polarity_icon="🟡" if n.polarity=="pika" else "🟣" if n.polarity=="barney" else "💨" if n.node_type=="air_pocket" else "👑" if n.node_type=="king" else "🛡"
                nc="#FFD60A" if n.polarity=="pika" else "#BF5AF2" if n.polarity=="barney" else "#0A84FF" if n.node_type=="king" else "#FF9F0A" if n.node_type=="gatekeeper" else "rgba(255,255,255,0.4)"
                role_c="#34C759" if n.role=="support" else "#FF453A" if n.role=="resistance" else "#0A84FF"
                fresh="🔵 Fresh" if n.times_tested<3 else f"🔘 {n.times_tested}x tested"
                with cols[i%2]:
                    st.markdown(
                        f'<div class="kc" style="margin-bottom:6px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div><span style="font-size:13px;">{polarity_icon}</span> <span class="mono" style="font-size:13px;font-weight:600;color:{nc};">${n.price_level:.2f}</span>'
                        f'<span class="b" style="background:{role_c}18;color:{role_c};margin-left:5px;font-family:\'JetBrains Mono\';font-size:7px;">{n.role}</span></div>'
                        f'<span class="mono" style="font-size:10px;color:rgba(255,255,255,0.35);">{n.volume_pct:.1f}% vol · {n.distance_pct:+.1f}%</span></div>'
                        f'<div style="font-family:\'DM Sans\';font-size:9px;color:rgba(255,255,255,0.35);margin-top:3px;line-height:1.4;">{n.description}</div>'
                        f'<div style="font-family:\'JetBrains Mono\';font-size:8px;color:rgba(255,255,255,0.2);margin-top:2px;">{fresh} · Strength: {n.strength:.0f}/100</div>'
                        f'</div>',unsafe_allow_html=True)

        # All-tickers summary
        st.markdown('<p class="sl">All Tickers — Node Summary</p>',unsafe_allow_html=True)
        kn_tbl=[]
        for t in sorted(king_profiles.keys()):
            vp=king_profiles[t]
            kings=[n for n in vp.all_nodes if n.node_type=="king"]
            gk=[n for n in vp.all_nodes if n.node_type=="gatekeeper"]
            ap=[n for n in vp.all_nodes if n.node_type=="air_pocket"]
            pk=[n for n in vp.all_nodes if n.polarity=="pika"]
            bn=[n for n in vp.all_nodes if n.polarity=="barney"]
            nearest=min(vp.all_nodes,key=lambda n:abs(n.distance_pct)) if vp.all_nodes else None
            kn_tbl.append({"Ticker":t,"Price":f"${vp.current_price:.2f}","POC":f"${vp.poc:.2f}","Regime":vp.regime.replace("_"," ").title(),"👑 Kings":len(kings),"🛡 GK":len(gk),"💨 Air":len(ap),"🟡 Pika":len(pk),"🟣 Barney":len(bn),"Nearest":f"${nearest.price_level:.2f} ({nearest.distance_pct:+.1f}%)" if nearest else "—"})
        st.dataframe(pd.DataFrame(kn_tbl),use_container_width=True,hide_index=True,height=min(380,40+len(kn_tbl)*34))
    else:
        st.warning("No volume profile data for this ticker.")

st.markdown('<div style="height:16px"></div><p style="font-size:8px;color:rgba(255,255,255,0.08);text-align:center;padding:10px 0;">Signal · yfinance + Finnhub · Skylit-inspired concepts · Educational use only. Not financial advice.</p>',unsafe_allow_html=True)
