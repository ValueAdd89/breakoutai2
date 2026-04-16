"""
News, Catalyst, and Company Profile module.

Data sources:
  1. Finnhub (free tier: 60 req/min) — news, earnings calendar, company profile
  2. yfinance fallback — shares outstanding, float, short ratio

API key priority (first match wins):
  1. st.secrets["FINNHUB_API_KEY"] (Streamlit Cloud)
  2. os.environ["FINNHUB_API_KEY"] (local .env / shell)
  3. None — module gracefully degrades, returns empty results
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field

try:
    import finnhub
    HAS_FINNHUB = True
except ImportError:
    HAS_FINNHUB = False


def _get_api_key() -> str | None:
    """Try Streamlit secrets first, then environment variable."""
    try:
        key = st.secrets.get("FINNHUB_API_KEY", None)
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.environ.get("FINNHUB_API_KEY", "").strip() or None


@st.cache_resource
def _get_client():
    """Singleton Finnhub client (cached across reruns)."""
    if not HAS_FINNHUB:
        return None
    key = _get_api_key()
    if not key:
        return None
    try:
        return finnhub.Client(api_key=key)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NewsItem:
    headline: str
    summary: str
    source: str
    datetime_utc: datetime
    url: str
    category: str = ""
    sentiment: str = "neutral"   # "bullish" | "bearish" | "neutral"
    importance: int = 0          # 0-3


@dataclass
class CompanyProfile:
    ticker: str
    name: str
    market_cap_m: float = 0.0         # in millions
    shares_outstanding_m: float = 0.0 # in millions
    float_shares_m: float = 0.0       # in millions (from yfinance)
    short_ratio: float = 0.0
    short_pct_float: float = 0.0
    industry: str = ""
    exchange: str = ""
    ipo: str = ""
    logo: str = ""
    weburl: str = ""


@dataclass
class EarningsEvent:
    ticker: str
    date: str
    hour: str          # "bmo" | "amc" | "dmh"
    eps_estimate: float | None = None
    revenue_estimate: float | None = None


# ═══════════════════════════════════════════════════════════════════
# News fetching with sentiment classification
# ═══════════════════════════════════════════════════════════════════

BULLISH_KEYWORDS = {
    "beat", "beats", "raises", "raised", "upgrades", "upgrade", "surges",
    "soars", "rallies", "breakout", "record high", "all-time high",
    "approval", "approved", "wins", "partnership", "acquires", "acquired",
    "buyback", "dividend increase", "outperform", "strong buy", "bullish",
    "crushes", "exceeds", "blowout", "launches", "expands", "deal signed",
    "fda cleared", "fda approves", "contract awarded", "order received",
}

BEARISH_KEYWORDS = {
    "misses", "miss", "cuts", "downgrades", "downgrade", "plunges",
    "tumbles", "drops", "falls", "crashes", "layoffs", "bankruptcy",
    "investigation", "lawsuit", "fraud", "sec probe", "delisting",
    "halts", "recall", "warning", "underperform", "sell rating",
    "bearish", "disappoints", "slashed", "restructuring", "loss widens",
    "dilution", "offering", "goes offline", "delay", "rejected",
}

HIGH_IMPORTANCE_KEYWORDS = {
    "fda", "sec", "earnings", "acquisition", "merger", "halt",
    "bankruptcy", "buyback", "dividend", "guidance", "ceo", "layoff",
    "lawsuit", "recall", "approval", "contract", "partnership",
}


def _classify_sentiment(headline: str, summary: str = "") -> str:
    text = (headline + " " + summary).lower()
    bull_score = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bear_score = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    if bull_score > bear_score and bull_score > 0:
        return "bullish"
    if bear_score > bull_score and bear_score > 0:
        return "bearish"
    return "neutral"


def _classify_importance(headline: str, summary: str = "") -> int:
    text = (headline + " " + summary).lower()
    hits = sum(1 for kw in HIGH_IMPORTANCE_KEYWORDS if kw in text)
    if hits >= 3:
        return 3
    if hits == 2:
        return 2
    if hits == 1:
        return 1
    return 0


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(ticker: str, days_back: int = 7, max_items: int = 10) -> list[NewsItem]:
    """Fetch recent news for a ticker from Finnhub."""
    client = _get_client()
    if client is None:
        return []

    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days_back)
        raw_news = client.company_news(
            ticker, _from=start.isoformat(), to=end.isoformat()
        )
        items = []
        for n in raw_news[:max_items]:
            try:
                headline = n.get("headline", "")
                summary = n.get("summary", "") or ""
                if not headline:
                    continue
                items.append(NewsItem(
                    headline=headline,
                    summary=summary[:400],
                    source=n.get("source", "Unknown"),
                    datetime_utc=datetime.fromtimestamp(n.get("datetime", 0)),
                    url=n.get("url", ""),
                    category=n.get("category", ""),
                    sentiment=_classify_sentiment(headline, summary),
                    importance=_classify_importance(headline, summary),
                ))
            except Exception:
                continue
        items.sort(key=lambda x: (x.importance, x.datetime_utc), reverse=True)
        return items
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# Company profile with float / short interest
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_profile(ticker: str) -> CompanyProfile | None:
    """Combine Finnhub profile + yfinance float/short data."""
    profile = CompanyProfile(ticker=ticker, name=ticker)

    # Finnhub profile (market cap, shares outstanding, industry)
    client = _get_client()
    if client is not None:
        try:
            prof = client.company_profile2(symbol=ticker)
            if prof:
                profile.name = prof.get("name", ticker)
                profile.market_cap_m = float(prof.get("marketCapitalization", 0) or 0)
                profile.shares_outstanding_m = float(prof.get("shareOutstanding", 0) or 0)
                profile.industry = prof.get("finnhubIndustry", "") or ""
                profile.exchange = prof.get("exchange", "") or ""
                profile.ipo = prof.get("ipo", "") or ""
                profile.logo = prof.get("logo", "") or ""
                profile.weburl = prof.get("weburl", "") or ""
        except Exception:
            pass

    # yfinance float + short data (these aren't in Finnhub's free tier)
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        float_shares = info.get("floatShares") or info.get("sharesOutstanding")
        if float_shares:
            profile.float_shares_m = float(float_shares) / 1_000_000
        profile.short_ratio = float(info.get("shortRatio") or 0)
        short_pct = info.get("shortPercentOfFloat")
        if short_pct:
            profile.short_pct_float = float(short_pct) * 100
        if not profile.name or profile.name == ticker:
            profile.name = info.get("longName") or info.get("shortName") or ticker
        if not profile.shares_outstanding_m:
            so = info.get("sharesOutstanding")
            if so:
                profile.shares_outstanding_m = float(so) / 1_000_000
        if not profile.industry:
            profile.industry = info.get("industry", "") or ""
    except Exception:
        pass

    return profile


# ═══════════════════════════════════════════════════════════════════
# Earnings calendar
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_calendar(tickers: tuple, days_ahead: int = 14) -> list[EarningsEvent]:
    """Fetch upcoming earnings for a set of tickers."""
    client = _get_client()
    if client is None:
        return []

    try:
        end = datetime.utcnow().date() + timedelta(days=days_ahead)
        start = datetime.utcnow().date()
        cal = client.earnings_calendar(
            _from=start.isoformat(), to=end.isoformat(), symbol="", international=False
        )
        events = []
        for e in cal.get("earningsCalendar", []):
            sym = e.get("symbol", "")
            if sym in tickers:
                events.append(EarningsEvent(
                    ticker=sym,
                    date=e.get("date", ""),
                    hour=e.get("hour", ""),
                    eps_estimate=e.get("epsEstimate"),
                    revenue_estimate=e.get("revenueEstimate"),
                ))
        return events
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# Float-weighted volume scoring helper
# ═══════════════════════════════════════════════════════════════════

def float_turnover_rate(daily_volume: float, float_shares_m: float) -> float:
    """
    Compute what % of float traded today.
    Pro traders watch for >50% turnover as a major breakout signal.
    """
    if float_shares_m <= 0:
        return 0.0
    float_shares = float_shares_m * 1_000_000
    return (daily_volume / float_shares) * 100


def classify_float_tier(float_shares_m: float) -> str:
    """Categorize by float size — low float = bigger breakout potential."""
    if float_shares_m == 0:
        return "Unknown"
    if float_shares_m < 10:
        return "Micro Float"       # <10M shares — extreme volatility
    if float_shares_m < 50:
        return "Low Float"         # 10-50M — classic runner profile
    if float_shares_m < 200:
        return "Medium Float"      # 50-200M
    if float_shares_m < 1000:
        return "Large Float"       # 200M-1B
    return "Mega Float"            # 1B+
