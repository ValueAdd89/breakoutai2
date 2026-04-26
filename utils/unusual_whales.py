"""
Unusual Whales API Integration
===============================
Connects to the Unusual Whales API for institutional-grade data:
  • Options flow alerts (unusual activity, sweeps, blocks)
  • Dark pool trades
  • Flow alerts (high-conviction setups)
  • Stock screener
  • Congressional trading activity
  • News headlines

API Key priority:
  1. st.secrets["UW_API_KEY"]
  2. os.environ["UW_API_KEY"]
  3. None — module returns empty results, no crash

Free tier: delayed data, limited flow. Paid: real-time everything.
Docs: https://api.unusualwhales.com/docs
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import requests
from datetime import datetime
from dataclasses import dataclass, field

UW_BASE = "https://api.unusualwhales.com"


def _get_uw_key() -> str | None:
    """
    Get Unusual Whales API key. Checks (in order):
    1. st.secrets["UW_API_KEY"]
    2. st.secrets["UNUSUAL_WHALES_API_KEY"]
    3. os.environ["UW_API_KEY"]
    4. os.environ["UNUSUAL_WHALES_API_KEY"]
    Supports both names for compatibility with the official MCP server convention.
    """
    try:
        key = st.secrets.get("UW_API_KEY", None) or st.secrets.get("UNUSUAL_WHALES_API_KEY", None)
        if key: return str(key).strip()
    except Exception:
        pass
    return (os.environ.get("UW_API_KEY", "").strip()
            or os.environ.get("UNUSUAL_WHALES_API_KEY", "").strip()
            or None)


def _uw_get(endpoint: str, params: dict = None) -> dict | list | None:
    """Make authenticated GET request to UW API."""
    key = _get_uw_key()
    if not key:
        return None
    try:
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        r = requests.get(f"{UW_BASE}{endpoint}", headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FlowAlert:
    ticker: str
    alert_rule: str         # e.g. "Unusual Options Activity"
    sentiment: str          # "bullish" | "bearish" | "neutral"
    option_type: str        # "call" | "put"
    strike: float
    expiry: str
    premium: float          # total premium $
    volume: int
    open_interest: int
    timestamp: str
    description: str = ""


@dataclass
class DarkPoolTrade:
    ticker: str
    price: float
    size: int
    notional: float         # $ value
    timestamp: str
    exchange: str = ""


@dataclass
class CongressTrade:
    politician: str
    ticker: str
    transaction_type: str   # "purchase" | "sale"
    amount: str             # "$1,001 - $15,000" etc.
    date: str
    party: str = ""
    chamber: str = ""


@dataclass
class UWScreenerResult:
    ticker: str
    price: float
    change_pct: float
    volume: int
    market_cap: float
    sector: str = ""
    avg_volume: int = 0
    short_interest: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# API functions
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
def fetch_flow_alerts(limit: int = 20) -> list[FlowAlert]:
    """Fetch recent options flow alerts (unusual activity)."""
    data = _uw_get("/api/option-trades/flow-alerts")
    if not data:
        return []
    alerts = []
    items = data if isinstance(data, list) else data.get("data", data.get("alerts", []))
    for item in items[:limit]:
        try:
            if isinstance(item, dict):
                alerts.append(FlowAlert(
                    ticker=item.get("ticker", item.get("underlying_symbol", "")),
                    alert_rule=item.get("alert_rule", item.get("rule", "Flow Alert")),
                    sentiment=_infer_sentiment(item),
                    option_type=item.get("put_call", item.get("option_type", "")).lower() or "call",
                    strike=float(item.get("strike", item.get("strike_price", 0)) or 0),
                    expiry=str(item.get("expires", item.get("expiry", item.get("expiration_date", "")))),
                    premium=float(item.get("premium", item.get("total_premium", 0)) or 0),
                    volume=int(item.get("volume", item.get("size", 0)) or 0),
                    open_interest=int(item.get("open_interest", item.get("oi", 0)) or 0),
                    timestamp=str(item.get("created_at", item.get("timestamp", item.get("executed_at", "")))),
                    description=item.get("description", item.get("alert_rule", "")),
                ))
        except Exception:
            continue
    return alerts


@st.cache_data(ttl=120, show_spinner=False)
def fetch_dark_pool(ticker: str = None, limit: int = 15) -> list[DarkPoolTrade]:
    """Fetch recent dark pool trades."""
    endpoint = f"/api/darkpool/{ticker}" if ticker else "/api/darkpool/recent"
    data = _uw_get(endpoint)
    if not data:
        return []
    trades = []
    items = data if isinstance(data, list) else data.get("data", [])
    for item in items[:limit]:
        try:
            if isinstance(item, dict):
                price = float(item.get("price", item.get("avg_price", 0)) or 0)
                size = int(item.get("size", item.get("volume", item.get("shares", 0))) or 0)
                trades.append(DarkPoolTrade(
                    ticker=item.get("ticker", item.get("symbol", ticker or "")),
                    price=price,
                    size=size,
                    notional=price * size,
                    timestamp=str(item.get("tracking_timestamp", item.get("executed_at", item.get("date", "")))),
                    exchange=item.get("market_center", item.get("exchange", "")),
                ))
        except Exception:
            continue
    return trades


@st.cache_data(ttl=300, show_spinner=False)
def fetch_congress_trades(limit: int = 15) -> list[CongressTrade]:
    """Fetch recent congressional trades."""
    data = _uw_get("/api/congress/recent-trades")
    if not data:
        return []
    trades = []
    items = data if isinstance(data, list) else data.get("data", [])
    for item in items[:limit]:
        try:
            if isinstance(item, dict):
                trades.append(CongressTrade(
                    politician=item.get("politician", item.get("name", item.get("representative", "Unknown"))),
                    ticker=item.get("ticker", item.get("symbol", "")),
                    transaction_type=item.get("transaction_type", item.get("type", "")),
                    amount=item.get("amount", item.get("range", "")),
                    date=str(item.get("transaction_date", item.get("date", item.get("traded_at", "")))),
                    party=item.get("party", ""),
                    chamber=item.get("chamber", item.get("office", "")),
                ))
        except Exception:
            continue
    return trades


@st.cache_data(ttl=180, show_spinner=False)
def fetch_uw_news(limit: int = 10) -> list[dict]:
    """Fetch market news headlines."""
    data = _uw_get("/api/news/headlines", params={"limit": limit})
    if not data:
        return []
    items = data if isinstance(data, list) else data.get("data", [])
    return items[:limit]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_screener(min_volume: int = 500000, max_price: float = 5.0) -> list[UWScreenerResult]:
    """Screen for penny stocks via UW screener."""
    data = _uw_get("/api/screener/stocks", params={
        "max_market_price": max_price,
        "min_volume": min_volume,
        "order": "desc",
        "order_by": "volume",
        "limit": 30,
    })
    if not data:
        return []
    results = []
    items = data if isinstance(data, list) else data.get("data", [])
    for item in items:
        try:
            if isinstance(item, dict):
                results.append(UWScreenerResult(
                    ticker=item.get("ticker", item.get("symbol", "")),
                    price=float(item.get("last", item.get("price", item.get("close", 0))) or 0),
                    change_pct=float(item.get("change_percent", item.get("price_change_pct", 0)) or 0),
                    volume=int(item.get("volume", 0) or 0),
                    market_cap=float(item.get("market_cap", item.get("marketcap", 0)) or 0),
                    sector=item.get("sector", ""),
                    avg_volume=int(item.get("avg_30_volume", item.get("average_volume", 0)) or 0),
                ))
        except Exception:
            continue
    return results


def _infer_sentiment(item: dict) -> str:
    """Infer sentiment from flow alert data."""
    put_call = str(item.get("put_call", item.get("option_type", ""))).lower()
    bid_ask = str(item.get("bid_ask", item.get("aggressor", item.get("side", "")))).lower()
    if put_call == "call" and "ask" in bid_ask:
        return "bullish"
    if put_call == "put" and "ask" in bid_ask:
        return "bearish"
    if put_call == "call" and "bid" in bid_ask:
        return "bearish"
    if put_call == "put" and "bid" in bid_ask:
        return "bullish"
    if put_call == "call":
        return "bullish"
    if put_call == "put":
        return "bearish"
    return "neutral"


def has_uw_key() -> bool:
    return bool(_get_uw_key())
