"""
Live Quotes Module — Real-Time Price Data
==========================================
Uses Finnhub REST API for real-time US stock quotes.
Free tier: 60 calls/min, no artificial delay on US stocks.

For Streamlit, we poll every 15-30 seconds (cache TTL=15s).
This gives near-real-time data — the best a polling architecture can do.

For true sub-second data, users should use a dedicated trading platform
alongside Signal. This module provides the "latest snapshot" approach
that works within Streamlit's request-response model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import requests
from datetime import datetime
from dataclasses import dataclass

try:
    import finnhub
    HAS_FINNHUB = True
except ImportError:
    HAS_FINNHUB = False


def _get_fh_key():
    try:
        key = st.secrets.get("FINNHUB_API_KEY", None)
        if key: return str(key).strip()
    except Exception:
        pass
    return os.environ.get("FINNHUB_API_KEY", "").strip() or None


@dataclass
class LiveQuote:
    ticker: str
    current: float          # current price
    change: float           # $ change
    change_pct: float       # % change
    high: float             # day high
    low: float              # day low
    open: float             # day open
    prev_close: float       # previous close
    timestamp: int          # unix timestamp
    updated_at: str         # human-readable time
    is_live: bool           # True if data is from today's session


@st.cache_data(ttl=15, show_spinner=False)
def fetch_live_quote(ticker: str) -> LiveQuote | None:
    """
    Fetch real-time quote from Finnhub.
    Returns current price, day high/low/open, change, and timestamp.
    15-second cache TTL for near-real-time updates.
    """
    key = _get_fh_key()
    if not key:
        return None

    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/quote",
            params={"symbol": ticker, "token": key},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or data.get("c", 0) == 0:
            return None

        ts = data.get("t", 0)
        return LiveQuote(
            ticker=ticker,
            current=float(data.get("c", 0)),
            change=float(data.get("d", 0) or 0),
            change_pct=float(data.get("dp", 0) or 0),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            open=float(data.get("o", 0)),
            prev_close=float(data.get("pc", 0)),
            timestamp=ts,
            updated_at=datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 0 else "N/A",
            is_live=ts > (datetime.now().timestamp() - 86400),
        )
    except Exception:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_live_quotes_batch(tickers: tuple) -> dict[str, LiveQuote]:
    """Fetch live quotes for multiple tickers. Rate-limited to avoid hitting 60/min."""
    results = {}
    for t in tickers[:20]:  # cap at 20 to stay under rate limit
        q = fetch_live_quote(t)
        if q:
            results[t] = q
    return results


def has_live_data() -> bool:
    """Check if live quote data is available (Finnhub key present)."""
    return bool(_get_fh_key())
