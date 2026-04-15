"""
Market data fetching — PERFORMANCE OPTIMIZED.

Key optimizations:
  1. Single yf.download() call for ALL tickers at once (one HTTP batch)
  2. ThreadPoolExecutor fallback for retries on failed tickers
  3. Aggressive TTL caching at both batch and individual levels
  4. Timeout guards to prevent hanging on slow API responses
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# Default watchlist
DEFAULT_TICKERS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "V", "UNH", "LLY", "AVGO",
    "AMD", "NFLX", "CRM", "COST", "PEP", "INTC",
]

TICKER_META = {
    "AAPL": ("Apple Inc.", "Technology"),
    "NVDA": ("NVIDIA Corp.", "Semiconductors"),
    "MSFT": ("Microsoft Corp.", "Technology"),
    "GOOGL": ("Alphabet Inc.", "Technology"),
    "AMZN": ("Amazon.com Inc.", "Consumer"),
    "META": ("Meta Platforms", "Technology"),
    "TSLA": ("Tesla Inc.", "Automotive"),
    "JPM": ("JPMorgan Chase", "Finance"),
    "V": ("Visa Inc.", "Finance"),
    "UNH": ("UnitedHealth", "Healthcare"),
    "LLY": ("Eli Lilly", "Healthcare"),
    "AVGO": ("Broadcom Inc.", "Semiconductors"),
    "AMD": ("AMD Inc.", "Semiconductors"),
    "NFLX": ("Netflix Inc.", "Entertainment"),
    "CRM": ("Salesforce Inc.", "Technology"),
    "COST": ("Costco Wholesale", "Consumer"),
    "PEP": ("PepsiCo Inc.", "Consumer"),
    "INTC": ("Intel Corp.", "Semiconductors"),
}


def _split_multi_download(raw_df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Split a multi-ticker yf.download() result into per-ticker DataFrames.
    Handles both MultiIndex columns (multi-ticker) and flat columns (single-ticker).
    """
    results = {}
    if raw_df.empty:
        return results

    if isinstance(raw_df.columns, pd.MultiIndex):
        available = raw_df.columns.get_level_values(1).unique().tolist()
        for t in tickers:
            if t in available:
                try:
                    tdf = raw_df.xs(t, level=1, axis=1).copy()
                    tdf.dropna(subset=["Close"], inplace=True)
                    if len(tdf) > 60:
                        results[t] = tdf
                except Exception:
                    continue
    else:
        if len(tickers) == 1 and len(raw_df) > 60:
            df = raw_df.copy()
            df.dropna(subset=["Close"], inplace=True)
            if len(df) > 60:
                results[tickers[0]] = df
    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_data(tickers: tuple, period: str = "6mo") -> dict[str, pd.DataFrame]:
    """
    Fetch ALL tickers in a SINGLE yf.download() call.

    This is the #1 performance fix: yfinance batches multiple tickers into
    one HTTP request, cutting network time from 12×3s = 36s → ~4s total.
    Failed tickers are retried individually via ThreadPoolExecutor.
    """
    ticker_list = list(tickers)

    # Phase 1: Batch download (single HTTP call)
    try:
        raw = yf.download(
            " ".join(ticker_list),
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            timeout=15,
            threads=True,
        )
        results = _split_multi_download(raw, ticker_list)
    except Exception:
        results = {}

    # Phase 2: Retry failures individually (parallel threads)
    failed = [t for t in ticker_list if t not in results]
    if failed:
        def _fetch_one(ticker):
            try:
                df = yf.download(
                    ticker, period=period, interval="1d",
                    progress=False, auto_adjust=True, timeout=8,
                )
                if df.empty:
                    return ticker, None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(subset=["Close"], inplace=True)
                return (ticker, df) if len(df) > 60 else (ticker, None)
            except Exception:
                return ticker, None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in failed}
            for future in as_completed(futures, timeout=20):
                try:
                    t, df = future.result(timeout=10)
                    if df is not None:
                        results[t] = df
                except Exception:
                    continue

    return results


@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday(ticker: str) -> pd.DataFrame | None:
    """Fetch 5-minute intraday data for real-time view."""
    try:
        data = yf.download(
            ticker, period="5d", interval="5m",
            progress=False, auto_adjust=True, timeout=8,
        )
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None


def get_ticker_info(ticker: str) -> tuple[str, str]:
    """Return (name, sector) for a ticker."""
    return TICKER_META.get(ticker, (ticker, "Unknown"))
