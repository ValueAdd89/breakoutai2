"""
Market data fetching — batch optimized.
Single yf.download() for all tickers, parallel retry for failures.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Large-cap + Mid-cap watchlist ─────────────────────────────────
LARGE_CAP_TICKERS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "V", "UNH", "LLY", "AVGO",
]

# ── Penny / Small-cap breakout candidates ─────────────────────────
# High-volatility, low-float names that pro traders scan for breakouts
PENNY_TICKERS = [
    "SOFI", "PLTR", "NIO", "MARA", "RIOT", "LCID",
    "PLUG", "BB", "OPEN", "DNA", "IONQ", "QUBT",
    "SOUN", "RGTI", "BTBT", "HIMS", "CIFR", "PSNY",
]

DEFAULT_TICKERS = LARGE_CAP_TICKERS + PENNY_TICKERS

TICKER_META = {
    "AAPL": ("Apple Inc.", "Large Cap", "Technology"),
    "NVDA": ("NVIDIA Corp.", "Large Cap", "Semiconductors"),
    "MSFT": ("Microsoft Corp.", "Large Cap", "Technology"),
    "GOOGL": ("Alphabet Inc.", "Large Cap", "Technology"),
    "AMZN": ("Amazon.com Inc.", "Large Cap", "Consumer"),
    "META": ("Meta Platforms", "Large Cap", "Technology"),
    "TSLA": ("Tesla Inc.", "Large Cap", "Automotive"),
    "JPM": ("JPMorgan Chase", "Large Cap", "Finance"),
    "V": ("Visa Inc.", "Large Cap", "Finance"),
    "UNH": ("UnitedHealth", "Large Cap", "Healthcare"),
    "LLY": ("Eli Lilly", "Large Cap", "Healthcare"),
    "AVGO": ("Broadcom Inc.", "Large Cap", "Semiconductors"),
    "SOFI": ("SoFi Technologies", "Penny/Small", "Fintech"),
    "PLTR": ("Palantir Tech", "Penny/Small", "Software"),
    "NIO": ("NIO Inc.", "Penny/Small", "EV"),
    "MARA": ("MARA Holdings", "Penny/Small", "Crypto Mining"),
    "RIOT": ("Riot Platforms", "Penny/Small", "Crypto Mining"),
    "LCID": ("Lucid Group", "Penny/Small", "EV"),
    "PLUG": ("Plug Power", "Penny/Small", "Clean Energy"),
    "BB": ("BlackBerry Ltd", "Penny/Small", "Cybersecurity"),
    "OPEN": ("Opendoor Tech", "Penny/Small", "Real Estate"),
    "DNA": ("Ginkgo Bioworks", "Penny/Small", "Biotech"),
    "IONQ": ("IonQ Inc.", "Penny/Small", "Quantum"),
    "QUBT": ("Quantum Computing", "Penny/Small", "Quantum"),
    "SOUN": ("SoundHound AI", "Penny/Small", "AI"),
    "RGTI": ("Rigetti Computing", "Penny/Small", "Quantum"),
    "BTBT": ("Bit Digital Inc.", "Penny/Small", "Crypto Mining"),
    "HIMS": ("Hims & Hers", "Penny/Small", "Health"),
    "CIFR": ("Cipher Mining", "Penny/Small", "Crypto Mining"),
    "PSNY": ("Polestar Auto", "Penny/Small", "EV"),
}


# ── Refresh intervals (in seconds) ───────────────────────────────
REFRESH_INTERVALS = {
    "30 seconds": 30_000,
    "1 minute": 60_000,
    "2 minutes": 120_000,
    "5 minutes": 300_000,
    "15 minutes": 900_000,
}

CACHE_TTL_MAP = {
    "30 seconds": 25,
    "1 minute": 55,
    "2 minutes": 110,
    "5 minutes": 280,
    "15 minutes": 850,
}


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has flat, unique column names and 1-D columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Deduplicate any accidental column repeats
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _split_multi_download(raw_df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    results = {}
    if raw_df.empty:
        return results
    if isinstance(raw_df.columns, pd.MultiIndex):
        available = raw_df.columns.get_level_values(1).unique().tolist()
        for t in tickers:
            if t in available:
                try:
                    tdf = raw_df.xs(t, level=1, axis=1).copy()
                    tdf = _clean_cols(tdf)
                    tdf.dropna(subset=["Close"], inplace=True)
                    if len(tdf) > 60:
                        results[t] = tdf
                except Exception:
                    continue
    else:
        if len(tickers) == 1 and len(raw_df) > 60:
            df = _clean_cols(raw_df.copy())
            df.dropna(subset=["Close"], inplace=True)
            if len(df) > 60:
                results[tickers[0]] = df
    return results


@st.cache_data(ttl=120, show_spinner=False)
def fetch_batch_data(tickers: tuple, period: str = "6mo") -> dict[str, pd.DataFrame]:
    ticker_list = list(tickers)
    try:
        raw = yf.download(
            " ".join(ticker_list), period=period, interval="1d",
            progress=False, auto_adjust=True, timeout=15, threads=True,
        )
        results = _split_multi_download(raw, ticker_list)
    except Exception:
        results = {}

    failed = [t for t in ticker_list if t not in results]
    if failed:
        def _fetch_one(ticker):
            try:
                df = yf.download(ticker, period=period, interval="1d",
                                 progress=False, auto_adjust=True, timeout=8)
                if df.empty:
                    return ticker, None
                df = _clean_cols(df)
                df.dropna(subset=["Close"], inplace=True)
                return (ticker, df) if len(df) > 60 else (ticker, None)
            except Exception:
                return ticker, None

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in failed}
            for future in as_completed(futures, timeout=25):
                try:
                    t, df = future.result(timeout=10)
                    if df is not None:
                        results[t] = df
                except Exception:
                    continue
    return results


def get_ticker_info(ticker: str) -> tuple[str, str, str]:
    """Return (name, cap_tier, sector)."""
    return TICKER_META.get(ticker, (ticker, "Unknown", "Unknown"))


def is_penny_stock(ticker: str) -> bool:
    info = TICKER_META.get(ticker)
    return info is not None and info[1] == "Penny/Small"
