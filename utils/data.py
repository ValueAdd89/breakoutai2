"""
Market data — batch optimized.
Penny stock classification is DYNAMIC based on actual price data:
  - Penny stock: current price < $5 (SEC definition)
  - True penny / sub-dollar: current price < $1
  - Runner candidate: stock that has languished under $1 for 12+ months
    but shows signs it could break out to $5+
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Watchlists ────────────────────────────────────────────────────
# Large caps for the main scanner
LARGE_CAP_TICKERS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "V", "UNH", "LLY", "AVGO",
]

# Penny tickers: dynamically discovered each session by the screener.
# This static list is only a FALLBACK if the screener can't run.
PENNY_TICKERS_FALLBACK = [
    "DNA", "PSNY", "LCID", "PLUG", "OPEN", "SIRI", "GRAB", "NKLA",
    "GSAT", "TELL", "BTBT", "CIFR", "WULF", "ANY", "GOEV", "SNDL", "TLRY",
    "CLOV", "FCEL", "MULN", "FFIE", "WKHS", "QUBT", "SOUN",
]

# Will be populated dynamically in app.py via the screener module
PENNY_TICKERS = PENNY_TICKERS_FALLBACK

DEFAULT_TICKERS = LARGE_CAP_TICKERS + PENNY_TICKERS_FALLBACK

TICKER_META = {
    "AAPL": ("Apple Inc.", "Technology"), "NVDA": ("NVIDIA Corp.", "Semiconductors"),
    "MSFT": ("Microsoft Corp.", "Technology"), "GOOGL": ("Alphabet Inc.", "Technology"),
    "AMZN": ("Amazon.com Inc.", "Consumer"), "META": ("Meta Platforms", "Technology"),
    "TSLA": ("Tesla Inc.", "Automotive"), "JPM": ("JPMorgan Chase", "Finance"),
    "V": ("Visa Inc.", "Finance"), "UNH": ("UnitedHealth", "Healthcare"),
    "LLY": ("Eli Lilly", "Healthcare"), "AVGO": ("Broadcom Inc.", "Semiconductors"),
    "DNA": ("Ginkgo Bioworks", "Biotech"), "PSNY": ("Polestar Auto", "EV"),
    "LCID": ("Lucid Group", "EV"), "PLUG": ("Plug Power", "Clean Energy"),
    "OPEN": ("Opendoor Tech", "Real Estate"), "SIRI": ("Sirius XM", "Media"),
    "GRAB": ("Grab Holdings", "Fintech"), "NKLA": ("Nikola Corp", "EV"),
    "GSAT": ("Globalstar Inc.", "Telecom"), "TELL": ("Tellurian Inc.", "Energy"),
    "ASTS": ("AST SpaceMobile", "Telecom"), "BTBT": ("Bit Digital", "Crypto Mining"),
    "CIFR": ("Cipher Mining", "Crypto Mining"), "WULF": ("TeraWulf Inc.", "Crypto Mining"),
    "ANY": ("Sphere 3D Corp", "Crypto Mining"), "GOEV": ("Canoo Inc.", "EV"),
    "VLD": ("Velo3D Inc.", "3D Printing"), "MVST": ("Microvast Holdings", "Battery"),
    "QS": ("QuantumScape", "Battery"), "ORGN": ("Origin Materials", "Materials"),
    "APRE": ("Aprea Therapeutics", "Biotech"), "BIVI": ("BioVie Inc.", "Biotech"),
    "CLOV": ("Clover Health", "Healthcare"), "SNDL": ("SNDL Inc.", "Cannabis"),
    "TLRY": ("Tilray Brands", "Cannabis"),
}


# ── Refresh intervals ────────────────────────────────────────────
REFRESH_INTERVALS = {
    "30 seconds": 30_000, "1 minute": 60_000, "2 minutes": 120_000,
    "5 minutes": 300_000, "15 minutes": 900_000,
}
CACHE_TTL_MAP = {
    "30 seconds": 25, "1 minute": 55, "2 minutes": 110,
    "5 minutes": 280, "15 minutes": 850,
}


# ── Dynamic penny stock classification ───────────────────────────

def classify_stock(ticker: str, df: pd.DataFrame) -> dict:
    """
    Classify a stock dynamically from its price history.
    Returns dict with:
      tier: "Sub-Dollar" | "Penny (<$5)" | "Small Cap" | "Large Cap"
      is_penny: True if price < $5
      is_sub_dollar: True if price < $1
      is_runner_candidate: True if languished under $1 for 12+ months
      months_under_1: how many months the stock averaged under $1
      current_price: latest price
    """
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    current = float(close.iloc[-1])

    # Calculate how long the stock has been under $1
    # Use monthly resampled means over available history
    monthly = close.resample("ME").mean().dropna()
    months_under_1 = int((monthly < 1.0).sum())
    total_months = len(monthly)

    # Runner candidate: averaged under $1 for 12+ of the available months
    is_sub_dollar = current < 1.0
    is_penny = current < 5.0
    is_runner = months_under_1 >= 12 or (months_under_1 >= 6 and is_sub_dollar)

    if current < 1.0:
        tier = "Sub-Dollar"
    elif current < 5.0:
        tier = "Penny (<$5)"
    elif current < 20.0:
        tier = "Small Cap"
    else:
        tier = "Large Cap"

    return {
        "tier": tier,
        "is_penny": is_penny,
        "is_sub_dollar": is_sub_dollar,
        "is_runner_candidate": is_runner,
        "months_under_1": months_under_1,
        "total_months": total_months,
        "current_price": round(current, 4),
    }


# ── Column cleaning ──────────────────────────────────────────────

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _split_multi_download(raw_df, tickers, min_bars=40):
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
                    if len(tdf) >= min_bars:
                        results[t] = tdf
                except Exception:
                    continue
    else:
        if len(tickers) == 1 and len(raw_df) >= min_bars:
            df = _clean_cols(raw_df.copy())
            df.dropna(subset=["Close"], inplace=True)
            if len(df) >= min_bars:
                results[tickers[0]] = df
    return results


@st.cache_data(ttl=120, show_spinner=False)
def fetch_batch_data(tickers: tuple, period: str = "6mo") -> dict[str, pd.DataFrame]:
    ticker_list = list(tickers)
    MIN_BARS = 40
    try:
        raw = yf.download(" ".join(ticker_list), period=period, interval="1d",
                          progress=False, auto_adjust=True, timeout=20, threads=True)
        results = _split_multi_download(raw, ticker_list, min_bars=MIN_BARS)
    except Exception:
        results = {}

    failed = [t for t in ticker_list if t not in results]
    if failed:
        def _fetch_one(ticker):
            try:
                df = yf.download(ticker, period=period, interval="1d",
                                 progress=False, auto_adjust=True, timeout=10)
                if df.empty: return ticker, None
                df = _clean_cols(df)
                df.dropna(subset=["Close"], inplace=True)
                return (ticker, df) if len(df) >= MIN_BARS else (ticker, None)
            except Exception:
                return ticker, None

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in failed}
            for future in as_completed(futures, timeout=30):
                try:
                    t, df = future.result(timeout=12)
                    if df is not None: results[t] = df
                except Exception:
                    continue
    return results


def get_ticker_info(ticker: str) -> tuple[str, str]:
    """Return (name, sector). Tier is now dynamic from price data."""
    meta = TICKER_META.get(ticker, (ticker, "Unknown"))
    return meta[0], meta[1] if len(meta) > 1 else "Unknown"


def is_penny_stock(ticker: str) -> bool:
    """Static fallback — use classify_stock() for dynamic classification."""
    return ticker in PENNY_TICKERS
