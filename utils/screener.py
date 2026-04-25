"""
Live Penny Stock Screener
=========================
Instead of hardcoded tickers, maintains a broad seed universe (~80 small-cap
symbols across biotech, EV, crypto, cannabis, tech, energy) and dynamically
filters each session to find:
  • Current price < $5 (penny stocks)
  • Current price < $1 (sub-dollar / true pennies)
  • Volume spike today (RVol > 1.5x)
  • Runner candidates (under $1 for extended periods)

The seed universe is intentionally large so that on any given day,
the screener surfaces the 10-20 names that are actually moving.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Broad seed universe of small/micro-cap stocks ────────────────
# These span biotech, EV, crypto mining, cannabis, clean energy,
# quantum, AI, fintech, real estate, and SPACs.
# Many trade under $5; some will be above $5 on any given day and
# will be filtered out dynamically.
SEED_UNIVERSE = [
    # Biotech / Pharma
    "DNA", "CLOV", "BIVI", "APRE", "NRXP", "SAVA", "AGEN", "VRPX",
    "IMVT", "ABEO", "PRTG", "CRIS", "TARS",
    # EV / Automotive
    "LCID", "PSNY", "NKLA", "GOEV", "FFIE", "MULN", "WKHS", "REE",
    "ARVL", "PTRA",
    # Crypto Mining / Blockchain
    "BTBT", "CIFR", "ANY", "CORZ", "IREN", "BTDR", "DGHI",
    # Cannabis
    "SNDL", "TLRY", "ACB", "CGC", "HEXO", "OGI", "GRNH",
    # Clean Energy / Battery
    "PLUG", "FCEL", "BE", "QS", "MVST", "ORGN", "WULF",
    "STEM",
    # Quantum / AI
    "QUBT", "RGTI", "SOUN", "BBAI", "GFAI",
    # Fintech / Tech
    "OPEN", "GRAB", "SOFI", "WISH", "ASTS",
    # Real Estate / SPAC shells
    "VLD", "GSAT", "TELL", "SIRI", "IVP",
    # Additional volatile small caps
    "BYRN", "KULR", "DM", "AEVA", "OUST", "VNET",
    "MAPS", "MNMD", "ATAI",
]


@st.cache_data(ttl=180, show_spinner=False)
def scan_penny_stocks(max_price: float = 5.0, min_volume: int = 100_000) -> dict:
    """
    Live scan: download current quotes for the seed universe,
    filter to penny stocks (< max_price), and return enriched data.
    
    Returns dict of ticker -> {price, change_pct, volume, avg_volume, rvol, is_sub_dollar}
    """
    results = {}
    tickers_str = " ".join(SEED_UNIVERSE)

    try:
        # Fast batch download — 1 day of data just to get current price + volume
        df = yf.download(tickers_str, period="5d", interval="1d",
                         progress=False, auto_adjust=True, timeout=15, threads=True)
        if df.empty:
            return results

        if isinstance(df.columns, pd.MultiIndex):
            available = df.columns.get_level_values(1).unique().tolist()
            for t in SEED_UNIVERSE:
                if t not in available:
                    continue
                try:
                    tdf = df.xs(t, level=1, axis=1)
                    if isinstance(tdf, pd.DataFrame) and tdf.columns.duplicated().any():
                        tdf = tdf.loc[:, ~tdf.columns.duplicated()]
                    tdf = tdf.dropna(subset=["Close"])
                    if len(tdf) < 2:
                        continue

                    price = float(tdf["Close"].iloc[-1])
                    if price > max_price or price <= 0:
                        continue

                    prev_close = float(tdf["Close"].iloc[-2]) if len(tdf) >= 2 else price
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                    volume = float(tdf["Volume"].iloc[-1])
                    avg_vol = float(tdf["Volume"].mean()) if len(tdf) >= 3 else volume
                    rvol = volume / avg_vol if avg_vol > 0 else 0

                    if volume < min_volume:
                        continue

                    results[t] = {
                        "price": round(price, 4),
                        "change_pct": round(change_pct, 2),
                        "volume": volume,
                        "avg_volume": avg_vol,
                        "rvol": round(rvol, 2),
                        "is_sub_dollar": price < 1.0,
                        "is_penny": price < 5.0,
                    }
                except Exception:
                    continue
    except Exception:
        pass

    return results


def get_live_penny_tickers(max_price: float = 5.0) -> list[str]:
    """
    Returns a list of tickers from the seed universe that are
    currently trading under max_price with sufficient volume.
    This is what gets fed into the main data pipeline.
    """
    scanned = scan_penny_stocks(max_price=max_price)
    # Sort by relative volume (most active first)
    sorted_tickers = sorted(scanned.keys(), key=lambda t: scanned[t]["rvol"], reverse=True)
    return sorted_tickers


def get_scan_results() -> dict:
    """Get the cached scan results for display."""
    return scan_penny_stocks()
