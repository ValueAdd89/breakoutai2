"""
Intraday 5-minute bar data and VWAP computation.
Uses yfinance (free, no key) for 5m bars over up to 60 days.
Computes VWAP, cumulative volume, and session statistics.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class IntradayStats:
    ticker: str
    last_price: float
    day_open: float
    day_high: float
    day_low: float
    day_change_pct: float
    vwap: float
    above_vwap: bool
    cumulative_volume: float
    avg_daily_volume: float
    volume_vs_avg: float  # cumulative vs average
    morning_range_pct: float  # first 30-min high-low as % of open
    breakout_of_day: bool  # hit new intraday high in last 3 bars
    opening_range_break: bool  # broke above first 15-min high


@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday_5m(ticker: str, days: int = 5) -> pd.DataFrame | None:
    """Fetch 5-minute OHLCV bars for recent trading days."""
    try:
        period = f"{min(days, 60)}d"
        df = yf.download(
            ticker, period=period, interval="5m",
            progress=False, auto_adjust=True, timeout=10,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Deduplicate any repeated column names to prevent 2-D column access
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        df.dropna(subset=["Close"], inplace=True)
        return df if len(df) > 10 else None
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday_batch(tickers: tuple, days: int = 5) -> dict[str, pd.DataFrame]:
    """Fetch 5-min bars for multiple tickers in parallel."""
    results = {}

    def _fetch_one(t):
        return t, fetch_intraday_5m(t, days=days)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_fetch_one, t) for t in tickers]
        for f in as_completed(futures, timeout=30):
            try:
                t, df = f.result(timeout=10)
                if df is not None:
                    results[t] = df
            except Exception:
                continue
    return results


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price — the single most important intraday level
    for professional traders. Price above VWAP = buyers in control.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def compute_intraday_stats(ticker: str, df_5m: pd.DataFrame, avg_daily_vol: float = 0.0) -> IntradayStats | None:
    """Compute intraday session statistics from 5-minute bars."""
    if df_5m is None or len(df_5m) < 5:
        return None

    try:
        # Identify today's session (last trading day in the data)
        df = df_5m.copy()
        df.index = pd.to_datetime(df.index)
        df["date"] = df.index.date
        latest_date = df["date"].max()
        today_df = df[df["date"] == latest_date].copy()

        if len(today_df) < 2:
            return None

        # VWAP on today's session only
        today_df["vwap"] = compute_vwap(today_df)

        last = today_df.iloc[-1]
        first = today_df.iloc[0]

        last_price = float(last["Close"])
        day_open = float(first["Open"])
        day_high = float(today_df["High"].max())
        day_low = float(today_df["Low"].min())
        day_change = ((last_price - day_open) / day_open * 100) if day_open > 0 else 0

        vwap = float(last["vwap"])
        cum_vol = float(today_df["Volume"].sum())

        # Morning range (first 6 bars = 30 min)
        morning = today_df.head(6)
        m_range = float((morning["High"].max() - morning["Low"].min()) / day_open * 100) if day_open > 0 else 0

        # Opening range breakout (first 3 bars = 15 min)
        or_3 = today_df.head(3)
        or_high = float(or_3["High"].max())
        or_break = last_price > or_high if len(today_df) > 3 else False

        # Breakout-of-day (new high in last 3 bars)
        recent_high = float(today_df.tail(3)["High"].max())
        prior_high = float(today_df.iloc[:-3]["High"].max()) if len(today_df) > 3 else recent_high
        bod = recent_high >= prior_high and len(today_df) > 3

        vol_vs_avg = (cum_vol / avg_daily_vol) if avg_daily_vol > 0 else 0

        return IntradayStats(
            ticker=ticker,
            last_price=round(last_price, 2),
            day_open=round(day_open, 2),
            day_high=round(day_high, 2),
            day_low=round(day_low, 2),
            day_change_pct=round(day_change, 2),
            vwap=round(vwap, 2),
            above_vwap=last_price > vwap,
            cumulative_volume=cum_vol,
            avg_daily_volume=avg_daily_vol,
            volume_vs_avg=round(vol_vs_avg, 2),
            morning_range_pct=round(m_range, 2),
            breakout_of_day=bod,
            opening_range_break=or_break,
        )
    except Exception:
        return None


def is_market_hours() -> bool:
    """Check if US market is currently open (approximate — doesn't handle holidays)."""
    try:
        now = pd.Timestamp.now(tz="America/New_York")
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception:
        return False
