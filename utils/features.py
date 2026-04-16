"""
Feature engineering for professional-grade stock signal prediction.
Includes standard TA indicators PLUS penny stock breakout features:
  - Relative volume spikes (RVol)
  - Float turnover rate
  - Price compression (consolidation detection)
  - Gap analysis
  - Multi-timeframe momentum alignment
"""

import numpy as np
import pandas as pd
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    opn = df["Open"]

    # ── Trend ─────────────────────────────────────────────────────
    df["sma_10"] = ta.trend.sma_indicator(close, window=10)
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["ema_9"] = ta.trend.ema_indicator(close, window=9)
    df["ema_21"] = ta.trend.ema_indicator(close, window=21)

    df["close_to_sma10"] = close / df["sma_10"] - 1
    df["close_to_sma20"] = close / df["sma_20"] - 1
    df["close_to_sma50"] = close / df["sma_50"] - 1
    df["sma10_to_sma20"] = df["sma_10"] / df["sma_20"] - 1
    df["ema9_to_ema21"] = df["ema_9"] / df["ema_21"] - 1

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # ── Momentum ──────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["rsi_7"] = ta.momentum.rsi(close, window=7)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)
    df["roc_10"] = ta.momentum.roc(close, window=10)
    df["roc_5"] = ta.momentum.roc(close, window=5)
    df["roc_1"] = ta.momentum.roc(close, window=1)

    # ── Volatility ────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / close
    df["bb_pct"] = bb.bollinger_pband()

    atr_ind = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr_ind.average_true_range()
    df["atr_pct"] = df["atr"] / close

    # Keltner Channel squeeze (pro trader staple)
    kc_mid = df["ema_21"]
    kc_upper = kc_mid + 1.5 * df["atr"]
    kc_lower = kc_mid - 1.5 * df["atr"]
    df["squeeze_on"] = ((df["bb_lower"] > kc_lower) & (df["bb_upper"] < kc_upper)).astype(int)

    # ── Volume (critical for penny stocks) ────────────────────────
    df["vol_sma5"] = volume.rolling(5).mean()
    df["vol_sma20"] = volume.rolling(20).mean()
    df["vol_sma50"] = volume.rolling(50).mean()
    df["rvol_5"] = volume / df["vol_sma5"].replace(0, np.nan)
    df["rvol_20"] = volume / df["vol_sma20"].replace(0, np.nan)
    df["rvol_50"] = volume / df["vol_sma50"].replace(0, np.nan)

    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["obv_slope"] = df["obv"].diff(5) / df["obv"].shift(5).replace(0, np.nan)

    mfi = ta.volume.MFIIndicator(high, low, close, volume, window=14)
    df["mfi"] = mfi.money_flow_index()

    # Volume-price trend divergence
    df["vpt"] = (volume * ((close - close.shift(1)) / close.shift(1).replace(0, np.nan))).cumsum()
    df["vpt_slope"] = df["vpt"].diff(5)

    # Dollar volume (liquidity proxy)
    df["dollar_vol"] = close * volume
    df["dollar_vol_sma20"] = df["dollar_vol"].rolling(20).mean()

    # ── Price Action ──────────────────────────────────────────────
    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_10d"] = close.pct_change(10)
    df["vol_10d"] = df["returns_1d"].rolling(10).std()
    df["vol_20d"] = df["returns_1d"].rolling(20).std()

    body = (close - opn).abs()
    full_range = (high - low).replace(0, np.nan)
    df["body_ratio"] = body / full_range
    df["upper_wick"] = (high - pd.concat([close, opn], axis=1).max(axis=1)) / full_range

    # Gap detection
    df["gap_pct"] = (opn - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100

    # ── Penny Stock / Breakout Specifics ──────────────────────────
    # Price compression: 10-day range as % of price
    df["range_10d"] = (high.rolling(10).max() - low.rolling(10).min()) / close
    df["range_20d"] = (high.rolling(20).max() - low.rolling(20).min()) / close

    # Proximity to 52-week high/low (using available data)
    roll_high = high.rolling(min(len(df) - 1, 252), min_periods=20).max()
    roll_low = low.rolling(min(len(df) - 1, 252), min_periods=20).min()
    df["pct_from_high"] = (close - roll_high) / roll_high
    df["pct_from_low"] = (close - roll_low) / roll_low.replace(0, np.nan)

    # Consecutive up/down days
    up_days = (close > close.shift(1)).astype(int)
    df["consec_up"] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
    down_days = (close < close.shift(1)).astype(int)
    df["consec_down"] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

    # Accumulation/Distribution
    clv = ((close - low) - (high - close)) / full_range
    df["ad_line"] = (clv * volume).cumsum()
    df["ad_slope"] = df["ad_line"].diff(5)

    # ── Target ────────────────────────────────────────────────────
    df["target"] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "close_to_sma10", "close_to_sma20", "close_to_sma50",
    "sma10_to_sma20", "ema9_to_ema21",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d", "williams_r",
    "roc_10", "roc_5", "roc_1",
    "bb_width", "bb_pct", "squeeze_on",
    "atr_pct",
    "rvol_5", "rvol_20", "rvol_50",
    "obv_slope", "mfi", "vpt_slope",
    "returns_1d", "returns_5d", "returns_10d",
    "vol_10d", "vol_20d",
    "body_ratio", "upper_wick", "gap_pct",
    "range_10d", "range_20d",
    "pct_from_high", "pct_from_low",
    "consec_up", "consec_down",
    "ad_slope",
]
