"""
Feature engineering pipeline for stock signal prediction.
Computes 30+ technical indicators from OHLCV data using the `ta` library
and custom calculations. All features are designed for real-time computation
with minimal lookback requirements.
"""

import numpy as np
import pandas as pd
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume

    Returns
    -------
    pd.DataFrame
        Original data plus ~35 feature columns, NaN rows dropped.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Trend Indicators ──────────────────────────────────────────
    df["sma_10"] = ta.trend.sma_indicator(close, window=10)
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)

    # Price relative to moving averages
    df["close_to_sma10"] = close / df["sma_10"] - 1
    df["close_to_sma20"] = close / df["sma_20"] - 1
    df["close_to_sma50"] = close / df["sma_50"] - 1
    df["sma10_to_sma20"] = df["sma_10"] / df["sma_20"] - 1

    # MACD
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ADX
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # ── Momentum Indicators ───────────────────────────────────────
    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["rsi_7"] = ta.momentum.rsi(close, window=7)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)
    df["roc_10"] = ta.momentum.roc(close, window=10)
    df["roc_5"] = ta.momentum.roc(close, window=5)

    # ── Volatility Indicators ─────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / close
    df["bb_pct"] = bb.bollinger_pband()

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / close

    # ── Volume Indicators ─────────────────────────────────────────
    df["volume_sma20"] = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / df["volume_sma20"]
    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["obv_slope"] = df["obv"].diff(5) / df["obv"].shift(5)

    mfi = ta.volume.MFIIndicator(high, low, close, volume, window=14)
    df["mfi"] = mfi.money_flow_index()

    # ── Price Action Features ─────────────────────────────────────
    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_10d"] = close.pct_change(10)
    df["volatility_10d"] = df["returns_1d"].rolling(10).std()
    df["volatility_20d"] = df["returns_1d"].rolling(20).std()

    # Candle body & wick ratios
    body = (close - df["Open"]).abs()
    full_range = high - low
    df["body_ratio"] = body / full_range.replace(0, np.nan)
    df["upper_wick"] = (high - pd.concat([close, df["Open"]], axis=1).max(axis=1)) / full_range.replace(0, np.nan)

    # ── Target: next-day direction (1 = up, 0 = down) ────────────
    df["target"] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)
    return df


# Feature columns used by the model
FEATURE_COLS = [
    "close_to_sma10", "close_to_sma20", "close_to_sma50", "sma10_to_sma20",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d", "williams_r",
    "roc_10", "roc_5",
    "bb_width", "bb_pct",
    "atr_pct",
    "volume_ratio", "obv_slope", "mfi",
    "returns_1d", "returns_5d", "returns_10d",
    "volatility_10d", "volatility_20d",
    "body_ratio", "upper_wick",
]
