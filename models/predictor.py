"""
Prediction engine — ensemble ML + penny stock breakout scanner.
Parallel model training, lightweight trees, breakout scoring.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.features import compute_features, FEATURE_COLS


@dataclass
class Signal:
    ticker: str
    direction: str
    confidence: float
    probability: float
    momentum: float
    volume_ratio: float
    volatility: float
    rsi: float
    macd_hist: float
    signals: list
    accuracy: float
    price: float
    change_1d: float
    change_5d: float
    # Penny stock breakout fields
    breakout_score: float = 0.0       # 0-100 composite score
    breakout_grade: str = ""          # A+ through F
    rvol_5: float = 0.0
    rvol_20: float = 0.0
    squeeze_on: bool = False
    range_compression: float = 0.0
    gap_pct: float = 0.0
    pct_from_high: float = 0.0
    consec_up: int = 0
    ad_slope: float = 0.0
    breakout_factors: list = field(default_factory=list)


def predict_signal(ticker: str, df: pd.DataFrame) -> Signal | None:
    try:
        feat_df = compute_features(df)
        if len(feat_df) < 50:
            return None

        X = feat_df[FEATURE_COLS].values
        y = feat_df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]

        gbt = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.12,
            subsample=0.8, random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42, n_jobs=-1,
        )

        gbt.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        prob_gbt = gbt.predict_proba(X_scaled[-1:])[:, 1][0]
        prob_rf = rf.predict_proba(X_scaled[-1:])[:, 1][0]
        probability = 0.6 * prob_gbt + 0.4 * prob_rf

        gbt_acc = accuracy_score(y_test, gbt.predict(X_test))
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        accuracy = 0.6 * gbt_acc + 0.4 * rf_acc

        if probability > 0.58:
            direction = "bullish"
            confidence = min(95, 50 + (probability - 0.5) * 100)
        elif probability < 0.42:
            direction = "bearish"
            confidence = min(95, 50 + (0.5 - probability) * 100)
        else:
            direction = "neutral"
            confidence = max(30, 50 - abs(probability - 0.5) * 100)

        latest = feat_df.iloc[-1]
        price = df["Close"].iloc[-1]
        if isinstance(price, pd.Series):
            price = price.iloc[0]
        price = float(price)

        signals = _detect_signals(latest, probability, direction)
        breakout_score, breakout_grade, breakout_factors = _score_breakout(latest, probability, direction)

        return Signal(
            ticker=ticker, direction=direction,
            confidence=round(confidence, 1),
            probability=round(probability, 4),
            momentum=round(float(latest["roc_5"]), 2),
            volume_ratio=round(float(latest["rvol_20"]), 2),
            volatility=round(float(latest["atr_pct"] * 100), 2),
            rsi=round(float(latest["rsi_14"]), 1),
            macd_hist=round(float(latest["macd_hist"]), 4),
            signals=signals,
            accuracy=round(accuracy * 100, 1),
            price=round(price, 2),
            change_1d=round(float(latest["returns_1d"] * 100), 2),
            change_5d=round(float(latest["returns_5d"] * 100), 2),
            breakout_score=round(breakout_score, 1),
            breakout_grade=breakout_grade,
            rvol_5=round(float(latest["rvol_5"]), 2),
            rvol_20=round(float(latest["rvol_20"]), 2),
            squeeze_on=bool(latest["squeeze_on"]),
            range_compression=round(float(latest["range_10d"] * 100), 1),
            gap_pct=round(float(latest["gap_pct"]), 2),
            pct_from_high=round(float(latest["pct_from_high"] * 100), 1),
            consec_up=int(latest["consec_up"]),
            ad_slope=round(float(latest["ad_slope"]), 2),
            breakout_factors=breakout_factors,
        )
    except Exception:
        return None


def predict_batch_parallel(data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
    results = {}
    def _predict_one(item):
        ticker, df = item
        return ticker, predict_signal(ticker, df)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_predict_one, item): item[0] for item in data.items()}
        for future in as_completed(futures):
            try:
                ticker, sig = future.result(timeout=30)
                if sig is not None:
                    results[ticker] = sig
            except Exception:
                continue
    return results


def _score_breakout(row: pd.Series, prob: float, direction: str) -> tuple[float, str, list]:
    """
    Composite breakout score (0-100) using factors a professional trader checks:
    1. Volume confirmation (RVol) — 25 pts
    2. Price compression / squeeze — 20 pts
    3. Momentum alignment — 15 pts
    4. Accumulation evidence — 15 pts
    5. Proximity to key levels — 10 pts
    6. Model conviction — 15 pts
    """
    score = 0.0
    factors = []

    # 1. VOLUME (25 pts) — the single most important breakout factor
    rvol5 = float(row["rvol_5"])
    rvol20 = float(row["rvol_20"])
    if rvol5 >= 3.0:
        score += 25
        factors.append(("Volume Explosion", f"RVol(5) at {rvol5:.1f}x — extreme institutional interest", "critical"))
    elif rvol5 >= 2.0:
        score += 20
        factors.append(("Volume Surge", f"RVol(5) at {rvol5:.1f}x — strong above-average activity", "strong"))
    elif rvol5 >= 1.5:
        score += 12
        factors.append(("Elevated Volume", f"RVol(5) at {rvol5:.1f}x — moderate uptick", "moderate"))
    elif rvol20 >= 1.3:
        score += 8
        factors.append(("Warming Volume", f"RVol(20) at {rvol20:.1f}x — slow build", "weak"))
    else:
        factors.append(("Low Volume", f"RVol at {rvol5:.1f}x / {rvol20:.1f}x — no conviction", "negative"))

    # 2. COMPRESSION / SQUEEZE (20 pts) — tighter range = bigger breakout
    squeeze = bool(row["squeeze_on"])
    range10 = float(row["range_10d"])
    bb_width = float(row["bb_width"])
    if squeeze and bb_width < 0.04:
        score += 20
        factors.append(("Squeeze Firing", f"BB inside Keltner + width {bb_width:.3f} — imminent expansion", "critical"))
    elif squeeze:
        score += 15
        factors.append(("Squeeze Active", "Bollinger Bands inside Keltner Channel", "strong"))
    elif range10 < 0.06:
        score += 10
        factors.append(("Tight Range", f"10-day range only {range10*100:.1f}% of price — compression building", "moderate"))
    elif bb_width < 0.05:
        score += 7
        factors.append(("Narrowing Bands", f"BB width at {bb_width:.3f} — volatility contracting", "moderate"))
    else:
        factors.append(("Wide Range", f"No compression detected (range: {range10*100:.1f}%)", "neutral"))

    # 3. MOMENTUM (15 pts)
    rsi = float(row["rsi_14"])
    roc5 = float(row["roc_5"])
    macd_h = float(row["macd_hist"])
    ema_cross = float(row["ema9_to_ema21"])
    momentum_pts = 0
    if 50 < rsi < 70 and roc5 > 0 and macd_h > 0:
        momentum_pts = 15
        factors.append(("Momentum Aligned", f"RSI {rsi:.0f} + ROC +{roc5:.1f}% + MACD bullish — full alignment", "strong"))
    elif roc5 > 0 and macd_h > 0:
        momentum_pts = 10
        factors.append(("Momentum Building", f"ROC +{roc5:.1f}% with MACD bullish", "moderate"))
    elif ema_cross > 0:
        momentum_pts = 5
        factors.append(("EMA Bullish Cross", f"9 EMA above 21 EMA ({ema_cross*100:.2f}%)", "weak"))
    else:
        factors.append(("No Momentum", f"RSI {rsi:.0f}, ROC {roc5:.1f}% — directionless", "negative"))
    score += momentum_pts

    # 4. ACCUMULATION (15 pts) — smart money footprint
    ad = float(row["ad_slope"])
    mfi = float(row["mfi"])
    obv = float(row["obv_slope"])
    if ad > 0 and mfi > 50 and obv > 0:
        score += 15
        factors.append(("Strong Accumulation", f"A/D rising, MFI {mfi:.0f}, OBV positive — institutional buying", "strong"))
    elif (ad > 0 and mfi > 40) or (mfi > 60 and obv > 0):
        score += 10
        factors.append(("Moderate Accumulation", f"MFI {mfi:.0f}, A/D {'rising' if ad > 0 else 'flat'}", "moderate"))
    elif mfi > 50:
        score += 5
        factors.append(("Mild Inflow", f"MFI at {mfi:.0f} — slight buying pressure", "weak"))
    else:
        factors.append(("Distribution", f"MFI at {mfi:.0f} — selling pressure evident", "negative"))

    # 5. KEY LEVELS (10 pts) — proximity to breakout level
    pct_high = float(row["pct_from_high"]) * 100
    gap = float(row["gap_pct"])
    if -3 < pct_high < 0:
        score += 10
        factors.append(("Near Breakout", f"Only {abs(pct_high):.1f}% from 52w high — potential breakout zone", "strong"))
    elif pct_high > 0:
        score += 8
        factors.append(("New High Territory", f"Trading {pct_high:.1f}% above prior high", "strong"))
    elif -10 < pct_high < -3:
        score += 5
        factors.append(("Approaching Highs", f"{abs(pct_high):.1f}% from high — building toward resistance", "moderate"))
    else:
        factors.append(("Far From Highs", f"{abs(pct_high):.1f}% below high — deep pullback", "negative"))

    if abs(gap) > 2:
        factors.append(("Gap Alert", f"{'Gap up' if gap > 0 else 'Gap down'} {abs(gap):.1f}% — momentum catalyst", "signal"))

    # 6. MODEL (15 pts)
    if prob > 0.68:
        score += 15
        factors.append(("Strong Model Signal", f"Ensemble probability {prob:.1%} — high conviction", "strong"))
    elif prob > 0.58:
        score += 10
        factors.append(("Positive Model", f"Ensemble probability {prob:.1%}", "moderate"))
    elif prob > 0.50:
        score += 5
        factors.append(("Lean Bullish", f"Ensemble probability {prob:.1%} — marginal edge", "weak"))
    else:
        factors.append(("Bearish Model", f"Ensemble probability {prob:.1%} — counter-trend", "negative"))

    # Grade
    if score >= 85:
        grade = "A+"
    elif score >= 75:
        grade = "A"
    elif score >= 65:
        grade = "B+"
    elif score >= 55:
        grade = "B"
    elif score >= 45:
        grade = "C+"
    elif score >= 35:
        grade = "C"
    elif score >= 25:
        grade = "D"
    else:
        grade = "F"

    return score, grade, factors


def _detect_signals(row: pd.Series, prob: float, direction: str) -> list[str]:
    signals = []
    if row["rsi_14"] > 70:
        signals.append("RSI overbought (>70) — reversal risk")
    elif row["rsi_14"] < 30:
        signals.append("RSI oversold (<30) — bounce candidate")
    if row["rvol_5"] > 2.0:
        signals.append(f"Volume surge: {row['rvol_5']:.1f}x 5d avg")
    elif row["rvol_20"] > 1.5:
        signals.append(f"Elevated volume: {row['rvol_20']:.1f}x 20d avg")
    if row["squeeze_on"]:
        signals.append("Bollinger-Keltner squeeze active — breakout imminent")
    if row["macd_hist"] > 0 and row["macd"] > row["macd_signal"]:
        signals.append("MACD bullish crossover active")
    elif row["macd_hist"] < 0 and row["macd"] < row["macd_signal"]:
        signals.append("MACD bearish crossover active")
    if row["bb_pct"] > 1.0:
        signals.append("Price above upper Bollinger Band")
    elif row["bb_pct"] < 0.0:
        signals.append("Price below lower Bollinger Band")
    if row["adx"] > 30:
        d = "bullish" if row["adx_pos"] > row["adx_neg"] else "bearish"
        signals.append(f"Strong {d} trend (ADX: {row['adx']:.0f})")
    if row["close_to_sma10"] > 0 and row["close_to_sma20"] > 0 and row["close_to_sma50"] > 0:
        signals.append("Above all major SMAs — bullish alignment")
    elif row["close_to_sma10"] < 0 and row["close_to_sma20"] < 0 and row["close_to_sma50"] < 0:
        signals.append("Below all major SMAs — bearish alignment")
    if abs(row["gap_pct"]) > 3:
        signals.append(f"{'Gap up' if row['gap_pct'] > 0 else 'Gap down'} {abs(row['gap_pct']):.1f}%")
    if prob > 0.72:
        signals.append("High ensemble conviction — strong bullish")
    elif prob < 0.28:
        signals.append("High ensemble conviction — strong bearish")
    if not signals:
        signals.append("No strong signals — monitoring")
    return signals
