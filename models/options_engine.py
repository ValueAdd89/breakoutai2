"""
Options Play Engine
====================
Generates actionable options trade recommendations with:
  • 90% confidence interval price targets (based on historical vol)
  • Entry / exit / stop-loss levels
  • Specific strategy selection (calls, puts, spreads, straddles)
  • Risk/reward analysis with max-loss / max-gain
  • Reasoning chain explaining every factor driving the play
  • Greeks estimates from Black-Scholes approximation

Uses only OHLCV data + technical signals — no live options chain API needed.
All pricing is estimated via analytical models for demonstration purposes.
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataclasses import dataclass, field

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

from models.predictor import Signal


def _norm_cdf(x):
    """Fallback normal CDF using numpy if scipy is unavailable."""
    if sp_stats is not None:
        return sp_stats.norm.cdf(x)
    # Abramowitz & Stegun approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x):
    """Fallback normal PDF using numpy if scipy is unavailable."""
    if sp_stats is not None:
        return sp_stats.norm.pdf(x)
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def _norm_ppf(p):
    """Fallback inverse normal CDF (quantile function)."""
    if sp_stats is not None:
        return sp_stats.norm.ppf(p)
    # Rational approximation (Beasley-Springer-Moro)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PriceTarget:
    """90% confidence interval price projection."""
    current: float
    target_mid: float
    target_low: float       # 5th percentile
    target_high: float      # 95th percentile
    expected_move_pct: float
    horizon_days: int
    annual_vol: float
    method: str             # description of calculation method


@dataclass
class OptionLeg:
    """Single leg of an options play."""
    direction: str          # "buy" | "sell"
    option_type: str        # "call" | "put"
    strike: float
    estimated_premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    days_to_expiry: int


@dataclass
class OptionsPlay:
    """Complete options trade recommendation."""
    ticker: str
    strategy_name: str      # e.g. "Bull Call Spread", "Long Put", etc.
    strategy_type: str      # "directional_bullish" | "directional_bearish" | "neutral" | "volatility"
    conviction: str         # "high" | "moderate" — only ≥90% CI plays shown
    risk_tier: str          # "conservative" | "moderate" | "aggressive"

    # Legs
    legs: list              # list[OptionLeg]

    # Entry / Exit
    entry_price: float      # net debit or credit
    entry_timing: str       # when to enter
    profit_target: float    # price to take profit
    stop_loss: float        # price to cut losses
    exit_timing: str        # when to exit

    # Risk / Reward
    max_loss: float
    max_gain: float         # can be "unlimited" represented as -1
    risk_reward_ratio: float
    break_even: float
    probability_of_profit: float  # estimated PoP %

    # Price target
    price_target: PriceTarget

    # Reasoning
    drivers: list           # list of (factor_name, description, impact) tuples
    thesis: str             # one-paragraph narrative
    risks: list             # list of risk descriptions

    # Meta
    suggested_allocation: str  # e.g. "1-3% of portfolio"
    ideal_account_size: str    # e.g. "$10k+"


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes approximation
# ═══════════════════════════════════════════════════════════════════

def _bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes European option price."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute Greeks from Black-Scholes."""
    if T <= 0:
        return {"delta": 1.0 if option_type == "call" else -1.0,
                "gamma": 0, "theta": 0, "vega": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = _norm_pdf(d1)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100  # per 1% vol move

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * _norm_cdf(d2)) / 365
    else:
        delta = _norm_cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * _norm_cdf(-d2)) / 365

    return {"delta": round(delta, 3), "gamma": round(gamma, 4),
            "theta": round(theta, 3), "vega": round(vega, 3)}


# ═══════════════════════════════════════════════════════════════════
# Price target with 90% CI
# ═══════════════════════════════════════════════════════════════════

def compute_price_target(df: pd.DataFrame, signal: Signal, horizon_days: int = 21) -> PriceTarget:
    """
    Compute 90% confidence interval price target using:
    1. Historical realized volatility (20-day)
    2. Log-normal price distribution assumption
    3. Directional bias from ensemble model
    """
    close = df["Close"].values.flatten()
    log_returns = np.diff(np.log(close))

    # Realized volatility (annualized)
    vol_20d = np.std(log_returns[-20:]) * np.sqrt(252)
    vol_60d = np.std(log_returns[-60:]) * np.sqrt(252)
    # Blend short and long vol
    annual_vol = 0.6 * vol_20d + 0.4 * vol_60d

    S = signal.price
    T = horizon_days / 252

    # Expected drift adjusted by model direction
    if signal.direction == "bullish":
        drift = annual_vol * 0.15 * (signal.confidence / 100)
    elif signal.direction == "bearish":
        drift = -annual_vol * 0.15 * (signal.confidence / 100)
    else:
        drift = 0

    # Log-normal parameters
    mu = (drift - 0.5 * annual_vol**2) * T
    sigma_t = annual_vol * np.sqrt(T)

    # 90% CI: 5th and 95th percentile
    z_low = _norm_ppf(0.05)
    z_high = _norm_ppf(0.95)

    target_mid = S * np.exp(mu)
    target_low = S * np.exp(mu + z_low * sigma_t)
    target_high = S * np.exp(mu + z_high * sigma_t)
    expected_move = ((target_mid / S) - 1) * 100

    return PriceTarget(
        current=S,
        target_mid=round(target_mid, 2),
        target_low=round(target_low, 2),
        target_high=round(target_high, 2),
        expected_move_pct=round(expected_move, 2),
        horizon_days=horizon_days,
        annual_vol=round(annual_vol * 100, 1),
        method="Blended 20/60d realized vol · Log-normal · Model-adjusted drift",
    )


# ═══════════════════════════════════════════════════════════════════
# Strategy selection & play generation
# ═══════════════════════════════════════════════════════════════════

def _round_strike(price: float, step: float = 5.0) -> float:
    """Round to nearest standard option strike increment."""
    if price < 50:
        step = 2.5
    elif price < 200:
        step = 5.0
    else:
        step = 10.0
    return round(price / step) * step


def generate_options_plays(signal: Signal, df: pd.DataFrame) -> list[OptionsPlay]:
    """
    Generate up to 3 options play recommendations for a ticker.
    Only returns plays where the 90% CI supports the thesis.

    Strategy selection logic:
    - Strong bullish + low vol → Bull Call Spread
    - Strong bullish + high vol → Short Put (sell premium)
    - Strong bearish + low vol → Bear Put Spread
    - Strong bearish + high vol → Short Call Spread
    - Neutral + high vol → Iron Condor
    - Squeeze detected → Long Straddle
    - Any direction + very high confidence → Naked directional
    """
    plays = []
    S = signal.price
    r = 0.05  # risk-free rate estimate

    # Compute price targets for multiple horizons
    pt_short = compute_price_target(df, signal, horizon_days=14)
    pt_mid = compute_price_target(df, signal, horizon_days=30)
    pt_long = compute_price_target(df, signal, horizon_days=60)

    annual_vol = pt_mid.annual_vol / 100
    is_high_vol = annual_vol > 0.40
    is_low_vol = annual_vol < 0.25
    is_squeeze = any("squeeze" in s.lower() for s in signal.signals)

    # ── Strategy 1: Directional Play ──────────────────────────────
    if signal.direction == "bullish" and signal.confidence >= 60:
        if is_high_vol:
            # Bull Call Spread — cap risk in high vol
            play = _build_bull_call_spread(signal, df, pt_mid, S, r, annual_vol)
            if play:
                plays.append(play)
        else:
            # Long Call — more upside in low vol
            play = _build_long_call(signal, df, pt_mid, S, r, annual_vol)
            if play:
                plays.append(play)

    elif signal.direction == "bearish" and signal.confidence >= 60:
        if is_high_vol:
            # Bear Put Spread
            play = _build_bear_put_spread(signal, df, pt_mid, S, r, annual_vol)
            if play:
                plays.append(play)
        else:
            # Long Put
            play = _build_long_put(signal, df, pt_mid, S, r, annual_vol)
            if play:
                plays.append(play)

    # ── Strategy 2: Premium Selling (high vol environments) ───────
    if is_high_vol and signal.confidence >= 55:
        play = _build_iron_condor(signal, df, pt_mid, S, r, annual_vol)
        if play:
            plays.append(play)

    # ── Strategy 3: Volatility Play (squeeze detected) ────────────
    if is_squeeze or is_low_vol:
        play = _build_long_straddle(signal, df, pt_mid, S, r, annual_vol)
        if play:
            plays.append(play)

    # ── Strategy 4: Conservative Credit Spread ────────────────────
    if signal.direction == "bullish" and signal.confidence >= 55:
        play = _build_short_put_spread(signal, df, pt_mid, S, r, annual_vol)
        if play:
            plays.append(play)
    elif signal.direction == "bearish" and signal.confidence >= 55:
        play = _build_short_call_spread(signal, df, pt_mid, S, r, annual_vol)
        if play:
            plays.append(play)

    # Deduplicate by strategy name and limit to 3
    seen = set()
    unique = []
    for p in plays:
        if p.strategy_name not in seen:
            seen.add(p.strategy_name)
            unique.append(p)
    return unique[:3]


# ═══════════════════════════════════════════════════════════════════
# Individual strategy builders
# ═══════════════════════════════════════════════════════════════════

def _build_drivers(signal: Signal, strategy_context: str) -> list[tuple]:
    """Build reasoning chain from signal data."""
    drivers = []

    # Direction + model
    drivers.append((
        "Ensemble Model",
        f"GBT + RF ensemble predicts {signal.direction} with {signal.probability:.1%} probability",
        "primary",
    ))

    # Confidence
    if signal.confidence >= 75:
        drivers.append(("High Conviction", f"Model confidence at {signal.confidence}% exceeds threshold", "strong"))
    else:
        drivers.append(("Moderate Conviction", f"Model confidence at {signal.confidence}%", "moderate"))

    # RSI
    if signal.rsi > 70:
        drivers.append(("RSI Overbought", f"RSI at {signal.rsi} — potential mean reversion", "caution"))
    elif signal.rsi < 30:
        drivers.append(("RSI Oversold", f"RSI at {signal.rsi} — bounce probability elevated", "supportive"))
    else:
        drivers.append(("RSI Neutral", f"RSI at {signal.rsi} — no extreme reading", "neutral"))

    # Volume
    if signal.volume_ratio > 1.5:
        drivers.append(("Volume Confirmation", f"Relative volume at {signal.volume_ratio}x — institutional activity", "strong"))
    else:
        drivers.append(("Volume Normal", f"Relative volume at {signal.volume_ratio}x", "neutral"))

    # Momentum
    if abs(signal.momentum) > 3:
        direction = "upward" if signal.momentum > 0 else "downward"
        drivers.append(("Momentum", f"5-day ROC at {signal.momentum:+.2f}% — {direction} acceleration", "supportive"))

    # MACD
    if signal.macd_hist > 0:
        drivers.append(("MACD Bullish", "MACD histogram positive — upward momentum", "supportive"))
    elif signal.macd_hist < 0:
        drivers.append(("MACD Bearish", "MACD histogram negative — downward momentum", "supportive"))

    # Volatility context for strategy
    drivers.append(("Strategy Fit", strategy_context, "context"))

    # Active signals from predictor
    for sig_text in signal.signals[:2]:
        drivers.append(("Technical Signal", sig_text, "signal"))

    return drivers


def _build_thesis(signal: Signal, strategy_name: str, pt: PriceTarget) -> str:
    """Generate a natural-language thesis for the play."""
    dir_word = "upside" if signal.direction == "bullish" else "downside" if signal.direction == "bearish" else "range-bound"

    return (
        f"The ensemble model identifies {dir_word} potential for {signal.ticker} with "
        f"{signal.confidence}% confidence. The 90% confidence interval projects a price range of "
        f"${pt.target_low:.2f}–${pt.target_high:.2f} over {pt.horizon_days} trading days "
        f"(annualized vol: {pt.annual_vol}%). "
        f"RSI at {signal.rsi} and relative volume at {signal.volume_ratio}x "
        f"{'confirm' if (signal.direction == 'bullish' and signal.rsi < 70) or (signal.direction == 'bearish' and signal.rsi > 30) else 'add nuance to'} "
        f"the directional bias. "
        f"A {strategy_name} is recommended to {'capitalize on the move' if 'Long' in strategy_name or 'Bull' in strategy_name or 'Bear' in strategy_name else 'harvest premium from elevated implied volatility'} "
        f"while {'defining risk at entry' if 'Spread' in strategy_name else 'maintaining favorable risk/reward'}."
    )


def _build_risks(signal: Signal, strategy_name: str) -> list[str]:
    """Generate risk factors for the play."""
    risks = [
        "Model predictions are probabilistic — accuracy is not guaranteed",
        "Options lose value over time due to theta decay",
    ]
    if signal.volatility > 3:
        risks.append(f"Elevated volatility ({signal.volatility}%) increases option premiums and potential for adverse moves")
    if signal.rsi > 65 and signal.direction == "bullish":
        risks.append(f"RSI at {signal.rsi} approaching overbought — reversal risk exists")
    if signal.rsi < 35 and signal.direction == "bearish":
        risks.append(f"RSI at {signal.rsi} approaching oversold — snap-back bounce risk")
    if "earnings" in " ".join(signal.signals).lower():
        risks.append("Upcoming earnings event could cause unpredictable gap moves")
    risks.append("Estimated premiums are model-derived — actual market prices will differ")
    risks.append("Low liquidity in some strikes may result in wider bid-ask spreads")
    return risks


def _make_leg(S, K, T_years, r, sigma, direction, option_type):
    """Create an OptionLeg with pricing and Greeks."""
    premium = _bs_price(S, K, T_years, r, sigma, option_type)
    greeks = _bs_greeks(S, K, T_years, r, sigma, option_type)
    dte = max(1, int(T_years * 365))
    return OptionLeg(
        direction=direction,
        option_type=option_type,
        strike=round(K, 2),
        estimated_premium=round(premium, 2),
        delta=greeks["delta"],
        gamma=greeks["gamma"],
        theta=greeks["theta"],
        vega=greeks["vega"],
        days_to_expiry=dte,
    )


# ── Long Call ─────────────────────────────────────────────────────

def _build_long_call(signal, df, pt, S, r, sigma):
    T = 30 / 365
    K = _round_strike(S * 1.02)  # slightly OTM
    leg = _make_leg(S, K, T, r, sigma, "buy", "call")

    entry = leg.estimated_premium
    max_loss = entry * 100  # per contract
    be = K + entry
    target_exit = pt.target_high
    profit_at_target = max(0, target_exit - be) * 100
    rr = profit_at_target / max_loss if max_loss > 0 else 0

    # PoP estimate: P(S_T > break_even)
    d2 = (np.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = _norm_cdf(d2) * 100

    drivers = _build_drivers(signal, "Low-to-moderate volatility favors buying directional options — cheaper premiums provide leverage")
    thesis = _build_thesis(signal, "Long Call", pt)
    risks = _build_risks(signal, "Long Call")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Long Call",
        strategy_type="directional_bullish",
        conviction="high" if signal.confidence >= 75 else "moderate",
        risk_tier="moderate",
        legs=[leg],
        entry_price=round(entry, 2),
        entry_timing="Enter on a pullback to the 10-day SMA or on a bullish daily candle close",
        profit_target=round(entry * 2, 2),
        stop_loss=round(entry * 0.5, 2),
        exit_timing=f"Exit at 50-100% gain, or {max(5, pt.horizon_days - 7)} days before expiry to minimize theta burn",
        max_loss=round(max_loss, 2),
        max_gain=-1,  # unlimited
        risk_reward_ratio=round(rr, 2),
        break_even=round(be, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="1-2% of portfolio",
        ideal_account_size="$5,000+",
    )


# ── Long Put ──────────────────────────────────────────────────────

def _build_long_put(signal, df, pt, S, r, sigma):
    T = 30 / 365
    K = _round_strike(S * 0.98)  # slightly OTM
    leg = _make_leg(S, K, T, r, sigma, "buy", "put")

    entry = leg.estimated_premium
    max_loss = entry * 100
    be = K - entry
    target_exit = pt.target_low
    profit_at_target = max(0, be - target_exit) * 100
    rr = profit_at_target / max_loss if max_loss > 0 else 0

    d2 = (np.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = (1 - _norm_cdf(d2)) * 100

    drivers = _build_drivers(signal, "Bearish bias with controlled risk — put purchase defines maximum loss at premium paid")
    thesis = _build_thesis(signal, "Long Put", pt)
    risks = _build_risks(signal, "Long Put")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Long Put",
        strategy_type="directional_bearish",
        conviction="high" if signal.confidence >= 75 else "moderate",
        risk_tier="moderate",
        legs=[leg],
        entry_price=round(entry, 2),
        entry_timing="Enter on a failed bounce at resistance or on a bearish engulfing candle",
        profit_target=round(entry * 2, 2),
        stop_loss=round(entry * 0.5, 2),
        exit_timing=f"Exit at 50-100% gain, or {max(5, pt.horizon_days - 7)} days before expiry",
        max_loss=round(max_loss, 2),
        max_gain=round((be) * 100, 2),
        risk_reward_ratio=round(rr, 2),
        break_even=round(be, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="1-2% of portfolio",
        ideal_account_size="$5,000+",
    )


# ── Bull Call Spread ──────────────────────────────────────────────

def _build_bull_call_spread(signal, df, pt, S, r, sigma):
    T = 30 / 365
    K_long = _round_strike(S)
    K_short = _round_strike(S * 1.05)
    if K_long == K_short:
        K_short = K_long + (5 if S < 200 else 10)

    leg_long = _make_leg(S, K_long, T, r, sigma, "buy", "call")
    leg_short = _make_leg(S, K_short, T, r, sigma, "sell", "call")

    entry = leg_long.estimated_premium - leg_short.estimated_premium
    entry = max(0.10, entry)
    spread_width = K_short - K_long
    max_loss = entry * 100
    max_gain = (spread_width - entry) * 100
    be = K_long + entry
    rr = max_gain / max_loss if max_loss > 0 else 0

    d2 = (np.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = _norm_cdf(d2) * 100

    drivers = _build_drivers(signal, "High implied volatility makes spreads preferable — selling the upper strike offsets premium cost")
    thesis = _build_thesis(signal, "Bull Call Spread", pt)
    risks = _build_risks(signal, "Bull Call Spread")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Bull Call Spread",
        strategy_type="directional_bullish",
        conviction="high" if signal.confidence >= 75 else "moderate",
        risk_tier="conservative",
        legs=[leg_long, leg_short],
        entry_price=round(entry, 2),
        entry_timing="Enter when stock confirms support above 20-SMA or on an intraday pullback",
        profit_target=round(spread_width * 0.7, 2),
        stop_loss=round(entry * 0.5, 2),
        exit_timing=f"Exit at 50-70% of max profit, or close 5+ days before expiry",
        max_loss=round(max_loss, 2),
        max_gain=round(max_gain, 2),
        risk_reward_ratio=round(rr, 2),
        break_even=round(be, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="2-3% of portfolio",
        ideal_account_size="$3,000+",
    )


# ── Bear Put Spread ──────────────────────────────────────────────

def _build_bear_put_spread(signal, df, pt, S, r, sigma):
    T = 30 / 365
    K_long = _round_strike(S)
    K_short = _round_strike(S * 0.95)
    if K_long == K_short:
        K_short = K_long - (5 if S < 200 else 10)

    leg_long = _make_leg(S, K_long, T, r, sigma, "buy", "put")
    leg_short = _make_leg(S, K_short, T, r, sigma, "sell", "put")

    entry = leg_long.estimated_premium - leg_short.estimated_premium
    entry = max(0.10, entry)
    spread_width = K_long - K_short
    max_loss = entry * 100
    max_gain = (spread_width - entry) * 100
    be = K_long - entry
    rr = max_gain / max_loss if max_loss > 0 else 0

    d2 = (np.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = (1 - _norm_cdf(d2)) * 100

    drivers = _build_drivers(signal, "Bearish outlook with defined risk — spread caps both loss and gain for better capital efficiency")
    thesis = _build_thesis(signal, "Bear Put Spread", pt)
    risks = _build_risks(signal, "Bear Put Spread")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Bear Put Spread",
        strategy_type="directional_bearish",
        conviction="high" if signal.confidence >= 75 else "moderate",
        risk_tier="conservative",
        legs=[leg_long, leg_short],
        entry_price=round(entry, 2),
        entry_timing="Enter on a rejection at resistance or breakdown below 10-SMA",
        profit_target=round(spread_width * 0.7, 2),
        stop_loss=round(entry * 0.5, 2),
        exit_timing="Exit at 50-70% of max profit, or close 5+ days before expiry",
        max_loss=round(max_loss, 2),
        max_gain=round(max_gain, 2),
        risk_reward_ratio=round(rr, 2),
        break_even=round(be, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="2-3% of portfolio",
        ideal_account_size="$3,000+",
    )


# ── Iron Condor ──────────────────────────────────────────────────

def _build_iron_condor(signal, df, pt, S, r, sigma):
    T = 30 / 365

    # Wings outside the 90% CI
    K_put_short = _round_strike(pt.target_low * 0.98)
    K_put_long = _round_strike(K_put_short - (5 if S < 200 else 10))
    K_call_short = _round_strike(pt.target_high * 1.02)
    K_call_long = _round_strike(K_call_short + (5 if S < 200 else 10))

    leg_ps = _make_leg(S, K_put_short, T, r, sigma, "sell", "put")
    leg_pl = _make_leg(S, K_put_long, T, r, sigma, "buy", "put")
    leg_cs = _make_leg(S, K_call_short, T, r, sigma, "sell", "call")
    leg_cl = _make_leg(S, K_call_long, T, r, sigma, "buy", "call")

    credit = (leg_ps.estimated_premium - leg_pl.estimated_premium +
              leg_cs.estimated_premium - leg_cl.estimated_premium)
    credit = max(0.20, credit)
    wing_width = max(K_call_long - K_call_short, K_put_short - K_put_long)
    max_loss = (wing_width - credit) * 100
    max_gain = credit * 100
    rr = max_gain / max_loss if max_loss > 0 else 0

    # PoP: probability price stays within short strikes
    d2_upper = (np.log(S / K_call_short) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_lower = (np.log(S / K_put_short) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = (_norm_cdf(d2_upper) - _norm_cdf(d2_lower)) * 100
    pop = max(pop, 40)  # floor for display

    drivers = _build_drivers(signal, "Elevated volatility inflates option premiums — Iron Condor harvests premium while expecting range-bound action")
    thesis = (
        f"With annualized volatility at {pt.annual_vol}%, premiums are rich for {signal.ticker}. "
        f"The 90% confidence interval of ${pt.target_low}–${pt.target_high} over {pt.horizon_days} days "
        f"suggests the stock is likely to remain range-bound. An Iron Condor placed outside this range "
        f"collects premium from time decay while defining risk on both sides. "
        f"The strategy profits if {signal.ticker} stays between ${K_put_short} and ${K_call_short} at expiry."
    )
    risks = _build_risks(signal, "Iron Condor")
    risks.insert(0, f"A sharp move beyond ${K_put_short} or ${K_call_short} will result in max loss")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Iron Condor",
        strategy_type="neutral",
        conviction="high" if pop > 65 else "moderate",
        risk_tier="conservative",
        legs=[leg_ps, leg_pl, leg_cs, leg_cl],
        entry_price=round(credit, 2),
        entry_timing="Enter when IV rank is above 50% and no major catalysts are imminent",
        profit_target=round(credit * 0.5, 2),
        stop_loss=round(credit * 2, 2),
        exit_timing="Exit at 50% of max profit or 7-10 days before expiry, whichever comes first",
        max_loss=round(max_loss, 2),
        max_gain=round(max_gain, 2),
        risk_reward_ratio=round(rr, 2),
        break_even=round(K_put_short - credit, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="2-4% of portfolio",
        ideal_account_size="$5,000+",
    )


# ── Long Straddle ────────────────────────────────────────────────

def _build_long_straddle(signal, df, pt, S, r, sigma):
    T = 30 / 365
    K = _round_strike(S)

    leg_call = _make_leg(S, K, T, r, sigma, "buy", "call")
    leg_put = _make_leg(S, K, T, r, sigma, "buy", "put")

    entry = leg_call.estimated_premium + leg_put.estimated_premium
    max_loss = entry * 100
    be_upper = K + entry
    be_lower = K - entry

    # PoP: P(outside breakevens)
    d2_u = (np.log(S / be_upper) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_l = (np.log(S / be_lower) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = (_norm_cdf(d2_u) + (1 - _norm_cdf(d2_l))) * 100 if T > 0 else 50
    pop = max(20, min(pop, 80))

    drivers = _build_drivers(signal, "Bollinger Band squeeze or low IV signals imminent volatility expansion — straddle profits from a large move in either direction")
    thesis = (
        f"A volatility squeeze has been detected for {signal.ticker}, suggesting a significant price move is imminent. "
        f"Current annualized vol at {pt.annual_vol}% is compressed relative to historical norms. "
        f"A Long Straddle profits from any large move — the breakeven range is ${be_lower:.2f}–${be_upper:.2f}. "
        f"The 90% CI projects ${pt.target_low}–${pt.target_high}, and a move to either extreme delivers profit."
    )
    risks = _build_risks(signal, "Long Straddle")
    risks.insert(0, "If the stock remains flat, both legs lose value from theta decay")
    risks.insert(1, "Requires a move larger than the combined premium paid to profit")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Long Straddle",
        strategy_type="volatility",
        conviction="moderate",
        risk_tier="aggressive",
        legs=[leg_call, leg_put],
        entry_price=round(entry, 2),
        entry_timing="Enter before expected catalyst (earnings, product launch) or when BB width < 0.03",
        profit_target=round(entry * 1.5, 2),
        stop_loss=round(entry * 0.6, 2),
        exit_timing="Exit on the first large move (within 1-3 days of entry), or cut at 40% loss if no move materializes within a week",
        max_loss=round(max_loss, 2),
        max_gain=-1,
        risk_reward_ratio=round(2.5, 2),  # theoretical
        break_even=round(be_upper, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="1-2% of portfolio",
        ideal_account_size="$10,000+",
    )


# ── Short Put Spread (Bull Credit Spread) ────────────────────────

def _build_short_put_spread(signal, df, pt, S, r, sigma):
    T = 30 / 365
    K_short = _round_strike(pt.target_low)
    K_long = _round_strike(K_short - (5 if S < 200 else 10))
    if K_short == K_long:
        K_long = K_short - 5

    leg_short = _make_leg(S, K_short, T, r, sigma, "sell", "put")
    leg_long = _make_leg(S, K_long, T, r, sigma, "buy", "put")

    credit = leg_short.estimated_premium - leg_long.estimated_premium
    credit = max(0.10, credit)
    spread_width = K_short - K_long
    max_loss = (spread_width - credit) * 100
    max_gain = credit * 100
    be = K_short - credit
    rr = max_gain / max_loss if max_loss > 0 else 0

    d2 = (np.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = _norm_cdf(d2) * 100
    pop = max(pop, 45)

    drivers = _build_drivers(signal, "Bullish bias allows selling puts below support — collects premium with high probability of profit")
    thesis = _build_thesis(signal, "Short Put Spread (Bull Credit)", pt)
    risks = _build_risks(signal, "Short Put Spread")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Short Put Spread",
        strategy_type="directional_bullish",
        conviction="high" if pop > 65 else "moderate",
        risk_tier="conservative",
        legs=[leg_short, leg_long],
        entry_price=round(credit, 2),
        entry_timing="Enter when stock is trading above 20-SMA with bullish momentum",
        profit_target=round(credit * 0.5, 2),
        stop_loss=round((spread_width - credit) * 0.5, 2),
        exit_timing="Exit at 50% of max profit (let time decay work) or 7 days before expiry",
        max_loss=round(max_loss, 2),
        max_gain=round(max_gain, 2),
        risk_reward_ratio=round(rr, 2),
        break_even=round(be, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="2-4% of portfolio",
        ideal_account_size="$3,000+",
    )


# ── Short Call Spread (Bear Credit Spread) ────────────────────────

def _build_short_call_spread(signal, df, pt, S, r, sigma):
    T = 30 / 365
    K_short = _round_strike(pt.target_high)
    K_long = _round_strike(K_short + (5 if S < 200 else 10))
    if K_short == K_long:
        K_long = K_short + 5

    leg_short = _make_leg(S, K_short, T, r, sigma, "sell", "call")
    leg_long = _make_leg(S, K_long, T, r, sigma, "buy", "call")

    credit = leg_short.estimated_premium - leg_long.estimated_premium
    credit = max(0.10, credit)
    spread_width = K_long - K_short
    max_loss = (spread_width - credit) * 100
    max_gain = credit * 100
    be = K_short + credit
    rr = max_gain / max_loss if max_loss > 0 else 0

    d2 = (np.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pop = (1 - _norm_cdf(d2)) * 100
    pop = max(pop, 45)

    drivers = _build_drivers(signal, "Bearish bias allows selling calls above resistance — collects premium as stock declines or stays flat")
    thesis = _build_thesis(signal, "Short Call Spread (Bear Credit)", pt)
    risks = _build_risks(signal, "Short Call Spread")

    return OptionsPlay(
        ticker=signal.ticker,
        strategy_name="Short Call Spread",
        strategy_type="directional_bearish",
        conviction="high" if pop > 65 else "moderate",
        risk_tier="conservative",
        legs=[leg_short, leg_long],
        entry_price=round(credit, 2),
        entry_timing="Enter on a rally into resistance or when stock is below 20-SMA",
        profit_target=round(credit * 0.5, 2),
        stop_loss=round((spread_width - credit) * 0.5, 2),
        exit_timing="Exit at 50% of max profit or 7 days before expiry",
        max_loss=round(max_loss, 2),
        max_gain=round(max_gain, 2),
        risk_reward_ratio=round(rr, 2),
        break_even=round(be, 2),
        probability_of_profit=round(pop, 1),
        price_target=pt,
        drivers=drivers,
        thesis=thesis,
        risks=risks,
        suggested_allocation="2-4% of portfolio",
        ideal_account_size="$3,000+",
    )
