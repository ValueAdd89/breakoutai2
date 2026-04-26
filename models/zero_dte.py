"""
0DTE Options Analysis Module
==============================
Built for SPX/SPY/QQQ same-day expiration trading.

Core components:
  • Session window identification (ORB, prime scalp, premium sell, power hour)
  • Expected move calculation from ATR and historical volatility
  • Intraday regime detection (trending/choppy/volatile)
  • 0DTE strategy recommendations based on regime
  • Theta decay curve modeling
  • Position sizing calculator
  • Key level detection tuned for intraday timeframe
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, time


# ═══════════════════════════════════════════════════════════════════
# 0DTE Session Windows
# ═══════════════════════════════════════════════════════════════════

ZERO_DTE_TICKERS = ["SPY", "QQQ", "IWM", "SPX", "SPXW"]

SESSION_WINDOWS = [
    {"name": "Pre-Market Prep", "start": "04:00", "end": "09:30",
     "action": "Identify key levels, check overnight news, set bias. DO NOT TRADE.",
     "color": "#FFD60A"},
    {"name": "Opening Range (ORB)", "start": "09:30", "end": "09:45",
     "action": "Let the opening range establish. Mark the 15-min high and low. Wait for breakout confirmation.",
     "color": "#FF9F0A"},
    {"name": "Directional Window", "start": "09:45", "end": "10:30",
     "action": "Best window for directional 0DTE plays. Enter on ORB break with confirmation. Hold 30-90 min max.",
     "color": "#34C759"},
    {"name": "Prime Scalping", "start": "10:30", "end": "11:30",
     "action": "Highest probability window. Theta starting to bite. ATR compressed from open. Scalps and gamma fades.",
     "color": "#0A84FF"},
    {"name": "Midday Chop", "start": "11:30", "end": "13:30",
     "action": "Low volume, high chop risk. Premium selling only (credit spreads, iron condors). Or sit out.",
     "color": "#BF5AF2"},
    {"name": "Afternoon Setup", "start": "13:30", "end": "14:30",
     "action": "Theta accelerating hard. New setups for final push. Watch for trend resumption or reversal.",
     "color": "#FFD60A"},
    {"name": "Power Hour", "start": "14:30", "end": "15:30",
     "action": "Dealer hedging intensifies. Pin risk rises. Close directional trades. Only experts scalp here.",
     "color": "#FF453A"},
    {"name": "Final 30 Min", "start": "15:30", "end": "16:00",
     "action": "EXTREME gamma. Options go to $0 or explode. Close everything unless you have a specific pin thesis.",
     "color": "#FF453A"},
]


def get_current_session_window() -> dict | None:
    """Return the current 0DTE session window based on ET time."""
    try:
        now = pd.Timestamp.now(tz="America/New_York")
        current_time = now.strftime("%H:%M")
        for w in SESSION_WINDOWS:
            if w["start"] <= current_time < w["end"]:
                return w
        return None
    except Exception:
        return None


def get_all_session_windows() -> list[dict]:
    return SESSION_WINDOWS


# ═══════════════════════════════════════════════════════════════════
# Expected Move & Theta Decay
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ZeroDTEAnalysis:
    ticker: str
    current_price: float
    # Expected move
    expected_move_1sd: float      # 1 standard deviation expected move $
    expected_move_pct: float      # as percentage
    expected_range_high: float
    expected_range_low: float
    # Session info
    current_window: str
    current_window_action: str
    current_window_color: str
    minutes_to_close: int
    # Regime
    regime: str                   # "trending" | "range_bound" | "volatile"
    regime_description: str
    regime_strategy: str          # recommended strategy for this regime
    # Key levels (intraday)
    intraday_high: float
    intraday_low: float
    intraday_vwap: float
    opening_range_high: float
    opening_range_low: float
    orb_broken: str               # "above" | "below" | "inside"
    # Theta decay
    theta_per_hour: float         # estimated $/hour for ATM option
    theta_remaining: float        # estimated total theta left today
    # Strategy recommendations
    strategies: list              # list of (strategy_name, description, risk_level)
    # Risk management
    max_position_size_pct: float  # % of account per trade
    suggested_stop: str
    time_stop: str                # when to close regardless


def compute_0dte_analysis(ticker: str, df_daily: pd.DataFrame,
                          df_intraday: pd.DataFrame = None,
                          account_size: float = 25000) -> ZeroDTEAnalysis | None:
    """
    Compute comprehensive 0DTE analysis for a ticker.
    Uses daily data for expected move + intraday data for session analysis.
    """
    if df_daily is None or len(df_daily) < 20:
        return None

    try:
        close = df_daily["Close"]
        high = df_daily["High"]
        low = df_daily["Low"]
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        if isinstance(high, pd.DataFrame): high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame): low = low.iloc[:, 0]

        current_price = float(close.iloc[-1])
        log_returns = np.diff(np.log(close.values.astype(float)))

        # ── Expected Move (1σ) ────────────────────────────────
        # Use 20-day realized vol, annualize, then scale to 1 day
        vol_20d = np.std(log_returns[-20:]) * np.sqrt(252)
        daily_vol = vol_20d / np.sqrt(252)
        expected_move = current_price * daily_vol
        expected_pct = daily_vol * 100

        # ── ATR for intraday range ────────────────────────────
        tr = np.maximum(
            high.values[-14:].astype(float) - low.values[-14:].astype(float),
            np.maximum(
                np.abs(high.values[-14:].astype(float) - close.values[-15:-1].astype(float)),
                np.abs(low.values[-14:].astype(float) - close.values[-15:-1].astype(float))
            )
        )
        atr = float(np.mean(tr))

        # ── Session window ────────────────────────────────────
        window = get_current_session_window()
        if window:
            w_name = window["name"]
            w_action = window["action"]
            w_color = window["color"]
        else:
            w_name = "Market Closed"
            w_action = "Pre-market prep: identify key levels, set bias, plan entries for tomorrow's session."
            w_color = "rgba(255,255,255,0.3)"

        # Minutes to close
        try:
            now = pd.Timestamp.now(tz="America/New_York")
            close_time = now.replace(hour=16, minute=0, second=0)
            mins_left = max(0, int((close_time - now).total_seconds() / 60))
        except:
            mins_left = 0

        # ── Intraday analysis ─────────────────────────────────
        intraday_high = current_price + atr / 2
        intraday_low = current_price - atr / 2
        intraday_vwap = current_price
        or_high = current_price + atr * 0.2
        or_low = current_price - atr * 0.2
        orb_status = "inside"

        if df_intraday is not None and len(df_intraday) >= 3:
            id_close = df_intraday["Close"]
            id_high = df_intraday["High"]
            id_low = df_intraday["Low"]
            id_vol = df_intraday["Volume"]
            if isinstance(id_close, pd.DataFrame): id_close = id_close.iloc[:, 0]
            if isinstance(id_high, pd.DataFrame): id_high = id_high.iloc[:, 0]
            if isinstance(id_low, pd.DataFrame): id_low = id_low.iloc[:, 0]
            if isinstance(id_vol, pd.DataFrame): id_vol = id_vol.iloc[:, 0]

            # Get today's session
            df_intraday_copy = df_intraday.copy()
            df_intraday_copy.index = pd.to_datetime(df_intraday_copy.index)
            if hasattr(df_intraday_copy.index, 'date'):
                dates = df_intraday_copy.index.date
                latest_date = max(dates)
                today = df_intraday_copy[dates == latest_date]

                if len(today) >= 3:
                    intraday_high = float(today["High"].max()) if isinstance(today["High"].max(), (int, float, np.floating)) else float(today["High"].values.ravel().max())
                    intraday_low = float(today["Low"].min()) if isinstance(today["Low"].min(), (int, float, np.floating)) else float(today["Low"].values.ravel().min())

                    # VWAP
                    t_close = today["Close"] if not isinstance(today["Close"], pd.DataFrame) else today["Close"].iloc[:, 0]
                    t_high = today["High"] if not isinstance(today["High"], pd.DataFrame) else today["High"].iloc[:, 0]
                    t_low = today["Low"] if not isinstance(today["Low"], pd.DataFrame) else today["Low"].iloc[:, 0]
                    t_vol = today["Volume"] if not isinstance(today["Volume"], pd.DataFrame) else today["Volume"].iloc[:, 0]

                    typical = (t_high.values.ravel().astype(float) + t_low.values.ravel().astype(float) + t_close.values.ravel().astype(float)) / 3
                    vol_arr = t_vol.values.ravel().astype(float)
                    cum_vol = np.cumsum(vol_arr)
                    if cum_vol[-1] > 0:
                        intraday_vwap = float(np.sum(typical * vol_arr) / cum_vol[-1])
                    else:
                        intraday_vwap = current_price

                    # Opening range (first 3 bars = ~15 min)
                    or_bars = today.head(3)
                    or_h = or_bars["High"] if not isinstance(or_bars["High"], pd.DataFrame) else or_bars["High"].iloc[:, 0]
                    or_l = or_bars["Low"] if not isinstance(or_bars["Low"], pd.DataFrame) else or_bars["Low"].iloc[:, 0]
                    or_high = float(or_h.values.ravel().max())
                    or_low = float(or_l.values.ravel().min())

                    if current_price > or_high:
                        orb_status = "above"
                    elif current_price < or_low:
                        orb_status = "below"
                    else:
                        orb_status = "inside"

        # ── Regime detection ──────────────────────────────────
        recent_returns = log_returns[-5:]
        vol_ratio = np.std(log_returns[-5:]) / np.std(log_returns[-20:]) if np.std(log_returns[-20:]) > 0 else 1
        trend = np.mean(recent_returns)

        if vol_ratio > 1.5:
            regime = "volatile"
            regime_desc = "High intraday volatility — wide swings, dealer hedging flows dominate. Expect whipsaws at key levels."
            regime_strat = "Fade extremes at king nodes. Sell premium via credit spreads. Wide stops. Reduce size by 50%."
        elif abs(trend) > daily_vol * 0.3:
            regime = "trending"
            regime_desc = "Directional regime — price moving with conviction. Breakout plays and trend following work."
            regime_strat = "Trade ORB breakout direction. Hold for 30-90 min. Trail stops behind VWAP."
        else:
            regime = "range_bound"
            regime_desc = "Choppy/range-bound — price oscillating between levels. Pin risk elevated."
            regime_strat = "Sell premium (iron condors, credit spreads). Fade edges of range. Don't chase breakouts."

        # ── Theta decay modeling ──────────────────────────────
        # ATM option premium estimate (simplified)
        atm_premium = current_price * daily_vol * 0.4  # rough BS estimate
        hours_left = max(0.5, mins_left / 60)
        total_hours = 6.5  # trading session
        # Theta acceleration: exponential decay curve
        # Early: ~$0.15-0.25/hr, Midday: ~$0.40-0.55/hr, Final hour: ~$0.80-1.20/hr
        decay_rate = atm_premium / total_hours * (total_hours / max(0.5, hours_left)) ** 0.5
        theta_remaining = atm_premium * (hours_left / total_hours) ** 1.5

        # ── Strategy recommendations ──────────────────────────
        strategies = _build_0dte_strategies(regime, orb_status, hours_left, current_price,
                                           expected_move, intraday_vwap, or_high, or_low)

        # ── Risk management ───────────────────────────────────
        max_pct = 1.0 if regime == "volatile" else 2.0
        if hours_left < 1.5:
            stop = "Tighter stops — $0.20-0.50 on options. Gamma risk extreme."
            time_s = "Close ALL positions by 3:45 PM ET unless you have a specific pin thesis."
        elif hours_left < 3:
            stop = "Stop at 30-40% of premium paid. Move to breakeven after 50% gain."
            time_s = f"Close by 2:30 PM ET ({hours_left:.1f}h remain). Theta accelerating."
        else:
            stop = "Stop at 50% of premium paid. Take profits at 100% gain."
            time_s = f"Close directional trades by 1:00 PM ET. {hours_left:.1f}h remain."

        return ZeroDTEAnalysis(
            ticker=ticker,
            current_price=round(current_price, 2),
            expected_move_1sd=round(expected_move, 2),
            expected_move_pct=round(expected_pct, 2),
            expected_range_high=round(current_price + expected_move, 2),
            expected_range_low=round(current_price - expected_move, 2),
            current_window=w_name,
            current_window_action=w_action,
            current_window_color=w_color,
            minutes_to_close=mins_left,
            regime=regime,
            regime_description=regime_desc,
            regime_strategy=regime_strat,
            intraday_high=round(intraday_high, 2),
            intraday_low=round(intraday_low, 2),
            intraday_vwap=round(intraday_vwap, 2),
            opening_range_high=round(or_high, 2),
            opening_range_low=round(or_low, 2),
            orb_broken=orb_status,
            theta_per_hour=round(decay_rate, 2),
            theta_remaining=round(theta_remaining, 2),
            strategies=strategies,
            max_position_size_pct=max_pct,
            suggested_stop=stop,
            time_stop=time_s,
        )
    except Exception:
        return None


def _build_0dte_strategies(regime, orb, hours_left, price, exp_move, vwap,
                           or_high, or_low) -> list[tuple]:
    """Build 0DTE strategy recommendations based on regime and session state."""
    strats = []

    if regime == "trending":
        if orb == "above":
            strats.append(("ORB Breakout Long", f"Price broke above opening range (${or_high:.2f}). Buy ATM call, target +${exp_move*0.5:.2f}. Stop if price falls back below ORB high.", "moderate"))
        elif orb == "below":
            strats.append(("ORB Breakdown Short", f"Price broke below opening range (${or_low:.2f}). Buy ATM put, target -${exp_move*0.5:.2f}. Stop if price recovers above ORB low.", "moderate"))
        strats.append(("VWAP Trend Follow", f"Trade in direction of VWAP (${vwap:.2f}). Long above VWAP, short below. Trail stop ${exp_move*0.3:.2f} behind entry.", "moderate"))

    elif regime == "range_bound":
        strats.append(("Iron Condor", f"Sell wings outside expected range (${price-exp_move:.2f}–${price+exp_move:.2f}). Collect premium. Max risk = wing width minus credit.", "conservative"))
        strats.append(("Fade the Edges", f"Sell calls at intraday high, sell puts at intraday low. Or buy reversals at ORB extremes.", "moderate"))
        strats.append(("Pin Play", f"If price gravitating to a round number or high-OI strike near ${round(price/5)*5:.0f}, sell straddle or buy butterfly.", "aggressive"))

    elif regime == "volatile":
        strats.append(("Gamma Scalp", f"Buy ATM straddle, delta-hedge by selling/buying shares as price swings. Profit from realized vol > implied vol.", "aggressive"))
        strats.append(("Wide Credit Spread", f"Sell far OTM spread (10+ points wide) outside 1.5σ range. Lower probability of loss but smaller credit.", "conservative"))

    # Time-dependent strategies
    if hours_left < 2:
        strats.append(("Theta Harvest", "Sell ATM credit spreads. Theta decay is exponential now — time is your edge. Keep position small.", "conservative"))
    if hours_left < 1:
        strats.append(("Close Everything", "Under 1 hour left. Gamma risk is extreme. Close all directional positions. Only pin plays from here.", "exit"))

    return strats


def get_0dte_tickers() -> list[str]:
    """Return tickers suitable for 0DTE trading."""
    return ["SPY", "QQQ", "IWM"]
