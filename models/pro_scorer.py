"""
Professional-grade breakout scoring.

Extends the base 100-point breakout score with up to 50 additional "Pro Points"
sourced from data a professional trader actually uses:

  Float Tier Bonus (0-15 pts):
    Low-float runners are disproportionately explosive. A 5M-float
    penny stock with 2x RVol moves 10x harder than a 500M-float.

  Float Turnover Rate (0-10 pts):
    Volume as % of float. >50% turnover in one day is a major pro signal.

  Short Interest (0-10 pts):
    High short interest (>15%) primes a stock for short squeezes.

  News Catalyst (0-10 pts):
    Bullish high-importance news in the past 3 days supercharges breakouts.

  Intraday Structure (0-5 pts):
    Opening range break, above VWAP, new day highs — real-time confirmation.

Total possible score: 150 (base 100 + pro 50). Graded on the expanded scale.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ProBreakoutScore:
    base_score: float             # 0-100 from predictor
    pro_score: float              # 0-50 from this module
    total_score: float            # 0-150
    total_grade: str              # A+ through F

    # Breakdown
    float_tier: str = "Unknown"
    float_m: float = 0.0
    float_points: float = 0.0
    turnover_pct: float = 0.0
    turnover_points: float = 0.0
    short_pct_float: float = 0.0
    short_points: float = 0.0
    catalyst_points: float = 0.0
    catalyst_summary: str = ""
    intraday_points: float = 0.0
    intraday_summary: str = ""

    pro_factors: list = field(default_factory=list)


def compute_pro_breakout(
    base_score: float,
    ticker: str,
    daily_volume: float,
    profile=None,              # CompanyProfile from catalysts.py
    news_items=None,           # list[NewsItem] from catalysts.py
    intraday_stats=None,       # IntradayStats from intraday.py
) -> ProBreakoutScore:
    """Compute the professional breakout enhancement score."""
    factors = []
    float_pts = 0.0
    turnover_pts = 0.0
    short_pts = 0.0
    catalyst_pts = 0.0
    intraday_pts = 0.0

    float_tier = "Unknown"
    float_m = 0.0
    turnover = 0.0
    short_pct = 0.0
    catalyst_summary = ""
    intraday_summary = ""

    # ─── Float Tier (0-15 pts) ──────────────────────────────────
    if profile and profile.float_shares_m > 0:
        float_m = profile.float_shares_m
        if float_m < 10:
            float_tier = "Micro Float"
            float_pts = 15
            factors.append(("Micro Float", f"Only {float_m:.1f}M float — extreme volatility & breakout potential", "critical"))
        elif float_m < 50:
            float_tier = "Low Float"
            float_pts = 12
            factors.append(("Low Float", f"{float_m:.1f}M float — classic runner setup", "strong"))
        elif float_m < 200:
            float_tier = "Medium Float"
            float_pts = 6
            factors.append(("Medium Float", f"{float_m:.1f}M float — moderate breakout leverage", "moderate"))
        elif float_m < 1000:
            float_tier = "Large Float"
            float_pts = 3
            factors.append(("Large Float", f"{float_m:.1f}M float — institutional-grade, slower moves", "weak"))
        else:
            float_tier = "Mega Float"
            float_pts = 1
            factors.append(("Mega Float", f"{float_m:.0f}M float — mega-cap, limited breakout dynamics", "neutral"))
    else:
        factors.append(("Float Unknown", "Float data unavailable — scoring skipped", "neutral"))

    # ─── Float Turnover Rate (0-10 pts) ─────────────────────────
    if profile and profile.float_shares_m > 0 and daily_volume > 0:
        float_shares = profile.float_shares_m * 1_000_000
        turnover = (daily_volume / float_shares) * 100
        if turnover >= 100:
            turnover_pts = 10
            factors.append(("Float Flipped", f"{turnover:.0f}% of float traded — entire float rotated, explosive setup", "critical"))
        elif turnover >= 50:
            turnover_pts = 8
            factors.append(("Heavy Turnover", f"{turnover:.1f}% of float traded — institutional rotation", "strong"))
        elif turnover >= 25:
            turnover_pts = 6
            factors.append(("Strong Turnover", f"{turnover:.1f}% of float traded — elevated participation", "strong"))
        elif turnover >= 10:
            turnover_pts = 3
            factors.append(("Moderate Turnover", f"{turnover:.1f}% of float traded", "moderate"))
        elif turnover >= 5:
            turnover_pts = 1
            factors.append(("Light Turnover", f"{turnover:.1f}% of float traded — normal activity", "weak"))
        else:
            factors.append(("Low Turnover", f"{turnover:.1f}% of float traded — no institutional interest", "negative"))

    # ─── Short Interest / Squeeze Potential (0-10 pts) ──────────
    if profile:
        short_pct = profile.short_pct_float
        if short_pct >= 30:
            short_pts = 10
            factors.append(("Squeeze Primed", f"{short_pct:.1f}% short — extreme squeeze potential", "critical"))
        elif short_pct >= 20:
            short_pts = 8
            factors.append(("High Short Interest", f"{short_pct:.1f}% short — squeeze candidate", "strong"))
        elif short_pct >= 15:
            short_pts = 6
            factors.append(("Elevated Short", f"{short_pct:.1f}% short — watch for squeeze triggers", "strong"))
        elif short_pct >= 8:
            short_pts = 3
            factors.append(("Moderate Short", f"{short_pct:.1f}% short", "moderate"))
        elif short_pct > 0:
            factors.append(("Low Short Interest", f"{short_pct:.1f}% short — no squeeze setup", "neutral"))

    # ─── News / Catalyst Score (0-10 pts) ───────────────────────
    if news_items:
        recent_cutoff = datetime.utcnow() - timedelta(days=3)
        recent = [n for n in news_items if n.datetime_utc >= recent_cutoff]
        bull_items = [n for n in recent if n.sentiment == "bullish"]
        bear_items = [n for n in recent if n.sentiment == "bearish"]
        high_imp_bull = [n for n in bull_items if n.importance >= 2]

        if len(high_imp_bull) >= 2:
            catalyst_pts = 10
            catalyst_summary = f"{len(high_imp_bull)} high-importance bullish catalysts in last 3 days"
            factors.append(("Major Catalysts", catalyst_summary, "critical"))
        elif len(high_imp_bull) == 1:
            catalyst_pts = 7
            catalyst_summary = f"High-importance bullish news: {high_imp_bull[0].headline[:80]}"
            factors.append(("Catalyst Active", catalyst_summary, "strong"))
        elif len(bull_items) >= 3:
            catalyst_pts = 5
            catalyst_summary = f"{len(bull_items)} bullish headlines in last 3 days"
            factors.append(("Positive News Flow", catalyst_summary, "moderate"))
        elif len(bull_items) >= 1 and len(bear_items) == 0:
            catalyst_pts = 3
            catalyst_summary = f"{len(bull_items)} bullish news, no bearish"
            factors.append(("Mild Positive Flow", catalyst_summary, "weak"))
        elif len(bear_items) > len(bull_items):
            catalyst_pts = -3
            catalyst_summary = f"{len(bear_items)} bearish vs {len(bull_items)} bullish — headwind"
            factors.append(("Negative News Flow", catalyst_summary, "negative"))
        else:
            catalyst_summary = "No material catalysts in last 3 days"
            factors.append(("Quiet News", catalyst_summary, "neutral"))
    else:
        catalyst_summary = "News feed unavailable (set FINNHUB_API_KEY)"
        factors.append(("News N/A", catalyst_summary, "neutral"))

    # ─── Intraday Structure (0-5 pts) ───────────────────────────
    if intraday_stats:
        pts = 0
        sub = []
        if intraday_stats.above_vwap:
            pts += 2
            sub.append("above VWAP")
        else:
            sub.append("below VWAP")
        if intraday_stats.opening_range_break:
            pts += 2
            sub.append("ORB triggered")
        if intraday_stats.breakout_of_day:
            pts += 1
            sub.append("new day high")
        if intraday_stats.day_change_pct > 2:
            pts = min(pts + 1, 5)
            sub.append(f"+{intraday_stats.day_change_pct:.1f}% intraday")
        intraday_pts = max(0, pts)
        intraday_summary = ", ".join(sub) if sub else "flat session"
        imp = "strong" if intraday_pts >= 4 else "moderate" if intraday_pts >= 2 else "weak"
        factors.append(("Intraday Structure", intraday_summary, imp))
    else:
        intraday_summary = "No intraday data (market may be closed)"

    # ─── Totals ──────────────────────────────────────────────────
    pro_score = float_pts + turnover_pts + short_pts + catalyst_pts + intraday_pts
    pro_score = max(0, pro_score)  # clamp negative catalyst
    total = base_score + pro_score

    # Grading on 0-150 scale
    if total >= 130:
        grade = "A+"
    elif total >= 115:
        grade = "A"
    elif total >= 100:
        grade = "B+"
    elif total >= 85:
        grade = "B"
    elif total >= 70:
        grade = "C+"
    elif total >= 55:
        grade = "C"
    elif total >= 40:
        grade = "D"
    else:
        grade = "F"

    return ProBreakoutScore(
        base_score=round(base_score, 1),
        pro_score=round(pro_score, 1),
        total_score=round(total, 1),
        total_grade=grade,
        float_tier=float_tier,
        float_m=round(float_m, 1),
        float_points=float_pts,
        turnover_pct=round(turnover, 1),
        turnover_points=turnover_pts,
        short_pct_float=round(short_pct, 1),
        short_points=short_pts,
        catalyst_points=catalyst_pts,
        catalyst_summary=catalyst_summary,
        intraday_points=intraday_pts,
        intraday_summary=intraday_summary,
        pro_factors=factors,
    )
