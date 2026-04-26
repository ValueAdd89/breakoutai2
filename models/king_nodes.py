"""
Volume Profile Analysis — Skylit Heatseeker-Inspired
=====================================================
Implements the key concepts from Skylit's dealer positioning framework
using freely available OHLCV data:

  King Nodes     — Highest volume levels. Price gravitates toward these.
  Gatekeepers    — Secondary barriers between price and the King Node.
  Air Pockets    — Low-volume zones where price moves fast (no resistance).
  Pika Zones     — Levels where price stabilizes (high volume, price holds).
                   Analogous to positive GEX — "magnetic pillow."
  Barney Zones   — Levels where price whips violently (high vol, price breaks).
                   Analogous to negative GEX — "gasoline on fire."
  Node Lifecycle — Fresh (untested) nodes are strongest. Each retest weakens them.
  POC / Value Area — Standard volume profile metrics.

Note: Real GEX requires live options chain data ($$$). We approximate
the structural concepts using volume-at-price analysis which captures
similar dynamics — where institutional interest clusters and where
price moves freely.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Node:
    price_level: float
    volume: float
    volume_pct: float
    node_type: str          # "king" | "gatekeeper" | "cluster" | "air_pocket"
    polarity: str           # "pika" (stabilizing) | "barney" (volatile) | "neutral"
    role: str               # "support" | "resistance" | "poc"
    strength: float         # 0-100 magnitude score
    is_above_price: bool
    distance_pct: float
    times_tested: int       # lifecycle: 0 = fresh (strongest), each retest weakens
    description: str


@dataclass
class VolumeProfile:
    ticker: str
    current_price: float
    poc: float
    vah: float
    val: float
    king_node: Node | None
    gatekeepers: list       # list[Node]
    air_pockets: list       # list[Node]
    pika_zones: list        # list[Node] — stabilizing
    barney_zones: list      # list[Node] — volatile
    all_nodes: list         # list[Node] — full hierarchy
    levels: list
    volumes: list
    total_volume: float
    regime: str             # "trending" | "range_bound" | "volatile"
    regime_description: str


def compute_volume_profile(df: pd.DataFrame, n_bins: int = 50) -> VolumeProfile | None:
    """Compute full Skylit-style volume profile from OHLCV data."""
    if df is None or len(df) < 20:
        return None

    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        opn = df["Open"]

        for arr in [close, high, low, volume, opn]:
            if isinstance(arr, pd.DataFrame):
                arr = arr.iloc[:, 0]

        close_arr = np.asarray(close).astype(float).ravel()
        high_arr = np.asarray(high).astype(float).ravel()
        low_arr = np.asarray(low).astype(float).ravel()
        vol_arr = np.asarray(volume).astype(float).ravel()
        open_arr = np.asarray(opn).astype(float).ravel()

        price_min = float(np.nanmin(low_arr))
        price_max = float(np.nanmax(high_arr))
        current = float(close_arr[-1])

        if price_max <= price_min or np.isnan(price_min):
            return None

        # ── Build volume-at-price histogram ───────────────────
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_at_price = np.zeros(n_bins)
        # Track price action behavior at each level
        bars_at_level = np.zeros(n_bins)   # how many bars touched this level
        wicks_at_level = np.zeros(n_bins)  # wick rejections (price went there but closed away)

        for i in range(len(close_arr)):
            bl, bh, bc, bo = low_arr[i], high_arr[i], close_arr[i], open_arr[i]
            bv = vol_arr[i]
            if np.isnan(bl) or np.isnan(bh) or np.isnan(bv):
                continue

            mask = (bin_centers >= bl) & (bin_centers <= bh)
            n_hit = mask.sum()
            if n_hit > 0:
                vol_at_price[mask] += bv / n_hit
                bars_at_level[mask] += 1

            # Wick detection: price reached the level but closed away
            body_high = max(bc, bo)
            body_low = min(bc, bo)
            wick_mask_upper = (bin_centers > body_high) & (bin_centers <= bh)
            wick_mask_lower = (bin_centers < body_low) & (bin_centers >= bl)
            wicks_at_level[wick_mask_upper] += 1
            wicks_at_level[wick_mask_lower] += 1

        total_vol = vol_at_price.sum()
        if total_vol == 0:
            return None

        vol_pct = vol_at_price / total_vol * 100
        mean_vol = np.mean(vol_at_price)
        std_vol = np.std(vol_at_price)

        # ── POC and Value Area ────────────────────────────────
        poc_idx = np.argmax(vol_at_price)
        poc = float(bin_centers[poc_idx])

        sorted_idx = np.argsort(vol_at_price)[::-1]
        cum = 0
        va_idx = []
        for idx in sorted_idx:
            va_idx.append(idx)
            cum += vol_at_price[idx]
            if cum >= total_vol * 0.7:
                break
        va_prices = bin_centers[va_idx]
        vah = float(np.max(va_prices))
        val = float(np.min(va_prices))

        # ── Classify every level ──────────────────────────────
        all_nodes = []
        for i in range(n_bins):
            level = float(bin_centers[i])
            v = vol_at_price[i]
            pct = float(vol_pct[i])
            dist = (level - current) / current * 100
            above = level > current

            # Strength: normalized 0-100
            if std_vol > 0:
                z_score = (v - mean_vol) / std_vol
            else:
                z_score = 0
            strength = min(100, max(0, 50 + z_score * 20))

            # Times tested: approximate from bar count
            tested = int(bars_at_level[i])
            wick_count = int(wicks_at_level[i])

            # Polarity classification (Pika vs Barney)
            # Pika: high volume + many wicks (price comes here and bounces = stabilizing)
            # Barney: high volume + few wicks relative to bars (price breaks through = volatile)
            wick_ratio = wick_count / max(1, tested)

            if v > mean_vol + 1.5 * std_vol:
                # High volume level
                if wick_ratio > 0.4:
                    polarity = "pika"  # lots of wicks = price rejected here, stabilizing
                else:
                    polarity = "barney"  # price moved through = volatile
            elif v < mean_vol - 0.5 * std_vol:
                polarity = "neutral"  # low volume
            else:
                polarity = "neutral"

            # Role
            if i == poc_idx:
                role = "poc"
            elif above:
                role = "resistance"
            else:
                role = "support"

            # Node type classification
            if i == poc_idx or v >= mean_vol + 2 * std_vol:
                node_type = "king"
            elif v >= mean_vol + std_vol:
                node_type = "gatekeeper"
            elif v >= mean_vol + 0.3 * std_vol:
                node_type = "cluster"
            elif v < mean_vol - 0.5 * std_vol:
                node_type = "air_pocket"
            else:
                continue  # skip unremarkable levels

            # Description
            if node_type == "king":
                desc = f"King Node — highest institutional interest. Price gravitates here. {'Pika (stabilizing): expect price to slow and potentially pin.' if polarity == 'pika' else 'Barney (volatile): expect explosive moves if price reaches this level.'}"
            elif node_type == "gatekeeper":
                desc = f"Gatekeeper — barrier between current price and King Node. {'Price tends to bounce here (Pika).' if polarity == 'pika' else 'Price tends to spike through (Barney).' if polarity == 'barney' else 'Moderate resistance/support.'}"
            elif node_type == "air_pocket":
                desc = f"Air Pocket — low volume zone. Price moves fast through here with minimal resistance. {'Fresh (untested) — strongest effect.' if tested < 3 else f'Tested {tested}x — effect may be weakening.'}"
            else:
                desc = f"Volume cluster — moderate institutional interest."

            all_nodes.append(Node(
                price_level=round(level, 4),
                volume=float(v),
                volume_pct=round(pct, 1),
                node_type=node_type,
                polarity=polarity,
                role=role,
                strength=round(strength, 1),
                is_above_price=above,
                distance_pct=round(dist, 2),
                times_tested=tested,
                description=desc,
            ))

        # Sort by strength descending
        all_nodes.sort(key=lambda n: n.strength, reverse=True)

        # Separate into categories
        king = next((n for n in all_nodes if n.node_type == "king"), None)
        gatekeepers = [n for n in all_nodes if n.node_type == "gatekeeper"][:4]
        air_pockets = sorted([n for n in all_nodes if n.node_type == "air_pocket"],
                             key=lambda n: abs(n.distance_pct))[:4]
        pika = [n for n in all_nodes if n.polarity == "pika"][:5]
        barney = [n for n in all_nodes if n.polarity == "barney"][:5]

        # ── Market regime detection ───────────────────────────
        # Based on Skylit's framework:
        # - Pika-dominant near price = range-bound (choppy, good for selling premium)
        # - Barney-dominant near price = trending/volatile (good for momentum)
        # - Mixed = transitional
        near_nodes = [n for n in all_nodes if abs(n.distance_pct) < 3]
        pika_near = sum(1 for n in near_nodes if n.polarity == "pika")
        barney_near = sum(1 for n in near_nodes if n.polarity == "barney")

        if pika_near > barney_near + 1:
            regime = "range_bound"
            regime_desc = ("Pika-dominant near current price — expect low volatility, "
                           "choppy price action. Price likely to pin between key levels. "
                           "Ideal for premium-selling strategies.")
        elif barney_near > pika_near + 1:
            regime = "volatile"
            regime_desc = ("Barney-dominant near current price — expect high volatility, "
                           "directional moves. Once price breaks a level, it accelerates. "
                           "Ideal for momentum and breakout strategies.")
        else:
            regime = "trending"
            regime_desc = ("Mixed Pika/Barney environment — transitional regime. "
                           "Watch for Gatekeeper rejections to signal direction. "
                           "Flexible approach recommended.")

        return VolumeProfile(
            ticker=df.attrs.get("ticker", ""),
            current_price=round(current, 4),
            poc=round(poc, 4),
            vah=round(vah, 4),
            val=round(val, 4),
            king_node=king,
            gatekeepers=gatekeepers,
            air_pockets=air_pockets,
            pika_zones=pika,
            barney_zones=barney,
            all_nodes=all_nodes,
            levels=[round(float(x), 4) for x in bin_centers],
            volumes=[float(x) for x in vol_at_price],
            total_volume=float(total_vol),
            regime=regime,
            regime_description=regime_desc,
        )
    except Exception:
        return None
