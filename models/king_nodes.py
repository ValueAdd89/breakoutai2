"""
King Node Detection — Volume Profile Analysis
===============================================
King nodes are High Volume Nodes (HVNs) where concentrated trading
interest creates strong support/resistance. When price is above a
king node it acts as support; below, it acts as resistance.

This module computes volume-at-price profiles and identifies the
dominant price levels (king nodes) where institutional interest clusters.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class KingNode:
    price_level: float
    volume: float
    volume_pct: float      # % of total volume at this level
    node_type: str          # "support" | "resistance" | "poc" (point of control)
    strength: str           # "king" | "major" | "minor"
    is_above_price: bool
    distance_pct: float    # distance from current price as %


@dataclass
class VolumeProfile:
    ticker: str
    current_price: float
    poc: float              # Point of Control — single highest volume price
    vah: float              # Value Area High (70th percentile)
    val: float              # Value Area Low (30th percentile)
    king_nodes: list        # list[KingNode]
    levels: list            # all price levels
    volumes: list           # volume at each level
    total_volume: float


def compute_volume_profile(df: pd.DataFrame, n_bins: int = 40) -> VolumeProfile | None:
    """
    Compute volume-at-price profile from OHLCV data.
    Distributes each bar's volume across the price range it covers.
    """
    if df is None or len(df) < 20:
        return None

    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Handle multi-column DataFrames
        for col in [close, high, low, volume]:
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]

        close_arr = np.asarray(close).astype(float).ravel()
        high_arr = np.asarray(high).astype(float).ravel()
        low_arr = np.asarray(low).astype(float).ravel()
        vol_arr = np.asarray(volume).astype(float).ravel()

        price_min = float(np.nanmin(low_arr))
        price_max = float(np.nanmax(high_arr))
        current_price = float(close_arr[-1])

        if price_max <= price_min or np.isnan(price_min):
            return None

        # Create price bins
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_at_price = np.zeros(n_bins)

        # Distribute each bar's volume across the bins it spans
        for i in range(len(close_arr)):
            bar_low = low_arr[i]
            bar_high = high_arr[i]
            bar_vol = vol_arr[i]
            if np.isnan(bar_low) or np.isnan(bar_high) or np.isnan(bar_vol):
                continue
            if bar_high <= bar_low:
                # Single-price bar: assign to nearest bin
                idx = np.argmin(np.abs(bin_centers - close_arr[i]))
                vol_at_price[idx] += bar_vol
            else:
                # Distribute proportionally across bins the bar spans
                mask = (bin_centers >= bar_low) & (bin_centers <= bar_high)
                n_bins_hit = mask.sum()
                if n_bins_hit > 0:
                    vol_at_price[mask] += bar_vol / n_bins_hit

        total_vol = vol_at_price.sum()
        if total_vol == 0:
            return None

        # Point of Control (highest volume level)
        poc_idx = np.argmax(vol_at_price)
        poc = float(bin_centers[poc_idx])

        # Value Area (70% of volume around POC)
        sorted_indices = np.argsort(vol_at_price)[::-1]
        cum_vol = 0
        va_indices = []
        for idx in sorted_indices:
            va_indices.append(idx)
            cum_vol += vol_at_price[idx]
            if cum_vol >= total_vol * 0.7:
                break
        va_prices = bin_centers[va_indices]
        vah = float(np.max(va_prices))
        val = float(np.min(va_prices))

        # Identify king nodes (significant volume peaks)
        vol_pct = vol_at_price / total_vol * 100
        mean_vol = np.mean(vol_at_price)
        std_vol = np.std(vol_at_price)

        king_nodes = []
        for i in range(n_bins):
            pct = float(vol_pct[i])
            level_price = float(bin_centers[i])
            dist = (level_price - current_price) / current_price * 100

            # Classify node strength
            if vol_at_price[i] > mean_vol + 2 * std_vol:
                strength = "king"
            elif vol_at_price[i] > mean_vol + std_vol:
                strength = "major"
            elif vol_at_price[i] > mean_vol + 0.5 * std_vol:
                strength = "minor"
            else:
                continue  # skip low-volume levels

            # Classify as support or resistance relative to current price
            is_above = level_price > current_price
            if i == poc_idx:
                node_type = "poc"
            elif is_above:
                node_type = "resistance"
            else:
                node_type = "support"

            king_nodes.append(KingNode(
                price_level=round(level_price, 4),
                volume=float(vol_at_price[i]),
                volume_pct=round(pct, 1),
                node_type=node_type,
                strength=strength,
                is_above_price=is_above,
                distance_pct=round(dist, 2),
            ))

        # Sort by volume descending
        king_nodes.sort(key=lambda n: n.volume, reverse=True)

        return VolumeProfile(
            ticker=df.attrs.get("ticker", ""),
            current_price=round(current_price, 4),
            poc=round(poc, 4),
            vah=round(vah, 4),
            val=round(val, 4),
            king_nodes=king_nodes,
            levels=[round(float(x), 4) for x in bin_centers],
            volumes=[float(x) for x in vol_at_price],
            total_volume=float(total_vol),
        )
    except Exception:
        return None
