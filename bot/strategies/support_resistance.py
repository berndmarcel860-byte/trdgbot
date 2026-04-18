"""Support & Resistance entry strategy.

Detection algorithm
-------------------
1. Identify *pivot highs* (swing highs) and *pivot lows* (swing lows) using
   a rolling window (``pivot_lookback`` candles either side).
2. Cluster nearby pivots into S/R **zones** (within ``zone_tolerance`` %).
3. A zone is valid when it has been touched at least ``min_touches`` times.
4. A **long** signal fires when:
   - Price is near (within tolerance) a *support* zone.
   - ADX ≥ ``adx_threshold`` (trending market).
   - Stochastic RSI K-line is oversold (< 20) and turning up.
5. A **short** signal fires at *resistance* zones with the mirror conditions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from bot.strategies.base import BaseStrategy, Signal


def _find_pivots(
    df: pd.DataFrame,
    lookback: int = 20,
) -> Tuple[List[float], List[float]]:
    """Return (pivot_highs, pivot_lows) price lists."""
    highs: List[float] = []
    lows: List[float] = []
    n = len(df)
    for i in range(lookback, n - lookback):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        if h == df["high"].iloc[i - lookback: i + lookback + 1].max():
            highs.append(float(h))
        if l == df["low"].iloc[i - lookback: i + lookback + 1].min():
            lows.append(float(l))
    return highs, lows


def _cluster_levels(
    prices: List[float],
    tolerance: float = 0.003,
    min_touches: int = 2,
) -> List[float]:
    """Cluster nearby price levels into zones.

    Args:
        prices: Raw pivot prices.
        tolerance: Relative price distance to consider two prices as the
                   same level (default 0.3 %).
        min_touches: Minimum cluster size to qualify as a valid zone.

    Returns:
        List of representative zone prices.
    """
    if not prices:
        return []
    sorted_prices = sorted(prices)
    clusters: List[List[float]] = [[sorted_prices[0]]]
    for price in sorted_prices[1:]:
        if abs(price - clusters[-1][-1]) / clusters[-1][-1] <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [float(np.mean(c)) for c in clusters if len(c) >= min_touches]


class SupportResistanceStrategy(BaseStrategy):
    """Entry strategy based on validated support and resistance zones."""

    @property
    def name(self) -> str:
        return "support_resistance"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        cfg = self.strategy_cfg
        lookback = cfg.get("pivot_lookback", 20)
        tolerance = cfg.get("zone_tolerance", 0.003)
        min_touches = cfg.get("min_touches", 2)
        adx_threshold = cfg.get("adx_threshold", 20)

        required = ["high", "low", "close", "atr", "adx", "stochrsi_k", "stochrsi_d"]
        if df.empty or len(df) < lookback * 2 + 1 or not all(c in df.columns for c in required):
            return Signal(
                direction="neutral", strategy=self.name, symbol=symbol,
                meta={"reason": "insufficient data or missing columns"},
            )

        close = self.latest(df, "close")
        atr = self.latest(df, "atr")
        adx = self.latest(df, "adx")
        srsi_k = self.latest(df, "stochrsi_k")
        srsi_k_prev = self.prev(df, "stochrsi_k")
        srsi_d = self.latest(df, "stochrsi_d")

        if any(math.isnan(v) for v in [close, atr, adx, srsi_k, srsi_d]):
            return Signal(direction="neutral", strategy=self.name, symbol=symbol, meta={"reason": "nan values"})

        pivot_highs, pivot_lows = _find_pivots(df, lookback=lookback)
        support_levels = _cluster_levels(pivot_lows, tolerance=tolerance, min_touches=min_touches)
        resistance_levels = _cluster_levels(pivot_highs, tolerance=tolerance, min_touches=min_touches)

        near_support = any(abs(close - lvl) / lvl <= tolerance * 2 for lvl in support_levels)
        near_resistance = any(abs(close - lvl) / lvl <= tolerance * 2 for lvl in resistance_levels)

        trending = adx >= adx_threshold
        srsi_oversold = srsi_k < 20 and srsi_k > srsi_k_prev  # turning up from oversold
        srsi_overbought = srsi_k > 80 and srsi_k < srsi_k_prev  # turning down from overbought

        atr_sl = self.cfg.get("risk", {}).get("atr_sl_multiplier", 1.5)
        atr_tp = self.cfg.get("risk", {}).get("atr_tp_multiplier", 3.0)

        if near_support and srsi_oversold:
            # find closest support level for more precise SL
            closest_sup = min(support_levels, key=lambda l: abs(close - l))
            sl = closest_sup - atr * atr_sl
            tp = close + atr * atr_tp
            confidence = 0.6 + (0.2 if trending else 0.0) + (0.2 if srsi_k < 15 else 0.0)
            return Signal(
                direction="long",
                strategy=self.name,
                symbol=symbol,
                confidence=min(confidence, 1.0),
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                meta={
                    "close": close,
                    "support_level": closest_sup,
                    "adx": adx,
                    "srsi_k": srsi_k,
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels,
                },
            )

        if near_resistance and srsi_overbought:
            closest_res = min(resistance_levels, key=lambda l: abs(close - l))
            sl = closest_res + atr * atr_sl
            tp = close - atr * atr_tp
            confidence = 0.6 + (0.2 if trending else 0.0) + (0.2 if srsi_k > 85 else 0.0)
            return Signal(
                direction="short",
                strategy=self.name,
                symbol=symbol,
                confidence=min(confidence, 1.0),
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                meta={
                    "close": close,
                    "resistance_level": closest_res,
                    "adx": adx,
                    "srsi_k": srsi_k,
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels,
                },
            )

        return Signal(
            direction="neutral",
            strategy=self.name,
            symbol=symbol,
            meta={
                "near_support": near_support,
                "near_resistance": near_resistance,
                "trending": trending,
                "srsi_k": srsi_k,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
            },
        )
