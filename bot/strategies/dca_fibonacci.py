"""DCA (Dollar-Cost Averaging) strategy using Fibonacci retracement levels.

Concept
-------
After identifying the most recent significant swing (high to low for a
retracement long, low to high for a retracement short), Fibonacci levels
are computed.  The strategy issues **layered limit-order entries** at each
configured Fibonacci level, with position size weights that allocate more
capital to the deeper (higher confidence) levels.

Signal
------
The strategy returns the *first* (shallowest) Fibonacci level as the primary
signal entry.  The order manager is responsible for placing the remaining DCA
layers using the ``meta["dca_layers"]`` list.

Long entry criteria
-------------------
* RSI ≥ ``rsi_filter_long`` (momentum not deeply oversold at the HTF level –
  we want a healthy correction, not a crash).
* Price has retraced at least to the 0.382 Fibonacci level from a swing high.
* Current candle close > previous candle close (micro-reversal).

Short entry criteria (mirror)
------------------------------
* RSI ≤ ``rsi_filter_short``.
* Price has bounced at least to the 0.382 Fibonacci level from a swing low.
* Current close < previous close.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

from bot.indicators import fibonacci_levels, find_swing_high_low
from bot.strategies.base import BaseStrategy, Signal


class DCAFibonacciStrategy(BaseStrategy):
    """Multi-layer DCA entries at Fibonacci retracement levels."""

    @property
    def name(self) -> str:
        return "dca_fibonacci"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        cfg = self.strategy_cfg
        fib_levels_cfg: List[float] = cfg.get("fib_levels", [0.382, 0.5, 0.618, 0.786])
        fib_weights: List[float] = cfg.get("fib_weights", [0.4, 0.3, 0.2, 0.1])
        swing_lookback = cfg.get("swing_lookback", 50)
        rsi_filter_long = cfg.get("rsi_filter_long", 50)
        rsi_filter_short = cfg.get("rsi_filter_short", 50)

        required = ["close", "rsi", "atr"]
        if df.empty or len(df) < swing_lookback or not all(c in df.columns for c in required):
            return Signal(
                direction="neutral", strategy=self.name, symbol=symbol,
                meta={"reason": "insufficient data"},
            )

        close = self.latest(df, "close")
        close_prev = self.prev(df, "close")
        rsi = self.latest(df, "rsi")
        atr = self.latest(df, "atr")

        if any(math.isnan(v) for v in [close, rsi, atr]):
            return Signal(direction="neutral", strategy=self.name, symbol=symbol, meta={"reason": "nan values"})

        swing_high, swing_low = find_swing_high_low(df, lookback=swing_lookback)
        fib_prices = fibonacci_levels(swing_high, swing_low, levels=tuple(fib_levels_cfg))

        atr_sl = self.cfg.get("risk", {}).get("atr_sl_multiplier", 1.5)
        atr_tp = self.cfg.get("risk", {}).get("atr_tp_multiplier", 3.0)

        # ── Long: price retracing down from swing high ─────────────────────
        # Check if current price is AT or BELOW the 0.618 retracement level
        # (i.e. a meaningful pullback has occurred)
        level_618 = fib_prices.get(0.618, swing_low)
        price_retraced_long = close <= fib_prices.get(0.382, swing_high)
        rsi_ok_long = rsi >= rsi_filter_long
        micro_reversal_long = close > close_prev

        level_382 = fib_prices.get(0.382, swing_low)
        price_retraced_short = close >= level_382
        rsi_ok_short = rsi <= rsi_filter_short
        micro_reversal_short = close < close_prev

        if price_retraced_long and rsi_ok_long and micro_reversal_long:
            # Build DCA layers at each Fibonacci level below current price
            dca_layers = [
                {
                    "level": lvl,
                    "price": fib_prices[lvl],
                    "weight": fib_weights[i] if i < len(fib_weights) else 0.1,
                }
                for i, lvl in enumerate(fib_levels_cfg)
                if fib_prices[lvl] <= close
            ]
            if not dca_layers:
                return Signal(direction="neutral", strategy=self.name, symbol=symbol,
                              meta={"reason": "no DCA layers below current price"})

            entry = dca_layers[0]["price"]
            sl = swing_low - atr * atr_sl
            tp = close + (close - swing_low) * atr_tp / atr_sl  # scale TP with swing range
            confidence = min(0.5 + len(dca_layers) * 0.1, 1.0)

            return Signal(
                direction="long",
                strategy=self.name,
                symbol=symbol,
                confidence=confidence,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                meta={
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "fib_prices": fib_prices,
                    "dca_layers": dca_layers,
                    "rsi": rsi,
                    "close": close,
                },
            )

        if price_retraced_short and rsi_ok_short and micro_reversal_short:
            dca_layers = [
                {
                    "level": lvl,
                    "price": fib_prices[lvl],
                    "weight": fib_weights[i] if i < len(fib_weights) else 0.1,
                }
                for i, lvl in enumerate(fib_levels_cfg)
                if fib_prices[lvl] >= close
            ]
            if not dca_layers:
                return Signal(direction="neutral", strategy=self.name, symbol=symbol,
                              meta={"reason": "no DCA layers above current price"})

            entry = dca_layers[0]["price"]
            sl = swing_high + atr * atr_sl
            tp = close - (swing_high - close) * atr_tp / atr_sl
            confidence = min(0.5 + len(dca_layers) * 0.1, 1.0)

            return Signal(
                direction="short",
                strategy=self.name,
                symbol=symbol,
                confidence=confidence,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                meta={
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "fib_prices": fib_prices,
                    "dca_layers": dca_layers,
                    "rsi": rsi,
                    "close": close,
                },
            )

        return Signal(
            direction="neutral",
            strategy=self.name,
            symbol=symbol,
            meta={
                "close": close,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "rsi": rsi,
                "price_retraced_long": price_retraced_long,
                "price_retraced_short": price_retraced_short,
            },
        )
