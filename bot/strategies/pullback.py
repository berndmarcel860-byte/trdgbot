"""Pullback-to-EMA strategy.

Entry logic
-----------
**Long**
  1. Price is above the 200-period trend EMA  → uptrend confirmed.
  2. Price has pulled back to (or just below) the fast EMA (21).
  3. RSI is below *rsi_oversold* threshold  → momentum oversold on pullback.
  4. MACD histogram is turning up (previous bar < 0, current bar > previous).
  5. Volume on the signal candle is above the 20-period average.

**Short** (mirror conditions)
  1. Price is below trend EMA.
  2. Price has bounced up to (or just above) the fast EMA.
  3. RSI is above *rsi_overbought*.
  4. MACD histogram is turning down.
  5. Volume confirmation.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from bot.strategies.base import BaseStrategy, Signal


class PullbackStrategy(BaseStrategy):
    """EMA pullback strategy with RSI, MACD, and volume filters."""

    @property
    def name(self) -> str:
        return "pullback"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        cfg = self.strategy_cfg
        fast_ema = cfg.get("fast_ema", 21)
        trend_ema = cfg.get("trend_ema", 200)
        rsi_oversold = cfg.get("rsi_oversold", 40)
        rsi_overbought = cfg.get("rsi_overbought", 60)

        ema_fast_col = f"ema_{fast_ema}"
        ema_trend_col = f"ema_{trend_ema}"

        required = [ema_fast_col, ema_trend_col, "rsi", "macd_hist", "close", "atr", "vol_ratio"]
        if df.empty or not all(c in df.columns for c in required):
            return Signal(direction="neutral", strategy=self.name, symbol=symbol, meta={"reason": "missing columns"})

        close = self.latest(df, "close")
        ema_fast = self.latest(df, ema_fast_col)
        ema_trend = self.latest(df, ema_trend_col)
        rsi = self.latest(df, "rsi")
        macd_hist = self.latest(df, "macd_hist")
        macd_hist_prev = self.prev(df, "macd_hist")
        atr = self.latest(df, "atr")
        vol_ratio = self.latest(df, "vol_ratio")

        if any(math.isnan(v) for v in [close, ema_fast, ema_trend, rsi, macd_hist, atr]):
            return Signal(direction="neutral", strategy=self.name, symbol=symbol, meta={"reason": "nan values"})

        # ── Long conditions ───────────────────────────────────────────────
        long_trend = close > ema_trend
        long_pullback = close <= ema_fast * 1.005  # within 0.5 % above fast EMA
        long_rsi = rsi < rsi_oversold
        long_macd = macd_hist > macd_hist_prev  # histogram turning up
        long_volume = vol_ratio >= 1.0

        # ── Short conditions ──────────────────────────────────────────────
        short_trend = close < ema_trend
        short_pullback = close >= ema_fast * 0.995
        short_rsi = rsi > rsi_overbought
        short_macd = macd_hist < macd_hist_prev
        short_volume = vol_ratio >= 1.0

        confidence_long = sum([long_trend, long_pullback, long_rsi, long_macd, long_volume]) / 5
        confidence_short = sum([short_trend, short_pullback, short_rsi, short_macd, short_volume]) / 5

        if confidence_long >= 0.6:
            entry = close
            sl = entry - atr * self.cfg.get("risk", {}).get("atr_sl_multiplier", 1.5)
            tp = entry + atr * self.cfg.get("risk", {}).get("atr_tp_multiplier", 3.0)
            return Signal(
                direction="long",
                strategy=self.name,
                symbol=symbol,
                confidence=confidence_long,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                meta={
                    "close": close,
                    "ema_fast": ema_fast,
                    "ema_trend": ema_trend,
                    "rsi": rsi,
                    "macd_hist": macd_hist,
                    "vol_ratio": vol_ratio,
                },
            )

        if confidence_short >= 0.6:
            entry = close
            sl = entry + atr * self.cfg.get("risk", {}).get("atr_sl_multiplier", 1.5)
            tp = entry - atr * self.cfg.get("risk", {}).get("atr_tp_multiplier", 3.0)
            return Signal(
                direction="short",
                strategy=self.name,
                symbol=symbol,
                confidence=confidence_short,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                meta={
                    "close": close,
                    "ema_fast": ema_fast,
                    "ema_trend": ema_trend,
                    "rsi": rsi,
                    "macd_hist": macd_hist,
                    "vol_ratio": vol_ratio,
                },
            )

        return Signal(
            direction="neutral",
            strategy=self.name,
            symbol=symbol,
            meta={
                "confidence_long": confidence_long,
                "confidence_short": confidence_short,
            },
        )
