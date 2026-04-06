"""Tests for all trading strategies."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from bot.indicators import add_all_indicators, prepare_ohlcv
from bot.strategies.base import Signal
from bot.strategies.combined import CombinedStrategy
from bot.strategies.dca_fibonacci import DCAFibonacciStrategy
from bot.strategies.pullback import PullbackStrategy
from bot.strategies.support_resistance import SupportResistanceStrategy


# ── Helpers ────────────────────────────────────────────────────────────────

_DEFAULT_CFG: Dict[str, Any] = {
    "strategies": {
        "enabled": ["pullback", "support_resistance", "dca_fibonacci"],
        "min_confluence": 1,
        "pullback": {
            "fast_ema": 21,
            "slow_ema": 55,
            "trend_ema": 200,
            "rsi_period": 14,
            "rsi_oversold": 40,
            "rsi_overbought": 60,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        "support_resistance": {
            "pivot_lookback": 10,
            "zone_tolerance": 0.003,
            "min_touches": 2,
            "adx_period": 14,
            "adx_threshold": 20,
        },
        "dca_fibonacci": {
            "fib_levels": [0.382, 0.5, 0.618, 0.786],
            "fib_weights": [0.4, 0.3, 0.2, 0.1],
            "swing_lookback": 50,
            "rsi_filter_long": 50,
            "rsi_filter_short": 50,
        },
    },
    "risk": {
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 3.0,
    },
    "indicators": {
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2,
        "stochrsi_period": 14,
        "stochrsi_smooth_k": 3,
        "stochrsi_smooth_d": 3,
        "volume_ma_period": 20,
    },
}


def _make_ohlcv_list(n: int = 300, trend: str = "up") -> list:
    rng = np.random.default_rng(7)
    t = np.arange(n)
    if trend == "up":
        close = 30_000 + t * 50 + rng.normal(0, 150, n)
    elif trend == "down":
        close = 40_000 - t * 50 + rng.normal(0, 150, n)
    else:
        close = 30_000 + 3_000 * np.sin(t * 0.15) + rng.normal(0, 100, n)
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.001, 0.008, n))
    low = close * (1 - rng.uniform(0.001, 0.008, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.uniform(100, 1000, n)
    ts_start = 1_700_000_000_000
    return [[ts_start + i * 3_600_000, open_[i], high[i], low[i], close[i], volume[i]] for i in range(n)]


@pytest.fixture
def enriched_df_up() -> pd.DataFrame:
    df = prepare_ohlcv(_make_ohlcv_list(300, trend="up"))
    return add_all_indicators(df, _DEFAULT_CFG)


@pytest.fixture
def enriched_df_down() -> pd.DataFrame:
    df = prepare_ohlcv(_make_ohlcv_list(300, trend="down"))
    return add_all_indicators(df, _DEFAULT_CFG)


@pytest.fixture
def enriched_df_sideways() -> pd.DataFrame:
    df = prepare_ohlcv(_make_ohlcv_list(300, trend="sideways"))
    return add_all_indicators(df, _DEFAULT_CFG)


# ── Signal contract ────────────────────────────────────────────────────────

class TestSignalContract:
    def test_is_actionable_long(self):
        s = Signal(direction="long", strategy="test", symbol="BTC/USDT:USDT", confidence=0.8)
        assert s.is_actionable()

    def test_is_actionable_short(self):
        s = Signal(direction="short", strategy="test", symbol="BTC/USDT:USDT")
        assert s.is_actionable()

    def test_is_not_actionable_neutral(self):
        s = Signal(direction="neutral", strategy="test", symbol="BTC/USDT:USDT")
        assert not s.is_actionable()


# ── PullbackStrategy ───────────────────────────────────────────────────────

class TestPullbackStrategy:
    def _strategy(self) -> PullbackStrategy:
        return PullbackStrategy(_DEFAULT_CFG)

    def test_returns_signal_object(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        assert isinstance(result, Signal)

    def test_direction_valid(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        assert result.direction in ("long", "short", "neutral")

    def test_name_is_pullback(self):
        assert self._strategy().name == "pullback"

    def test_empty_df_returns_neutral(self):
        strategy = self._strategy()
        result = strategy.generate_signal(pd.DataFrame(), "BTC/USDT:USDT")
        assert result.direction == "neutral"

    def test_sl_below_entry_for_long(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        if result.direction == "long":
            assert result.stop_loss < result.entry_price

    def test_sl_above_entry_for_short(self, enriched_df_down):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_down, "BTC/USDT:USDT")
        if result.direction == "short":
            assert result.stop_loss > result.entry_price


# ── SupportResistanceStrategy ──────────────────────────────────────────────

class TestSupportResistanceStrategy:
    def _strategy(self) -> SupportResistanceStrategy:
        return SupportResistanceStrategy(_DEFAULT_CFG)

    def test_returns_signal_object(self, enriched_df_sideways):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_sideways, "ETH/USDT:USDT")
        assert isinstance(result, Signal)

    def test_name_is_support_resistance(self):
        assert self._strategy().name == "support_resistance"

    def test_empty_df_returns_neutral(self):
        strategy = self._strategy()
        result = strategy.generate_signal(pd.DataFrame(), "ETH/USDT:USDT")
        assert result.direction == "neutral"

    def test_direction_valid(self, enriched_df_sideways):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_sideways, "ETH/USDT:USDT")
        assert result.direction in ("long", "short", "neutral")

    def test_has_support_levels_in_meta(self, enriched_df_sideways):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_sideways, "ETH/USDT:USDT")
        assert "support_levels" in result.meta or "reason" in result.meta


# ── DCAFibonacciStrategy ───────────────────────────────────────────────────

class TestDCAFibonacciStrategy:
    def _strategy(self) -> DCAFibonacciStrategy:
        return DCAFibonacciStrategy(_DEFAULT_CFG)

    def test_returns_signal_object(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        assert isinstance(result, Signal)

    def test_name_is_dca_fibonacci(self):
        assert self._strategy().name == "dca_fibonacci"

    def test_empty_df_returns_neutral(self):
        strategy = self._strategy()
        result = strategy.generate_signal(pd.DataFrame(), "BTC/USDT:USDT")
        assert result.direction == "neutral"

    def test_dca_layers_in_meta_when_actionable(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        if result.is_actionable():
            assert "dca_layers" in result.meta
            assert "fib_prices" in result.meta

    def test_dca_layers_have_weight(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        if result.is_actionable():
            for layer in result.meta.get("dca_layers", []):
                assert "weight" in layer
                assert "price" in layer


# ── CombinedStrategy ───────────────────────────────────────────────────────

class TestCombinedStrategy:
    def _strategy(self) -> CombinedStrategy:
        return CombinedStrategy(_DEFAULT_CFG)

    def test_returns_signal_object(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        assert isinstance(result, Signal)

    def test_name_is_combined(self):
        assert self._strategy().name == "combined"

    def test_individual_signals_length(self, enriched_df_up):
        strategy = self._strategy()
        signals = strategy.get_individual_signals(enriched_df_up, "BTC/USDT:USDT")
        assert len(signals) == 3  # pullback, support_resistance, dca_fibonacci

    def test_confidence_bounded(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        assert 0.0 <= result.confidence <= 1.0

    def test_all_signals_in_meta(self, enriched_df_up):
        strategy = self._strategy()
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        assert "all_signals" in result.meta

    def test_min_confluence_zero_always_neutral(self, enriched_df_up):
        """With min_confluence=999 and only 3 strategies it should never agree."""
        cfg = dict(_DEFAULT_CFG)
        cfg["strategies"] = dict(cfg["strategies"])
        cfg["strategies"]["min_confluence"] = 999
        strategy = CombinedStrategy(cfg)
        result = strategy.generate_signal(enriched_df_up, "BTC/USDT:USDT")
        assert result.direction == "neutral"
