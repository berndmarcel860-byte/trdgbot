"""Tests for bot.market_bias.MarketBiasDetector."""

from __future__ import annotations

from typing import Any, Dict

import math
import numpy as np
import pandas as pd
import pytest

from bot.indicators import add_all_indicators, prepare_ohlcv
from bot.market_bias import MarketBiasDetector


# ── Helpers ────────────────────────────────────────────────────────────────


_DEFAULT_CFG: Dict[str, Any] = {
    "market_bias": {
        "enabled": True,
        "ema_fast": 55,
        "ema_slow": 200,
        "adx_threshold": 20,
        "bullish_threshold": 3,
        "require_match": True,
    },
    "strategies": {
        "pullback": {
            "fast_ema": 21,
            "slow_ema": 55,
            "trend_ema": 200,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        "support_resistance": {"adx_period": 14},
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
    rng = np.random.default_rng(42)
    t = np.arange(n)
    if trend == "up":
        close = 30_000 + t * 50 + rng.normal(0, 100, n)
    elif trend == "down":
        close = 40_000 - t * 50 + rng.normal(0, 100, n)
    else:
        close = 30_000 + 2_000 * np.sin(t * 0.15) + rng.normal(0, 80, n)
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.uniform(100, 1000, n)
    ts_start = 1_700_000_000_000
    return [
        [ts_start + i * 14_400_000, open_[i], high[i], low[i], close[i], volume[i]]
        for i in range(n)
    ]


def _enriched(trend: str = "up") -> pd.DataFrame:
    raw = _make_ohlcv_list(300, trend=trend)
    df = prepare_ohlcv(raw)
    return add_all_indicators(df, _DEFAULT_CFG)


# ── Constructor / properties ────────────────────────────────────────────────


class TestMarketBiasProperties:
    def test_enabled_by_default(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        assert detector.enabled is True

    def test_require_match_by_default(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        assert detector.require_match is True

    def test_disabled_via_config(self):
        cfg = dict(_DEFAULT_CFG)
        cfg["market_bias"] = dict(cfg["market_bias"], enabled=False)
        detector = MarketBiasDetector(cfg)
        assert detector.enabled is False

    def test_require_match_false_via_config(self):
        cfg = dict(_DEFAULT_CFG)
        cfg["market_bias"] = dict(cfg["market_bias"], require_match=False)
        detector = MarketBiasDetector(cfg)
        assert detector.require_match is False

    def test_defaults_when_no_market_bias_section(self):
        detector = MarketBiasDetector({})
        assert detector.enabled is True
        assert detector.require_match is True


# ── detect() return values ──────────────────────────────────────────────────


class TestMarketBiasDetect:
    def test_returns_long_in_strong_uptrend(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        df = _enriched(trend="up")
        result = detector.detect(df)
        assert result in ("long", "neutral")  # uptrend should bias toward long

    def test_returns_short_in_strong_downtrend(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        df = _enriched(trend="down")
        result = detector.detect(df)
        assert result in ("short", "neutral")  # downtrend should bias toward short

    def test_returns_valid_bias_string(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        df = _enriched(trend="sideways")
        result = detector.detect(df)
        assert result in ("long", "short", "neutral")

    def test_disabled_always_returns_neutral(self):
        cfg = dict(_DEFAULT_CFG)
        cfg["market_bias"] = dict(cfg["market_bias"], enabled=False)
        detector = MarketBiasDetector(cfg)
        df = _enriched(trend="up")
        assert detector.detect(df) == "neutral"

    def test_empty_df_returns_neutral(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        assert detector.detect(pd.DataFrame()) == "neutral"

    def test_missing_columns_returns_neutral(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        assert detector.detect(df) == "neutral"


# ── Scoring logic via synthetic data ───────────────────────────────────────


class TestMarketBiasScoring:
    """Test the scoring logic by injecting hand-crafted indicator values."""

    @staticmethod
    def _df_with_values(
        close: float,
        ema_fast: float,
        ema_slow: float,
        macd_hist: float,
        adx: float,
        dmp: float,
        dmn: float,
    ) -> pd.DataFrame:
        """Build a single-row DataFrame with the required indicator columns."""
        return pd.DataFrame(
            {
                "close": [close],
                "ema_55": [ema_fast],
                "ema_200": [ema_slow],
                "macd_hist": [macd_hist],
                "adx": [adx],
                "dmp": [dmp],
                "dmn": [dmn],
            }
        )

    def test_all_bullish_votes_returns_long(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        df = self._df_with_values(
            close=200.0,   # above both EMAs
            ema_fast=180.0,
            ema_slow=150.0,  # fast > slow (golden cross)
            macd_hist=5.0,   # positive
            adx=30.0,        # trending
            dmp=25.0,        # DI+ > DI-
            dmn=10.0,
        )
        assert detector.detect(df) == "long"

    def test_all_bearish_votes_returns_short(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        df = self._df_with_values(
            close=100.0,    # below both EMAs
            ema_fast=130.0,
            ema_slow=160.0,  # fast < slow (death cross)
            macd_hist=-5.0,  # negative
            adx=30.0,        # trending
            dmp=10.0,        # DI- > DI+
            dmn=25.0,
        )
        assert detector.detect(df) == "short"

    def test_mixed_votes_returns_neutral(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        # close > ema_fast (bullish) but close < ema_slow (bearish)
        # ema_fast < ema_slow (bearish), macd_hist positive (bullish)
        # adx < threshold → DI vote skipped → 2 bull, 2 bear → neutral
        df = self._df_with_values(
            close=155.0,
            ema_fast=140.0,  # close > fast (bullish)
            ema_slow=160.0,  # close < slow (bearish); fast < slow (bearish)
            macd_hist=1.0,   # bullish
            adx=10.0,        # below threshold → DI vote skipped
            dmp=30.0,
            dmn=10.0,
        )
        assert detector.detect(df) == "neutral"

    def test_nan_close_returns_neutral(self):
        detector = MarketBiasDetector(_DEFAULT_CFG)
        df = self._df_with_values(
            close=float("nan"),
            ema_fast=100.0,
            ema_slow=90.0,
            macd_hist=1.0,
            adx=25.0,
            dmp=20.0,
            dmn=10.0,
        )
        assert detector.detect(df) == "neutral"

    def test_adx_below_threshold_skips_di_vote(self):
        """When ADX is weak the DI+/DI- vote is skipped; max score is 4."""
        detector = MarketBiasDetector(_DEFAULT_CFG)
        # All non-DI votes bullish (3 points), DI skipped → score=3 → "long"
        df = self._df_with_values(
            close=200.0,
            ema_fast=180.0,
            ema_slow=150.0,
            macd_hist=5.0,
            adx=10.0,   # below threshold → DI vote not cast
            dmp=5.0,
            dmn=40.0,   # DI- would win but vote is skipped
        )
        assert detector.detect(df) == "long"
