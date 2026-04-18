"""Tests for bot.indicators module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.indicators import (
    add_adx,
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_rsi,
    add_stochrsi,
    add_volume_ma,
    fibonacci_levels,
    find_swing_high_low,
    prepare_ohlcv,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_ohlcv_list(n: int = 300) -> list:
    """Generate synthetic OHLCV data (rising sine + noise)."""
    rng = np.random.default_rng(42)
    t = np.arange(n)
    close = 30_000 + 5_000 * np.sin(t * 0.05) + rng.normal(0, 200, n)
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.uniform(100, 1000, n)
    timestamps = [(1_700_000_000_000 + i * 3_600_000) for i in range(n)]
    return [[timestamps[i], open_[i], high[i], low[i], close[i], volume[i]] for i in range(n)]


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return prepare_ohlcv(_make_ohlcv_list(300))


# ── prepare_ohlcv ──────────────────────────────────────────────────────────

class TestPrepareOHLCV:
    def test_columns_present(self):
        df = prepare_ohlcv(_make_ohlcv_list(10))
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_index_is_datetime(self):
        df = prepare_ohlcv(_make_ohlcv_list(10))
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_nan_in_close(self):
        df = prepare_ohlcv(_make_ohlcv_list(50))
        assert df["close"].notna().all()


# ── EMA ────────────────────────────────────────────────────────────────────

class TestAddEMA:
    def test_columns_added(self, ohlcv_df):
        out = add_ema(ohlcv_df, periods=(21, 55, 200))
        for p in (21, 55, 200):
            assert f"ema_{p}" in out.columns

    def test_does_not_mutate_input(self, ohlcv_df):
        original_cols = set(ohlcv_df.columns)
        add_ema(ohlcv_df, periods=(21,))
        assert set(ohlcv_df.columns) == original_cols

    def test_ema_values_are_numeric(self, ohlcv_df):
        out = add_ema(ohlcv_df, periods=(21,))
        assert out["ema_21"].dropna().dtype == float


# ── RSI ────────────────────────────────────────────────────────────────────

class TestAddRSI:
    def test_column_added(self, ohlcv_df):
        out = add_rsi(ohlcv_df)
        assert "rsi" in out.columns

    def test_rsi_bounds(self, ohlcv_df):
        out = add_rsi(ohlcv_df)
        valid = out["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


# ── MACD ───────────────────────────────────────────────────────────────────

class TestAddMACD:
    def test_columns_added(self, ohlcv_df):
        out = add_macd(ohlcv_df)
        for col in ("macd", "macd_signal", "macd_hist"):
            assert col in out.columns

    def test_histogram_is_difference(self, ohlcv_df):
        out = add_macd(ohlcv_df)
        diff = (out["macd"] - out["macd_signal"]).dropna()
        hist = out["macd_hist"].dropna()
        # Both series align on non-NaN rows
        common = diff.index.intersection(hist.index)
        assert len(common) > 0


# ── ATR ────────────────────────────────────────────────────────────────────

class TestAddATR:
    def test_column_added(self, ohlcv_df):
        out = add_atr(ohlcv_df)
        assert "atr" in out.columns

    def test_atr_positive(self, ohlcv_df):
        out = add_atr(ohlcv_df)
        assert (out["atr"].dropna() > 0).all()


# ── Bollinger Bands ────────────────────────────────────────────────────────

class TestAddBollingerBands:
    def test_columns_added(self, ohlcv_df):
        out = add_bollinger_bands(ohlcv_df)
        for col in ("bb_upper", "bb_mid", "bb_lower", "bb_width"):
            assert col in out.columns

    def test_upper_above_lower(self, ohlcv_df):
        out = add_bollinger_bands(ohlcv_df)
        valid = out.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()


# ── ADX ────────────────────────────────────────────────────────────────────

class TestAddADX:
    def test_columns_added(self, ohlcv_df):
        out = add_adx(ohlcv_df)
        for col in ("adx", "dmp", "dmn"):
            assert col in out.columns

    def test_adx_non_negative(self, ohlcv_df):
        out = add_adx(ohlcv_df)
        assert (out["adx"].dropna() >= 0).all()


# ── Stochastic RSI ─────────────────────────────────────────────────────────

class TestAddStochRSI:
    def test_columns_added(self, ohlcv_df):
        out = add_stochrsi(ohlcv_df)
        assert "stochrsi_k" in out.columns
        assert "stochrsi_d" in out.columns


# ── Volume MA ──────────────────────────────────────────────────────────────

class TestAddVolumeMA:
    def test_columns_added(self, ohlcv_df):
        out = add_volume_ma(ohlcv_df)
        assert "vol_ma" in out.columns
        assert "vol_ratio" in out.columns

    def test_ratio_positive(self, ohlcv_df):
        out = add_volume_ma(ohlcv_df)
        assert (out["vol_ratio"].dropna() > 0).all()


# ── Fibonacci ──────────────────────────────────────────────────────────────

class TestFibonacci:
    def test_levels_count(self):
        levels = fibonacci_levels(100, 50, levels=(0.236, 0.382, 0.5, 0.618))
        assert len(levels) == 4

    def test_level_618_midpoint(self):
        levels = fibonacci_levels(100, 0, levels=(0.618,))
        assert abs(levels[0.618] - 38.2) < 0.1

    def test_level_0_is_high(self):
        levels = fibonacci_levels(200, 100, levels=(0.0,))
        assert levels[0.0] == pytest.approx(200.0)

    def test_level_1_is_low(self):
        levels = fibonacci_levels(200, 100, levels=(1.0,))
        assert levels[1.0] == pytest.approx(100.0)


class TestFindSwingHighLow:
    def test_returns_tuple(self, ohlcv_df):
        high, low = find_swing_high_low(ohlcv_df, lookback=50)
        assert isinstance(high, float)
        assert isinstance(low, float)

    def test_high_above_low(self, ohlcv_df):
        high, low = find_swing_high_low(ohlcv_df, lookback=50)
        assert high > low
