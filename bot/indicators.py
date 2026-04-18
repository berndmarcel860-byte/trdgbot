"""Technical indicators used by trdgbot strategies.

All functions accept a :class:`pandas.DataFrame` with OHLCV columns
(``open``, ``high``, ``low``, ``close``, ``volume``) and return a new
DataFrame with the indicator columns appended.  The original DataFrame
is **never** mutated.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta


# ── OHLCV column normalisation ─────────────────────────────────────────────

def prepare_ohlcv(raw: list[list[float]]) -> pd.DataFrame:
    """Convert a ccxt OHLCV list to a properly typed DataFrame.

    Args:
        raw: List of ``[timestamp_ms, open, high, low, close, volume]`` rows
             as returned by :meth:`bot.exchange.ExchangeClient.fetch_ohlcv`.

    Returns:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` and a UTC :class:`~pandas.DatetimeIndex`.
    """
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df


# ── Moving averages ────────────────────────────────────────────────────────

def add_ema(df: pd.DataFrame, periods: tuple[int, ...] = (21, 55, 200)) -> pd.DataFrame:
    """Add Exponential Moving Average columns.

    Args:
        df: OHLCV DataFrame.
        periods: EMA period values to compute.

    Returns:
        Copy of *df* with ``ema_<period>`` columns added.
    """
    out = df.copy()
    for p in periods:
        out[f"ema_{p}"] = ta.ema(out["close"], length=p)
    return out


# ── Momentum ───────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index column (``rsi``)."""
    out = df.copy()
    out["rsi"] = ta.rsi(out["close"], length=period)
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD columns (``macd``, ``macd_signal``, ``macd_hist``)."""
    out = df.copy()
    macd_df = ta.macd(out["close"], fast=fast, slow=slow, signal=signal)
    if macd_df is not None and not macd_df.empty:
        out["macd"] = macd_df.iloc[:, 0]
        out["macd_signal"] = macd_df.iloc[:, 2]
        out["macd_hist"] = macd_df.iloc[:, 1]
    return out


def add_stochrsi(
    df: pd.DataFrame,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Add Stochastic RSI columns (``stochrsi_k``, ``stochrsi_d``)."""
    out = df.copy()
    stoch = ta.stochrsi(out["close"], length=period, rsi_length=period, k=smooth_k, d=smooth_d)
    if stoch is not None and not stoch.empty:
        out["stochrsi_k"] = stoch.iloc[:, 0]
        out["stochrsi_d"] = stoch.iloc[:, 1]
    return out


# ── Volatility ─────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range column (``atr``)."""
    out = df.copy()
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=period)
    return out


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Band columns (``bb_upper``, ``bb_mid``, ``bb_lower``, ``bb_width``)."""
    out = df.copy()
    bb = ta.bbands(out["close"], length=period, std=std)
    if bb is not None and not bb.empty:
        out["bb_lower"] = bb.iloc[:, 0]
        out["bb_mid"] = bb.iloc[:, 1]
        out["bb_upper"] = bb.iloc[:, 2]
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
    return out


# ── Trend ──────────────────────────────────────────────────────────────────

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX / DI columns (``adx``, ``dmp``, ``dmn``)."""
    out = df.copy()
    adx_df = ta.adx(out["high"], out["low"], out["close"], length=period)
    if adx_df is not None and not adx_df.empty:
        out["adx"] = adx_df.iloc[:, 0]
        out["dmp"] = adx_df.iloc[:, 1]
        out["dmn"] = adx_df.iloc[:, 2]
    return out


# ── Volume ─────────────────────────────────────────────────────────────────

def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume moving average column (``vol_ma``) and ratio (``vol_ratio``)."""
    out = df.copy()
    out["vol_ma"] = ta.sma(out["volume"], length=period)
    out["vol_ratio"] = out["volume"] / out["vol_ma"]
    return out


# ── Convenience: add all indicators at once ────────────────────────────────

def add_all_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add every indicator required by all strategies.

    Args:
        df: Raw OHLCV DataFrame.
        cfg: Full bot configuration dictionary.

    Returns:
        Enriched DataFrame with all indicator columns.
    """
    ind_cfg = cfg.get("indicators", {})
    strat_cfg = cfg.get("strategies", {})

    pb = strat_cfg.get("pullback", {})
    fast_ema = pb.get("fast_ema", 21)
    slow_ema = pb.get("slow_ema", 55)
    trend_ema = pb.get("trend_ema", 200)

    out = add_ema(df, periods=(fast_ema, slow_ema, trend_ema))
    out = add_rsi(out, period=pb.get("rsi_period", 14))
    out = add_macd(
        out,
        fast=pb.get("macd_fast", 12),
        slow=pb.get("macd_slow", 26),
        signal=pb.get("macd_signal", 9),
    )
    out = add_atr(out, period=ind_cfg.get("atr_period", 14))
    out = add_bollinger_bands(
        out,
        period=ind_cfg.get("bb_period", 20),
        std=ind_cfg.get("bb_std", 2.0),
    )
    out = add_adx(out, period=strat_cfg.get("support_resistance", {}).get("adx_period", 14))
    out = add_stochrsi(
        out,
        period=ind_cfg.get("stochrsi_period", 14),
        smooth_k=ind_cfg.get("stochrsi_smooth_k", 3),
        smooth_d=ind_cfg.get("stochrsi_smooth_d", 3),
    )
    out = add_volume_ma(out, period=ind_cfg.get("volume_ma_period", 20))
    return out


# ── Fibonacci utilities ────────────────────────────────────────────────────

def fibonacci_levels(
    swing_high: float,
    swing_low: float,
    levels: tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
) -> Dict[float, float]:
    """Compute Fibonacci retracement prices between *swing_low* and *swing_high*.

    Args:
        swing_high: Highest price of the measured swing.
        swing_low: Lowest price of the measured swing.
        levels: Fibonacci ratios to compute.

    Returns:
        Mapping of ``{ratio: price}`` for each retracement level.
    """
    rng = swing_high - swing_low
    return {lvl: swing_high - lvl * rng for lvl in levels}


def find_swing_high_low(
    df: pd.DataFrame,
    lookback: int = 50,
) -> tuple[float, float]:
    """Find the dominant swing high and swing low within *lookback* candles.

    Args:
        df: OHLCV DataFrame (most recent candle last).
        lookback: Number of recent candles to inspect.

    Returns:
        ``(swing_high, swing_low)`` tuple.
    """
    window = df.tail(lookback)
    return float(window["high"].max()), float(window["low"].min())
