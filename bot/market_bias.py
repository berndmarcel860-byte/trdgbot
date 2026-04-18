"""Market bias detector – determines the higher-timeframe market direction.

The bias is computed from a Higher Time Frame (HTF) DataFrame that has already
been enriched by :func:`bot.indicators.add_all_indicators`.  It acts as a
pre-filter in :class:`~bot.find_positions.PositionFinder`: when
``require_match`` is ``True`` (default), signals whose direction conflicts with
the HTF bias are discarded before being sent to Telegram.

Bias scoring
------------
Five independent conditions each cast one vote (+1 = bullish, −1 = bearish):

1. Close price above / below the slow EMA (``ema_slow``, default 200).
2. Close price above / below the fast EMA (``ema_fast``, default 55).
3. Fast EMA above / below slow EMA (golden cross / death cross).
4. MACD histogram positive / negative.
5. DI+ above / below DI− **when** ADX ≥ ``adx_threshold`` (trend strength gate).

Final bias
~~~~~~~~~~
* Score ≥  ``bullish_threshold`` (default 3) → ``"long"``
* Score ≤ −``bullish_threshold``              → ``"short"``
* Otherwise                                   → ``"neutral"``

Configuration (``config.yaml`` under ``market_bias:``)
-------------------------------------------------------
.. code-block:: yaml

    market_bias:
      enabled: true
      ema_fast: 55           # must match an EMA period in add_all_indicators
      ema_slow: 200          # must match an EMA period in add_all_indicators
      adx_threshold: 20      # minimum ADX to include the DI+/DI- vote
      bullish_threshold: 3   # minimum absolute score to commit to long/short
      require_match: true    # discard signals that contradict the bias
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from bot.logger import get_logger

logger = get_logger(__name__)


class MarketBiasDetector:
    """Classifies the higher-timeframe market bias.

    Args:
        cfg: Full bot configuration dictionary.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        bias_cfg: Dict[str, Any] = cfg.get("market_bias", {})
        self._enabled: bool = bool(bias_cfg.get("enabled", True))
        self._ema_fast: int = int(bias_cfg.get("ema_fast", 55))
        self._ema_slow: int = int(bias_cfg.get("ema_slow", 200))
        self._adx_threshold: float = float(bias_cfg.get("adx_threshold", 20))
        self._bullish_threshold: int = int(bias_cfg.get("bullish_threshold", 3))
        self._require_match: bool = bool(bias_cfg.get("require_match", True))

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """``True`` when bias detection is active."""
        return self._enabled

    @property
    def require_match(self) -> bool:
        """``True`` when signals conflicting with the bias should be dropped."""
        return self._require_match

    # ── Core logic ───────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> str:
        """Analyse *df* and return the market bias.

        Args:
            df: HTF OHLCV DataFrame enriched with all indicators.  Must
                contain at minimum ``close``, ``ema_<ema_fast>``,
                ``ema_<ema_slow>``, ``macd_hist``, ``adx``, ``dmp``, ``dmn``.

        Returns:
            ``"long"``, ``"short"``, or ``"neutral"``.
        """
        if not self._enabled:
            return "neutral"

        fast_col = f"ema_{self._ema_fast}"
        slow_col = f"ema_{self._ema_slow}"
        required = [fast_col, slow_col, "macd_hist", "adx", "dmp", "dmn", "close"]

        if df.empty or not all(c in df.columns for c in required):
            logger.debug(
                "MarketBiasDetector: missing columns in HTF df – returning neutral"
            )
            return "neutral"

        try:
            close = float(df["close"].iloc[-1])
            ema_fast = float(df[fast_col].iloc[-1])
            ema_slow = float(df[slow_col].iloc[-1])
            macd_hist = float(df["macd_hist"].iloc[-1])
            adx = float(df["adx"].iloc[-1])
            dmp = float(df["dmp"].iloc[-1])
            dmn = float(df["dmn"].iloc[-1])
        except (IndexError, ValueError):
            return "neutral"

        if any(math.isnan(v) for v in [close, ema_fast, ema_slow, macd_hist, adx]):
            return "neutral"

        score = 0

        # Vote 1 – price vs slow EMA
        score += 1 if close > ema_slow else -1
        # Vote 2 – price vs fast EMA
        score += 1 if close > ema_fast else -1
        # Vote 3 – EMA cross (golden / death)
        score += 1 if ema_fast > ema_slow else -1
        # Vote 4 – MACD histogram direction
        score += 1 if macd_hist > 0 else -1
        # Vote 5 – DI+/DI- (only when ADX confirms a trend)
        if not (math.isnan(dmp) or math.isnan(dmn)) and adx >= self._adx_threshold:
            score += 1 if dmp > dmn else -1

        if score >= self._bullish_threshold:
            bias = "long"
        elif score <= -self._bullish_threshold:
            bias = "short"
        else:
            bias = "neutral"

        logger.debug(
            "MarketBias | score=%+d → %s"
            " | close=%.4f ema_fast=%.4f ema_slow=%.4f"
            " | macd_hist=%.4f adx=%.2f dmp=%.2f dmn=%.2f",
            score,
            bias,
            close,
            ema_fast,
            ema_slow,
            macd_hist,
            adx,
            dmp,
            dmn,
        )
        return bias
