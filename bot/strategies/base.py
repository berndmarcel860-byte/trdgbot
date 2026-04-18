"""Abstract base class for all trdgbot trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Signal:
    """A trading signal produced by a strategy.

    Attributes:
        direction: ``"long"``, ``"short"``, or ``"neutral"``.
        strategy: Name of the strategy that generated this signal.
        symbol: Trading pair symbol.
        confidence: Score in ``[0.0, 1.0]`` – higher means stronger signal.
        entry_price: Suggested entry price (``None`` = market order).
        stop_loss: Suggested stop-loss price.
        take_profit: Suggested take-profit price.
        meta: Any extra diagnostic data (indicator values, reasons, etc.).
    """

    direction: str  # "long" | "short" | "neutral"
    strategy: str
    symbol: str
    confidence: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_actionable(self) -> bool:
        """Return ``True`` when the signal actually calls for an entry."""
        return self.direction in ("long", "short")


class BaseStrategy(ABC):
    """Base class that every strategy must inherit from.

    Sub-classes only need to implement :meth:`generate_signal`.

    Args:
        cfg: Full bot configuration dictionary.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.strategy_cfg: Dict[str, Any] = cfg.get("strategies", {}).get(self.name, {})

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy (matches config key)."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Analyse the enriched DataFrame and produce a :class:`Signal`.

        Args:
            df: OHLCV DataFrame **already enriched** with all indicators
                (as returned by :func:`bot.indicators.add_all_indicators`).
            symbol: The trading pair being analysed.

        Returns:
            A :class:`Signal` instance.
        """

    # ── Helper utilities ───────────────────────────────────────────────────

    @staticmethod
    def latest(df: pd.DataFrame, col: str) -> float:
        """Return the most recent value of *col* or ``float('nan')``."""
        try:
            return float(df[col].iloc[-1])
        except (KeyError, IndexError):
            return float("nan")

    @staticmethod
    def prev(df: pd.DataFrame, col: str, n: int = 1) -> float:
        """Return the value of *col* from *n* candles ago."""
        try:
            return float(df[col].iloc[-(1 + n)])
        except (KeyError, IndexError):
            return float("nan")
