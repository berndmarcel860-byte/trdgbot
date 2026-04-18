"""Signal aggregator – combines signals from multiple strategies.

The combined strategy runs every enabled strategy in parallel and applies a
*consensus filter*: an actionable signal is only returned when at least
``min_confluence`` strategies agree on the same direction.

When multiple strategies agree the signal with the highest confidence is
returned, with its ``confidence`` boosted by the agreement ratio so the risk
manager can size the position proportionally.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from bot.strategies.base import BaseStrategy, Signal
from bot.strategies.dca_fibonacci import DCAFibonacciStrategy
from bot.strategies.pullback import PullbackStrategy
from bot.strategies.support_resistance import SupportResistanceStrategy


_STRATEGY_REGISTRY: Dict[str, type] = {
    "pullback": PullbackStrategy,
    "support_resistance": SupportResistanceStrategy,
    "dca_fibonacci": DCAFibonacciStrategy,
}


class CombinedStrategy(BaseStrategy):
    """Meta-strategy that aggregates signals from all enabled strategies."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        enabled: List[str] = cfg.get("strategies", {}).get(
            "enabled", list(_STRATEGY_REGISTRY.keys())
        )
        self._strategies: List[BaseStrategy] = [
            _STRATEGY_REGISTRY[name](cfg)
            for name in enabled
            if name in _STRATEGY_REGISTRY
        ]
        self._min_confluence: int = cfg.get("strategies", {}).get("min_confluence", 1)

    @property
    def name(self) -> str:
        return "combined"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Run all strategies and return a consensus signal.

        Args:
            df: Enriched OHLCV DataFrame.
            symbol: Trading pair symbol.

        Returns:
            A :class:`~bot.strategies.base.Signal` with the aggregated
            direction and boosted confidence, or a ``"neutral"`` signal
            when no consensus is reached.
        """
        signals: List[Signal] = [s.generate_signal(df, symbol) for s in self._strategies]
        actionable = [s for s in signals if s.is_actionable()]

        long_signals = [s for s in actionable if s.direction == "long"]
        short_signals = [s for s in actionable if s.direction == "short"]

        def _best(group: List[Signal]) -> Signal:
            """Return the highest-confidence signal from *group*."""
            return max(group, key=lambda s: s.confidence)

        if len(long_signals) >= self._min_confluence and len(long_signals) >= len(short_signals):
            best = _best(long_signals)
            boost = len(long_signals) / len(self._strategies)
            return Signal(
                direction="long",
                strategy=self.name,
                symbol=symbol,
                confidence=min(best.confidence * (1.0 + boost * 0.3), 1.0),
                entry_price=best.entry_price,
                stop_loss=best.stop_loss,
                take_profit=best.take_profit,
                meta={
                    "agreeing_strategies": [s.strategy for s in long_signals],
                    "all_signals": [
                        {"strategy": s.strategy, "direction": s.direction, "confidence": s.confidence}
                        for s in signals
                    ],
                    "best_signal_meta": best.meta,
                },
            )

        if len(short_signals) >= self._min_confluence and len(short_signals) > len(long_signals):
            best = _best(short_signals)
            boost = len(short_signals) / len(self._strategies)
            return Signal(
                direction="short",
                strategy=self.name,
                symbol=symbol,
                confidence=min(best.confidence * (1.0 + boost * 0.3), 1.0),
                entry_price=best.entry_price,
                stop_loss=best.stop_loss,
                take_profit=best.take_profit,
                meta={
                    "agreeing_strategies": [s.strategy for s in short_signals],
                    "all_signals": [
                        {"strategy": s.strategy, "direction": s.direction, "confidence": s.confidence}
                        for s in signals
                    ],
                    "best_signal_meta": best.meta,
                },
            )

        return Signal(
            direction="neutral",
            strategy=self.name,
            symbol=symbol,
            meta={
                "reason": "no consensus",
                "all_signals": [
                    {"strategy": s.strategy, "direction": s.direction, "confidence": s.confidence}
                    for s in signals
                ],
            },
        )

    def get_individual_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Return raw signals from every individual strategy (for debugging)."""
        return [s.generate_signal(df, symbol) for s in self._strategies]
