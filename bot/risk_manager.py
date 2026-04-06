"""Risk management module.

Responsibilities
----------------
* **Position sizing** – computes the correct contract quantity so that the
  maximum loss on a trade equals ``max_risk_per_trade × equity``.
* **Stop-loss / take-profit** – validates and adjusts SL/TP prices returned
  by strategies to enforce the minimum R:R ratio.
* **Drawdown guard** – tracks daily and total equity drawdown and signals
  when trading should be halted.
* **Exposure limit** – enforces ``max_open_positions``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from bot.logger import get_logger
from bot.strategies.base import Signal

logger = get_logger(__name__)


@dataclass
class RiskParameters:
    """Parsed risk configuration values."""

    max_risk_per_trade: float = 0.01
    max_open_positions: int = 4
    max_daily_drawdown: float = 0.05
    max_total_drawdown: float = 0.15
    reward_risk_ratio: float = 2.5
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
    trailing_stop_activation: float = 1.5
    trailing_stop_distance: float = 1.0

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "RiskParameters":
        r = cfg.get("risk", {})
        return cls(
            max_risk_per_trade=r.get("max_risk_per_trade", 0.01),
            max_open_positions=r.get("max_open_positions", 4),
            max_daily_drawdown=r.get("max_daily_drawdown", 0.05),
            max_total_drawdown=r.get("max_total_drawdown", 0.15),
            reward_risk_ratio=r.get("reward_risk_ratio", 2.5),
            atr_sl_multiplier=r.get("atr_sl_multiplier", 1.5),
            atr_tp_multiplier=r.get("atr_tp_multiplier", 3.0),
            trailing_stop_activation=r.get("trailing_stop_activation", 1.5),
            trailing_stop_distance=r.get("trailing_stop_distance", 1.0),
        )


@dataclass
class TradeSetup:
    """Complete validated trade setup returned by the risk manager.

    Attributes:
        symbol: Trading pair.
        direction: ``"long"`` or ``"short"``.
        entry_price: Suggested entry price.
        stop_loss: Validated stop-loss price.
        take_profit: Validated take-profit price.
        quantity: Position size in base currency units.
        risk_amount: Max dollar loss on this trade.
        risk_reward: Actual R:R ratio.
        approved: ``True`` when the setup passes all risk checks.
        rejection_reason: Non-empty string when *approved* is ``False``.
    """

    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_amount: float
    risk_reward: float
    approved: bool = True
    rejection_reason: str = ""


class RiskManager:
    """Stateful risk manager that tracks equity and open positions.

    Args:
        cfg: Full bot configuration dictionary.
        initial_equity: Starting account equity in quote currency (USDT).
    """

    def __init__(self, cfg: Dict[str, Any], initial_equity: float = 10_000.0) -> None:
        self.params = RiskParameters.from_cfg(cfg)
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.daily_start_equity = initial_equity
        self.open_positions: int = 0
        self._halted: bool = False
        logger.info(
            "RiskManager initialised | equity=%.2f | max_risk_per_trade=%.1f%%",
            initial_equity,
            self.params.max_risk_per_trade * 100,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(self, signal: Signal, atr: float, current_price: float) -> TradeSetup:
        """Evaluate a strategy signal and return a :class:`TradeSetup`.

        Args:
            signal: Actionable signal from a strategy.
            atr: Current ATR value for the symbol.
            current_price: Latest close price.

        Returns:
            A :class:`TradeSetup` with *approved=False* if any risk rule is
            violated, otherwise a fully populated setup ready to be executed.
        """
        if self._halted:
            return self._reject(signal, current_price, atr, "trading halted due to drawdown")

        if self.open_positions >= self.params.max_open_positions:
            return self._reject(signal, current_price, atr, "max open positions reached")

        entry = signal.entry_price if signal.entry_price else current_price
        sl, tp = self._compute_sl_tp(signal, entry, atr)

        if sl is None or tp is None:
            return self._reject(signal, current_price, atr, "could not compute SL/TP")

        # Enforce minimum R:R ratio
        risk_dist = abs(entry - sl)
        reward_dist = abs(tp - entry)
        if risk_dist == 0:
            return self._reject(signal, current_price, atr, "zero risk distance (SL == entry)")
        rr = reward_dist / risk_dist
        if rr < self.params.reward_risk_ratio:
            # Adjust TP to meet minimum R:R
            if signal.direction == "long":
                tp = entry + risk_dist * self.params.reward_risk_ratio
            else:
                tp = entry - risk_dist * self.params.reward_risk_ratio
            rr = self.params.reward_risk_ratio
            logger.debug("TP adjusted to enforce %.1f R:R → tp=%.6f", rr, tp)

        quantity = self._position_size(entry, sl)
        if quantity <= 0:
            return self._reject(signal, current_price, atr, "calculated quantity is zero")

        risk_amount = quantity * risk_dist

        setup = TradeSetup(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            quantity=quantity,
            risk_amount=risk_amount,
            risk_reward=rr,
        )
        logger.info(
            "Trade approved | %s %s | entry=%.6f | sl=%.6f | tp=%.6f | qty=%.6f | R:R=%.2f",
            signal.direction.upper(), signal.symbol, entry, sl, tp, quantity, rr,
        )
        return setup

    def update_equity(self, new_equity: float) -> None:
        """Update the tracked equity and check drawdown limits.

        Args:
            new_equity: Latest account equity in quote currency.
        """
        self.equity = new_equity
        daily_dd = (self.daily_start_equity - new_equity) / self.daily_start_equity
        total_dd = (self.initial_equity - new_equity) / self.initial_equity

        if daily_dd >= self.params.max_daily_drawdown:
            logger.warning(
                "Daily drawdown limit hit (%.2f%% ≥ %.2f%%) – halting trading",
                daily_dd * 100, self.params.max_daily_drawdown * 100,
            )
            self._halted = True

        if total_dd >= self.params.max_total_drawdown:
            logger.warning(
                "Total drawdown limit hit (%.2f%% ≥ %.2f%%) – halting trading",
                total_dd * 100, self.params.max_total_drawdown * 100,
            )
            self._halted = True

    def reset_daily(self) -> None:
        """Reset the daily equity baseline (call at the start of each trading day)."""
        self.daily_start_equity = self.equity
        if self._halted:
            logger.info("Daily reset – resuming trading (daily DD counter reset)")
            self._halted = False

    def register_open(self) -> None:
        """Increment the open-position counter."""
        self.open_positions += 1

    def register_close(self) -> None:
        """Decrement the open-position counter."""
        self.open_positions = max(0, self.open_positions - 1)

    @property
    def is_halted(self) -> bool:
        """``True`` when trading is suspended due to drawdown limits."""
        return self._halted

    # ── Internal helpers ───────────────────────────────────────────────────

    def _position_size(self, entry: float, stop_loss: float) -> float:
        """Compute position size so that the risk amount equals the target %.

        Args:
            entry: Entry price.
            stop_loss: Stop-loss price.

        Returns:
            Number of base-currency units to buy/sell.
        """
        risk_amount = self.equity * self.params.max_risk_per_trade
        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit == 0:
            return 0.0
        return risk_amount / risk_per_unit

    def _compute_sl_tp(
        self,
        signal: Signal,
        entry: float,
        atr: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """Return ``(stop_loss, take_profit)`` using signal values or ATR fallback."""
        if signal.stop_loss and not math.isnan(signal.stop_loss):
            sl = signal.stop_loss
        else:
            offset = atr * self.params.atr_sl_multiplier
            sl = entry - offset if signal.direction == "long" else entry + offset

        if signal.take_profit and not math.isnan(signal.take_profit):
            tp = signal.take_profit
        else:
            offset = atr * self.params.atr_tp_multiplier
            tp = entry + offset if signal.direction == "long" else entry - offset

        return sl, tp

    def _reject(self, signal: Signal, entry: float, atr: float, reason: str) -> TradeSetup:
        """Build a rejected :class:`TradeSetup`."""
        sl_offset = atr * self.params.atr_sl_multiplier if not math.isnan(atr) else 0
        tp_offset = atr * self.params.atr_tp_multiplier if not math.isnan(atr) else 0
        logger.debug("Trade rejected (%s): %s %s", reason, signal.direction, signal.symbol)
        return TradeSetup(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry,
            stop_loss=entry - sl_offset if signal.direction == "long" else entry + sl_offset,
            take_profit=entry + tp_offset if signal.direction == "long" else entry - tp_offset,
            quantity=0.0,
            risk_amount=0.0,
            risk_reward=0.0,
            approved=False,
            rejection_reason=reason,
        )

    # ── Trailing stop helper ───────────────────────────────────────────────

    def compute_trailing_stop(
        self,
        direction: str,
        entry: float,
        current_price: float,
        atr: float,
        current_stop: float,
    ) -> float:
        """Return updated trailing stop price.

        The trailing stop only moves in the direction of profit and never
        retracts.

        Args:
            direction: ``"long"`` or ``"short"``.
            entry: Original entry price.
            current_price: Latest price.
            atr: Current ATR value.
            current_stop: Current stop-loss level.

        Returns:
            New stop-loss price (may equal *current_stop* if no update needed).
        """
        activation_dist = atr * self.params.trailing_stop_activation
        trail_dist = atr * self.params.trailing_stop_distance

        if direction == "long":
            if current_price >= entry + activation_dist:
                new_stop = current_price - trail_dist
                return max(new_stop, current_stop)
        else:
            if current_price <= entry - activation_dist:
                new_stop = current_price + trail_dist
                return min(new_stop, current_stop)

        return current_stop
