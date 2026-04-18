"""Order manager – translates trade setups into exchange orders.

Responsibilities
----------------
* Place entry market or limit orders.
* Place stop-loss and take-profit orders immediately after entry.
* Place DCA limit orders for the remaining Fibonacci layers (when the signal
  comes from :class:`~bot.strategies.dca_fibonacci.DCAFibonacciStrategy`).
* Update trailing stops on each tick.
* Track open orders and positions in memory (in-process state only – a
  real implementation would persist this to a database).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bot.exchange import ExchangeClient
from bot.logger import get_logger
from bot.risk_manager import RiskManager, TradeSetup

logger = get_logger(__name__)


@dataclass
class ManagedPosition:
    """Runtime state of a single open position.

    Attributes:
        symbol: Trading pair.
        direction: ``"long"`` or ``"short"``.
        entry_price: Actual fill price.
        stop_loss: Current stop-loss price (may be updated by trailing logic).
        take_profit: Target take-profit price.
        quantity: Total position size in base currency.
        entry_order_id: Exchange order ID for the entry order.
        sl_order_id: Exchange order ID for the stop-loss order (if placed).
        tp_order_id: Exchange order ID for the take-profit order (if placed).
        dca_order_ids: Exchange order IDs for the DCA layer orders.
    """

    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""
    dca_order_ids: List[str] = field(default_factory=list)


class OrderManager:
    """Manages order lifecycle for all open positions.

    Args:
        exchange: :class:`~bot.exchange.ExchangeClient` instance.
        risk_manager: :class:`~bot.risk_manager.RiskManager` instance.
        cfg: Full bot configuration dictionary.
    """

    def __init__(
        self,
        exchange: ExchangeClient,
        risk_manager: RiskManager,
        cfg: Dict[str, Any],
    ) -> None:
        self._exchange = exchange
        self._risk = risk_manager
        self._cfg = cfg
        self._positions: Dict[str, ManagedPosition] = {}

    # ── Position entry ─────────────────────────────────────────────────────

    def open_position(self, setup: TradeSetup, signal_meta: Optional[Dict[str, Any]] = None) -> Optional[ManagedPosition]:
        """Open a new position from a validated :class:`~bot.risk_manager.TradeSetup`.

        Sequence of orders placed:
        1. Market entry order.
        2. Stop-loss limit order (reduce-only).
        3. Take-profit limit order (reduce-only).
        4. DCA layer limit orders (if ``signal_meta["dca_layers"]`` is present).

        Args:
            setup: Validated trade setup from the risk manager.
            signal_meta: Optional ``meta`` dict from the generating signal (used
                         to extract DCA layer information).

        Returns:
            A :class:`ManagedPosition` instance, or ``None`` if entry failed.
        """
        if not setup.approved:
            logger.warning("Attempted to open a rejected setup: %s", setup.rejection_reason)
            return None

        if setup.symbol in self._positions:
            logger.info("Position already open for %s – skipping", setup.symbol)
            return None

        symbol = setup.symbol
        side = "buy" if setup.direction == "long" else "sell"
        close_side = "sell" if setup.direction == "long" else "buy"

        # ── 1. Entry ───────────────────────────────────────────────────────
        self._exchange.set_leverage(symbol)
        entry_order = self._exchange.place_market_order(symbol, side, setup.quantity)
        entry_id = entry_order.get("id", "")
        actual_entry = entry_order.get("average", entry_order.get("price", setup.entry_price)) or setup.entry_price

        # ── 2. Stop-loss ───────────────────────────────────────────────────
        sl_order = self._exchange.place_limit_order(
            symbol, close_side, setup.quantity, setup.stop_loss,
            params={"reduceOnly": True, "stopLoss": True},
        )
        sl_id = sl_order.get("id", "")

        # ── 3. Take-profit ─────────────────────────────────────────────────
        tp_order = self._exchange.place_limit_order(
            symbol, close_side, setup.quantity, setup.take_profit,
            params={"reduceOnly": True, "takeProfit": True},
        )
        tp_id = tp_order.get("id", "")

        # ── 4. DCA layers ──────────────────────────────────────────────────
        dca_ids: List[str] = []
        dca_layers = (signal_meta or {}).get("dca_layers", [])
        # Skip the first layer (already entered at market)
        for layer in dca_layers[1:]:
            layer_price = layer.get("price")
            layer_weight = layer.get("weight", 0.1)
            layer_qty = setup.quantity * layer_weight
            if layer_price and layer_qty > 0:
                dca_order = self._exchange.place_limit_order(
                    symbol, side, layer_qty, layer_price,
                    params={"reduceOnly": False},
                )
                dca_ids.append(dca_order.get("id", ""))

        pos = ManagedPosition(
            symbol=symbol,
            direction=setup.direction,
            entry_price=float(actual_entry),
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            quantity=setup.quantity,
            entry_order_id=entry_id,
            sl_order_id=sl_id,
            tp_order_id=tp_id,
            dca_order_ids=dca_ids,
        )
        self._positions[symbol] = pos
        self._risk.register_open()

        logger.info(
            "Position opened | %s %s | entry=%.6f | sl=%.6f | tp=%.6f | qty=%.6f",
            setup.direction.upper(), symbol, float(actual_entry),
            setup.stop_loss, setup.take_profit, setup.quantity,
        )
        return pos

    # ── Trailing stop update ───────────────────────────────────────────────

    def update_trailing_stops(self, symbol: str, current_price: float, atr: float) -> None:
        """Update the trailing stop for *symbol* if the position qualifies.

        Args:
            symbol: Trading pair.
            current_price: Latest price.
            atr: Current ATR value.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return

        new_sl = self._risk.compute_trailing_stop(
            direction=pos.direction,
            entry=pos.entry_price,
            current_price=current_price,
            atr=atr,
            current_stop=pos.stop_loss,
        )
        if new_sl != pos.stop_loss:
            logger.info(
                "Trailing stop updated for %s: %.6f → %.6f",
                symbol, pos.stop_loss, new_sl,
            )
            # Cancel old SL and place new one
            if pos.sl_order_id:
                self._exchange.cancel_order(pos.sl_order_id, symbol)
            close_side = "sell" if pos.direction == "long" else "buy"
            sl_order = self._exchange.place_limit_order(
                symbol, close_side, pos.quantity, new_sl,
                params={"reduceOnly": True, "stopLoss": True},
            )
            pos.stop_loss = new_sl
            pos.sl_order_id = sl_order.get("id", "")

    # ── Position close ─────────────────────────────────────────────────────

    def close_position(self, symbol: str, reason: str = "manual") -> None:
        """Close the open position for *symbol* at market.

        Args:
            symbol: Trading pair.
            reason: Human-readable reason for closing (logged only).
        """
        pos = self._positions.get(symbol)
        if pos is None:
            logger.debug("close_position called for %s but no position tracked", symbol)
            return

        # Cancel remaining open orders for this symbol
        self._exchange.cancel_all_orders(symbol)

        close_side = "sell" if pos.direction == "long" else "buy"
        self._exchange.place_market_order(
            symbol, close_side, pos.quantity, params={"reduceOnly": True}
        )

        self._risk.register_close()
        del self._positions[symbol]
        logger.info("Position closed | %s | reason=%s", symbol, reason)

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def open_symbols(self) -> List[str]:
        """List of symbols with open managed positions."""
        return list(self._positions.keys())

    def get_position(self, symbol: str) -> Optional[ManagedPosition]:
        """Return the :class:`ManagedPosition` for *symbol*, or ``None``."""
        return self._positions.get(symbol)
