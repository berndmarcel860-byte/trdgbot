"""Exchange connectivity layer (ccxt wrapper).

Provides a thin, testable abstraction over ccxt so the rest of the bot
never calls ccxt directly.  In *dry-run* mode every mutating call is a
no-op and the function returns a mock response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import ccxt

from bot.logger import get_logger

logger = get_logger(__name__)


class ExchangeClient:
    """Wraps a ccxt futures exchange instance.

    Args:
        cfg: Full bot configuration dictionary (as returned by
             :func:`bot.config.load_config`).
        dry_run: When ``True`` no real orders are placed.
    """

    def __init__(self, cfg: Dict[str, Any], dry_run: bool = True) -> None:
        exchange_cfg = cfg.get("exchange", {})
        exchange_name: str = exchange_cfg.get("name", "bybit")
        self.dry_run = dry_run or cfg.get("bot", {}).get("dry_run", True)
        self.leverage = exchange_cfg.get("leverage", 10)
        self.margin_mode = exchange_cfg.get("margin_mode", "cross")

        exchange_class = getattr(ccxt, exchange_name)
        options: Dict[str, Any] = {"defaultType": "future"}
        if exchange_cfg.get("testnet", True):
            options["testnet"] = True

        self._exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": exchange_cfg.get("api_key", ""),
                "secret": exchange_cfg.get("api_secret", ""),
                "password": exchange_cfg.get("passphrase", ""),
                "enableRateLimit": True,
                "options": options,
            }
        )

        if exchange_cfg.get("testnet", True):
            if hasattr(self._exchange, "set_sandbox_mode"):
                self._exchange.set_sandbox_mode(True)

        logger.info(
            "Exchange client initialised: %s | testnet=%s | dry_run=%s",
            exchange_name,
            exchange_cfg.get("testnet", True),
            self.dry_run,
        )

    # ── Market data ────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> List[List[float]]:
        """Fetch OHLCV candlestick data.

        Args:
            symbol: Market symbol, e.g. ``"BTC/USDT:USDT"``.
            timeframe: Candle timeframe string (``"1m"``, ``"15m"``, ``"1h"``…).
            limit: Number of candles to fetch.

        Returns:
            List of ``[timestamp, open, high, low, close, volume]`` rows.
        """
        logger.debug("Fetching OHLCV %s %s (limit=%d)", symbol, timeframe, limit)
        return self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return the latest ticker for *symbol*."""
        return self._exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> Dict[str, Any]:
        """Return account balance."""
        return self._exchange.fetch_balance()

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return open futures positions."""
        return self._exchange.fetch_positions(symbols)

    # ── Order management ───────────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: Optional[int] = None) -> None:
        """Set leverage for *symbol* on the exchange."""
        lev = leverage or self.leverage
        if self.dry_run:
            logger.info("[DRY-RUN] set_leverage %s × %d", symbol, lev)
            return
        try:
            self._exchange.set_leverage(lev, symbol)
        except ccxt.BaseError as exc:
            logger.warning("set_leverage failed for %s: %s", symbol, exc)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Place a market order.

        Args:
            symbol: Trading pair symbol.
            side: ``"buy"`` or ``"sell"``.
            amount: Order size in base currency.
            params: Extra ccxt params (e.g. ``{"reduceOnly": True}``).

        Returns:
            Order dict (or mock dict in dry-run mode).
        """
        params = params or {}
        if self.dry_run:
            mock: Dict[str, Any] = {
                "id": "DRY_RUN",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "type": "market",
                "status": "closed",
                "params": params,
            }
            logger.info("[DRY-RUN] MARKET %s %s %.6f | params=%s", side.upper(), symbol, amount, params)
            return mock
        return self._exchange.create_market_order(symbol, side, amount, params=params)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Place a limit order."""
        params = params or {}
        if self.dry_run:
            mock = {
                "id": "DRY_RUN",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "type": "limit",
                "status": "open",
                "params": params,
            }
            logger.info(
                "[DRY-RUN] LIMIT %s %s %.6f @ %.6f | params=%s",
                side.upper(), symbol, amount, price, params,
            )
            return mock
        return self._exchange.create_limit_order(symbol, side, amount, price, params=params)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an open order by ID."""
        if self.dry_run:
            logger.info("[DRY-RUN] cancel_order %s on %s", order_id, symbol)
            return {"id": order_id, "status": "canceled"}
        return self._exchange.cancel_order(order_id, symbol)

    def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all open orders for *symbol*."""
        if self.dry_run:
            logger.info("[DRY-RUN] cancel_all_orders for %s", symbol)
            return []
        return self._exchange.cancel_all_orders(symbol)
