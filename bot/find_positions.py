"""Position finder – scans top coins and sends trade signals to Telegram.

Usage
-----
.. code-block:: bash

    python main.py --find-positions

Behaviour
---------
* Fetches the top ``find_positions.max_coins`` (default 20) USDT perpetual
  symbols from the exchange, ranked by 24-hour quote volume.
* Runs the :class:`~bot.strategies.combined.CombinedStrategy` on each symbol.
* When an actionable signal is found, formats a full trade card (coin, direction,
  ×20 cross leverage, DCA entry levels, TP, SL, profit/loss percentages) and
  sends it to the configured Telegram channel.
* **Tracks every active signal** in memory.  On each monitoring cycle it checks
  the live price against entry levels, the take-profit, and the stop-loss and
  sends a *reply* to the original message when any of those prices is touched.
* A coin is **blocked from re-signalling** until its stop-loss is hit.  Only
  then is the coin eligible for a new trade card.
"""

from __future__ import annotations

import signal as _signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bot.exchange import ExchangeClient
from bot.indicators import add_all_indicators, prepare_ohlcv
from bot.logger import get_logger
from bot.risk_manager import RiskManager, TradeSetup
from bot.strategies.base import Signal
from bot.strategies.combined import CombinedStrategy
from bot.telegram_notifier import TelegramNotifier

logger = get_logger(__name__)

_RUNNING = True


def _stop_handler(signum: int, frame: Any) -> None:
    global _RUNNING
    logger.info("Shutdown signal received – stopping find_positions…")
    _RUNNING = False


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class EntryLevel:
    """A single DCA entry price level.

    Attributes:
        price: Target entry price.
        weight: Fraction of position allocated here (0 – 1).
        hit: ``True`` once the live price has crossed this level.
    """

    price: float
    weight: float
    hit: bool = False


@dataclass
class ActiveSignal:
    """Runtime state of a signal that has been sent to Telegram.

    Attributes:
        symbol: Trading pair.
        setup: The validated :class:`~bot.risk_manager.TradeSetup`.
        entries: Ordered list of DCA entry levels.
        message_id: Telegram ``message_id`` of the original trade card.
        direction: ``"long"`` or ``"short"``.
        avg_entry: Weighted-average entry price (used for PnL calculations).
        sl_hit: ``True`` once the stop-loss has been touched.
        tp_hit: ``True`` once the take-profit has been touched.
    """

    symbol: str
    setup: TradeSetup
    entries: List[EntryLevel]
    message_id: int
    direction: str
    avg_entry: float
    sl_hit: bool = False
    tp_hit: bool = False


# ── PositionFinder ───────────────────────────────────────────────────────────


class PositionFinder:
    """Scans top-N coins and sends trade signals to Telegram.

    Args:
        exchange: Initialised :class:`~bot.exchange.ExchangeClient`.
        cfg: Full bot configuration dictionary.
        notifier: :class:`~bot.telegram_notifier.TelegramNotifier` instance.
    """

    def __init__(
        self,
        exchange: ExchangeClient,
        cfg: Dict[str, Any],
        notifier: TelegramNotifier,
    ) -> None:
        self._exchange = exchange
        self._cfg = cfg
        self._notifier = notifier
        self._strategy = CombinedStrategy(cfg)
        # Use a large equity so position-size rejections don't suppress signals
        self._risk = RiskManager(cfg, initial_equity=1_000_000.0)
        self._active: Dict[str, ActiveSignal] = {}

        fp_cfg = cfg.get("find_positions", {})
        self._leverage: int = int(fp_cfg.get("leverage", 20))
        self._max_coins: int = int(fp_cfg.get("max_coins", 20))
        self._scan_interval: int = int(fp_cfg.get("scan_interval_seconds", 300))
        self._monitor_interval: int = int(fp_cfg.get("monitor_interval_seconds", 30))

    # ── Public API ──────────────────────────────────────────────────────────

    def scan(self) -> None:
        """Scan top coins and send new trade signals.

        Skips any symbol that already has an active signal whose stop-loss
        has not been hit yet.
        """
        symbols = self._get_top_symbols()
        logger.info("Scanning %d symbols for trade signals…", len(symbols))
        for symbol in symbols:
            active = self._active.get(symbol)
            if active is not None and not active.sl_hit:
                logger.debug(
                    "Skipping %s – active signal in place (SL not hit)", symbol
                )
                continue
            self._process_symbol(symbol)

    def monitor(self) -> None:
        """Check active signals against current prices and reply on hits."""
        for symbol in list(self._active.keys()):
            sig = self._active[symbol]
            if sig.sl_hit:
                # Signal lifecycle is complete; nothing more to monitor
                continue
            try:
                ticker = self._exchange.fetch_ticker(symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0.0)
                if price <= 0:
                    continue
                self._check_hits(symbol, sig, price)
            except Exception as exc:
                logger.warning("Monitor error for %s: %s", symbol, exc)

    def run(self) -> None:
        """Start the main event loop.

        Alternates between scanning for new signals (every
        ``scan_interval_seconds``) and monitoring active signals (every
        ``monitor_interval_seconds``).
        """
        _signal.signal(_signal.SIGINT, _stop_handler)
        _signal.signal(_signal.SIGTERM, _stop_handler)

        last_scan = 0.0
        logger.info(
            "PositionFinder started | max_coins=%d | leverage=×%d"
            " | scan_interval=%ds | monitor_interval=%ds",
            self._max_coins,
            self._leverage,
            self._scan_interval,
            self._monitor_interval,
        )

        while _RUNNING:
            now = time.monotonic()
            if now - last_scan >= self._scan_interval:
                self.scan()
                last_scan = now
            self.monitor()
            time.sleep(self._monitor_interval)

        logger.info("PositionFinder stopped.")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_top_symbols(self) -> List[str]:
        """Return up to *max_coins* USDT perpetual symbols by 24-hour volume."""
        try:
            markets = self._exchange.load_markets()
        except Exception as exc:
            logger.error("Failed to load markets: %s – using config symbols", exc)
            return list(self._cfg.get("symbols", []))

        perp_symbols = [
            sym
            for sym, mkt in markets.items()
            if mkt.get("quote") == "USDT"
            and mkt.get("type") in ("future", "swap")
            and mkt.get("settle") == "USDT"
            and not mkt.get("expiry")  # perpetuals only
        ]

        if not perp_symbols:
            logger.warning(
                "No USDT perpetual markets found – falling back to config symbols"
            )
            return list(self._cfg.get("symbols", []))

        # Fetch tickers to rank by volume (cap at 200 to avoid huge API calls)
        try:
            tickers = self._exchange.fetch_tickers(perp_symbols[:200])
        except Exception as exc:
            logger.warning(
                "fetch_tickers failed: %s – using first %d perp symbols",
                exc,
                self._max_coins,
            )
            return perp_symbols[: self._max_coins]

        ranked = sorted(
            [
                (sym, float(t.get("quoteVolume") or 0))
                for sym, t in tickers.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return [sym for sym, _ in ranked[: self._max_coins]]

    def _process_symbol(self, symbol: str) -> None:
        """Evaluate *symbol* and send a Telegram trade card if actionable."""
        primary_tf = self._cfg.get("timeframes", {}).get("primary", "1h")
        try:
            raw = self._exchange.fetch_ohlcv(symbol, timeframe=primary_tf, limit=500)
        except Exception as exc:
            logger.debug("OHLCV fetch failed for %s: %s", symbol, exc)
            return

        if not raw or len(raw) < 100:
            logger.debug(
                "Insufficient OHLCV data for %s (%d candles)", symbol, len(raw or [])
            )
            return

        df = prepare_ohlcv(raw)
        df = add_all_indicators(df, self._cfg)

        sig: Signal = self._strategy.generate_signal(df, symbol)
        if not sig.is_actionable():
            return

        latest_price = float(df["close"].iloc[-1])
        latest_atr = (
            float(df["atr"].iloc[-1]) if "atr" in df.columns else float("nan")
        )

        setup = self._risk.evaluate(sig, latest_atr, latest_price)
        if not setup.approved:
            logger.debug(
                "Signal for %s rejected by risk manager: %s",
                symbol,
                setup.rejection_reason,
            )
            return

        entries = self._build_entries(sig, setup)
        avg_entry = _weighted_avg([e.price for e in entries], [e.weight for e in entries])
        text = self._format_message(symbol, sig.direction, setup, entries, avg_entry)
        msg_id = self._notifier.send_message(text)

        if msg_id is None:
            logger.warning("Failed to send Telegram notification for %s", symbol)
            return

        self._active[symbol] = ActiveSignal(
            symbol=symbol,
            setup=setup,
            entries=entries,
            message_id=msg_id,
            direction=sig.direction,
            avg_entry=avg_entry,
        )
        logger.info(
            "Trade card sent | %s | direction=%s | msg_id=%d",
            symbol,
            sig.direction,
            msg_id,
        )

    def _build_entries(self, sig: Signal, setup: TradeSetup) -> List[EntryLevel]:
        """Extract DCA entry levels from the signal meta or fall back to single entry."""
        # Combined strategy wraps individual meta under best_signal_meta
        dca_layers: Optional[List[Dict[str, Any]]] = sig.meta.get("dca_layers") or (
            sig.meta.get("best_signal_meta") or {}
        ).get("dca_layers")

        if dca_layers:
            raw = [
                EntryLevel(
                    price=float(layer["price"]),
                    weight=float(layer.get("weight", 0.25)),
                )
                for layer in dca_layers
                if layer.get("price")
            ]
            if raw:
                total_w = sum(e.weight for e in raw)
                if total_w > 0:
                    for e in raw:
                        e.weight = e.weight / total_w
                return raw

        # Single-entry fallback
        return [EntryLevel(price=setup.entry_price, weight=1.0)]

    def _check_hits(self, symbol: str, sig: ActiveSignal, price: float) -> None:
        """Detect entry / TP / SL hits and send Telegram replies accordingly."""
        direction = sig.direction

        # ── Entry levels ───────────────────────────────────────────────────
        for idx, entry in enumerate(sig.entries):
            if entry.hit:
                continue
            crossed = (
                price <= entry.price if direction == "long" else price >= entry.price
            )
            if crossed:
                entry.hit = True
                text = (
                    f"✅ <b>Entry #{idx + 1} reached</b> "
                    f"@ <b>{_fmt_price(entry.price)}</b>"
                )
                self._notifier.reply_to(sig.message_id, text)
                logger.info(
                    "Entry #%d hit for %s @ %.6f", idx + 1, symbol, entry.price
                )

        # ── Take-profit ────────────────────────────────────────────────────
        if not sig.tp_hit:
            tp_crossed = (
                price >= sig.setup.take_profit
                if direction == "long"
                else price <= sig.setup.take_profit
            )
            if tp_crossed:
                sig.tp_hit = True
                pnl = self._pnl_pct(sig.avg_entry, sig.setup.take_profit, direction)
                text = (
                    f"🎯 <b>Take Profit Hit!</b>\n"
                    f"Price: <b>{_fmt_price(sig.setup.take_profit)}</b>\n"
                    f"Profit (×{self._leverage}): <b>+{pnl:.2f}%</b>"
                )
                self._notifier.reply_to(sig.message_id, text)
                logger.info("TP hit for %s | pnl=+%.2f%%", symbol, pnl)

        # ── Stop-loss ──────────────────────────────────────────────────────
        if not sig.sl_hit:
            sl_crossed = (
                price <= sig.setup.stop_loss
                if direction == "long"
                else price >= sig.setup.stop_loss
            )
            if sl_crossed:
                sig.sl_hit = True
                loss = self._pnl_pct(sig.avg_entry, sig.setup.stop_loss, direction)
                text = (
                    f"🛑 <b>Stop Loss Hit!</b>\n"
                    f"Price: <b>{_fmt_price(sig.setup.stop_loss)}</b>\n"
                    f"Loss (×{self._leverage}): <b>{loss:.2f}%</b>"
                )
                self._notifier.reply_to(sig.message_id, text)
                logger.info("SL hit for %s | pnl=%.2f%%", symbol, loss)

    def _pnl_pct(self, entry: float, exit_price: float, direction: str) -> float:
        """Compute leveraged PnL percentage.

        Returns a positive value for profit and negative for loss in both
        long and short scenarios.
        """
        if entry == 0:
            return 0.0
        raw = (exit_price - entry) / entry
        if direction == "short":
            raw = -raw
        return raw * self._leverage * 100

    def _format_message(
        self,
        symbol: str,
        direction: str,
        setup: TradeSetup,
        entries: List[EntryLevel],
        avg_entry: float,
    ) -> str:
        """Build the Telegram HTML trade card."""
        dir_emoji = "📈" if direction == "long" else "📉"
        dir_label = "LONG" if direction == "long" else "SHORT"
        display = _display_symbol(symbol)

        profit = self._pnl_pct(avg_entry, setup.take_profit, direction)
        risk = self._pnl_pct(avg_entry, setup.stop_loss, direction)

        _NUMERALS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

        lines = [
            f"🔍 <b>{display}</b>",
            "",
            f"{dir_emoji} <b>Direction:</b> {dir_label}",
            f"⚡ <b>Leverage:</b> ×{self._leverage} Cross",
            "",
            "📍 <b>Entries (DCA):</b>",
        ]

        for i, entry in enumerate(entries):
            num = _NUMERALS[i] if i < len(_NUMERALS) else f"{i + 1}."
            pct = f"{entry.weight * 100:.0f}%"
            lines.append(f"  {num} <b>{_fmt_price(entry.price)}</b> ({pct})")

        lines += [
            "",
            f"🎯 <b>Take Profit:</b> {_fmt_price(setup.take_profit)}",
            f"🛑 <b>Stop Loss:</b> {_fmt_price(setup.stop_loss)}",
            "",
            f"💰 <b>Profit if TP hit (×{self._leverage}):</b> +{profit:.2f}%",
            f"💸 <b>Risk if SL hit (×{self._leverage}):</b> {risk:.2f}%",
        ]

        return "\n".join(lines)


# ── Utility functions ────────────────────────────────────────────────────────


def _weighted_avg(values: List[float], weights: List[float]) -> float:
    """Return the weighted average of *values* using *weights*."""
    total_w = sum(weights)
    if total_w == 0:
        return values[0] if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _fmt_price(price: float) -> str:
    """Format a price with an appropriate number of decimal places."""
    if price >= 100:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:,.4f}"
    return f"${price:.6f}"


def _display_symbol(symbol: str) -> str:
    """Convert a ccxt symbol like ``BTC/USDT:USDT`` to ``BTCUSDT``."""
    base = symbol.split(":")[0]  # drop the settle currency suffix
    return base.replace("/", "")
