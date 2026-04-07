#!/usr/bin/env python3
"""trdgbot – main entry point.

Usage
-----
.. code-block:: bash

    # Dry-run (paper trading) – no real orders placed:
    python main.py

    # Live trading (set dry_run: false in config.yaml first):
    python main.py --live

    # Use a custom config file:
    python main.py --config /path/to/my_config.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from bot.config import load_config
from bot.exchange import ExchangeClient
from bot.indicators import add_all_indicators, prepare_ohlcv
from bot.logger import get_logger
from bot.order_manager import OrderManager
from bot.risk_manager import RiskManager
from bot.strategies.combined import CombinedStrategy
logger = get_logger(__name__)

_RUNNING = True


def _shutdown_handler(signum: int, frame: Any) -> None:
    global _RUNNING
    logger.info("Shutdown signal received – stopping after current cycle…")
    _RUNNING = False


signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ── Per-symbol processing ──────────────────────────────────────────────────

def process_symbol(
    symbol: str,
    exchange: ExchangeClient,
    strategy: CombinedStrategy,
    risk_manager: RiskManager,
    order_manager: OrderManager,
    cfg: Dict[str, Any],
) -> None:
    """Fetch data, compute indicators, generate signals, and act on them.

    Args:
        symbol: Trading pair to process.
        exchange: Exchange client.
        strategy: Combined strategy instance.
        risk_manager: Risk manager.
        order_manager: Order manager.
        cfg: Full configuration dictionary.
    """
    tf_cfg = cfg.get("timeframes", {})
    primary_tf = tf_cfg.get("primary", "1h")

    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=primary_tf, limit=500)
    except Exception as exc:
        logger.error("Failed to fetch OHLCV for %s: %s", symbol, exc)
        return

    if not raw or len(raw) < 100:
        logger.warning("Insufficient OHLCV data for %s (%d candles)", symbol, len(raw))
        return

    df = prepare_ohlcv(raw)
    df = add_all_indicators(df, cfg)

    signal = strategy.generate_signal(df, symbol)

    latest_price = float(df["close"].iloc[-1])
    latest_atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else float("nan")

    logger.info(
        "[%s] Signal: direction=%s confidence=%.2f (strategy=%s)",
        symbol,
        signal.direction,
        signal.confidence,
        signal.strategy,
    )

    # Update trailing stops for any open position
    if symbol in order_manager.open_symbols:
        order_manager.update_trailing_stops(symbol, latest_price, latest_atr)
        return  # don't open another position if one is already open

    if not signal.is_actionable():
        return

    # Validate the signal through the risk manager
    setup = risk_manager.evaluate(signal, latest_atr, latest_price)
    if not setup.approved:
        logger.info(
            "[%s] Setup rejected: %s",
            symbol,
            setup.rejection_reason,
        )
        return

    # Open the position
    order_manager.open_position(setup, signal_meta=signal.meta)


# ── Daily reset ────────────────────────────────────────────────────────────

def _is_new_day(last_reset_date: str) -> bool:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return today != last_reset_date


# ── Main loop ──────────────────────────────────────────────────────────────

def run(cfg: Dict[str, Any]) -> None:    """Start the main trading loop.

    Args:
        cfg: Fully loaded configuration dictionary.
    """
    dry_run = cfg.get("bot", {}).get("dry_run", True)
    poll_interval = cfg.get("bot", {}).get("poll_interval_seconds", 60)
    symbols: List[str] = cfg.get("symbols", ["BTC/USDT:USDT"])

    exchange = ExchangeClient(cfg, dry_run=dry_run)

    # Fetch initial equity
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance.get("USDT", {}).get("free", 10_000.0) or 10_000.0
        initial_equity = float(usdt_balance)
    except Exception:
        initial_equity = 10_000.0
        logger.warning("Could not fetch account balance – using default equity %.2f", initial_equity)

    risk_manager = RiskManager(cfg, initial_equity=initial_equity)
    order_manager = OrderManager(exchange, risk_manager, cfg)
    strategy = CombinedStrategy(cfg)

    last_reset_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    logger.info(
        "Bot started | symbols=%s | tf=%s | dry_run=%s",
        symbols,
        cfg.get("timeframes", {}).get("primary", "1h"),
        dry_run,
    )

    while _RUNNING:
        # Daily reset
        if _is_new_day(last_reset_date):
            risk_manager.reset_daily()
            last_reset_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            logger.info("Daily risk counter reset")

        if risk_manager.is_halted:
            logger.warning("Trading halted due to drawdown limits – waiting for daily reset…")
            time.sleep(poll_interval)
            continue

        for symbol in symbols:
            if not _RUNNING:
                break
            process_symbol(symbol, exchange, strategy, risk_manager, order_manager, cfg)

        logger.debug("Cycle complete – sleeping %ds", poll_interval)
        time.sleep(poll_interval)

    logger.info("Bot stopped.")


# ── Find-positions mode ────────────────────────────────────────────────────

def run_find_positions(cfg: Dict[str, Any]) -> None:
    """Start the position-finder / signal-scanner mode.

    Validates that Telegram credentials are present, then starts
    :class:`~bot.find_positions.PositionFinder` in its blocking event loop.

    Args:
        cfg: Fully loaded configuration dictionary.
    """
    from bot.find_positions import PositionFinder
    from bot.telegram_notifier import TelegramNotifier

    telegram_cfg = cfg.get("telegram", {})
    token: str = telegram_cfg.get("bot_token", "")
    channel_id: str = telegram_cfg.get("channel_id", "")

    if not token or not channel_id:
        logger.error(
            "Telegram credentials are missing. "
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID in your .env file."
        )
        sys.exit(1)

    dry_run = cfg.get("bot", {}).get("dry_run", True)
    exchange = ExchangeClient(cfg, dry_run=dry_run)
    notifier = TelegramNotifier(token, channel_id)
    finder = PositionFinder(exchange, cfg, notifier)
    finder.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="trdgbot – crypto futures trading bot")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (defaults to config.yaml in project root)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (overrides dry_run: true in config)",
    )
    parser.add_argument(
        "--find-positions",
        action="store_true",
        dest="find_positions",
        help=(
            "Run position-finder mode: scan top coins, send trade signals to "
            "Telegram, and monitor entries / TP / SL (requires TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHANNEL_ID in .env)"
        ),
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        logger.error("Config error: %s", exc)
        sys.exit(1)

    if args.live:
        cfg.setdefault("bot", {})["dry_run"] = False
        logger.warning("⚠  LIVE TRADING MODE – real orders will be placed!")

    if args.find_positions:
        run_find_positions(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
