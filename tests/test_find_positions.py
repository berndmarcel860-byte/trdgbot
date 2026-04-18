"""Tests for bot.telegram_notifier and bot.find_positions."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from bot.find_positions import (
    ActiveSignal,
    EntryLevel,
    PositionFinder,
    _display_symbol,
    _fmt_price,
    _weighted_avg,
)
from bot.risk_manager import TradeSetup
from bot.telegram_notifier import TelegramNotifier


# ── TelegramNotifier ─────────────────────────────────────────────────────────


class TestTelegramNotifier:
    def _notifier(self) -> TelegramNotifier:
        return TelegramNotifier(token="test_token", channel_id="@testchannel")

    def test_send_message_returns_message_id(self):
        notifier = self._notifier()
        mock_resp = {"ok": True, "result": {"message_id": 42, "text": "hello"}}
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_resp
            msg_id = notifier.send_message("hello")
        assert msg_id == 42

    def test_send_message_returns_none_on_api_error(self):
        notifier = self._notifier()
        mock_resp = {"ok": False, "description": "Forbidden"}
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_resp
            msg_id = notifier.send_message("hello")
        assert msg_id is None

    def test_send_message_returns_none_on_network_error(self):
        notifier = self._notifier()
        with patch("requests.post", side_effect=ConnectionError("timeout")):
            msg_id = notifier.send_message("hello")
        assert msg_id is None

    def test_reply_to_passes_reply_to_message_id(self):
        notifier = self._notifier()
        mock_resp = {"ok": True, "result": {"message_id": 99}}
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_resp
            msg_id = notifier.reply_to(42, "reply text")
        assert msg_id == 99
        payload = mock_post.call_args.kwargs["json"]
        assert payload["reply_to_message_id"] == 42

    def test_reply_to_returns_none_on_failure(self):
        notifier = self._notifier()
        mock_resp = {"ok": False, "description": "Bad Request"}
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_resp
            result = notifier.reply_to(1, "text")
        assert result is None


# ── Utility functions ────────────────────────────────────────────────────────


class TestUtilities:
    def test_weighted_avg_equal_weights(self):
        avg = _weighted_avg([100.0, 200.0], [0.5, 0.5])
        assert avg == pytest.approx(150.0)

    def test_weighted_avg_unequal_weights(self):
        avg = _weighted_avg([100.0, 200.0], [0.4, 0.6])
        assert avg == pytest.approx(160.0)

    def test_weighted_avg_single_value(self):
        avg = _weighted_avg([55.0], [1.0])
        assert avg == pytest.approx(55.0)

    def test_weighted_avg_zero_weights_returns_first(self):
        avg = _weighted_avg([77.0, 88.0], [0.0, 0.0])
        assert avg == pytest.approx(77.0)

    def test_fmt_price_large(self):
        assert _fmt_price(84_000.0) == "$84,000.00"

    def test_fmt_price_medium(self):
        price = _fmt_price(50.1234)
        assert "$" in price
        assert "50.1234" in price

    def test_fmt_price_small(self):
        price = _fmt_price(0.000123)
        assert "$" in price
        assert "0.000123" in price

    def test_display_symbol_strips_settle(self):
        assert _display_symbol("BTC/USDT:USDT") == "BTCUSDT"

    def test_display_symbol_no_suffix(self):
        assert _display_symbol("ETH/USDT") == "ETHUSDT"


# ── PositionFinder ──────────────────────────────────────────────────────────


_BASE_CFG: Dict[str, Any] = {
    "exchange": {"name": "bybit", "testnet": True, "leverage": 10, "margin_mode": "cross"},
    "timeframes": {"primary": "1h", "htf": "4h", "ltf": "15m"},
    "risk": {
        "max_risk_per_trade": 0.01,
        "max_open_positions": 20,
        "max_daily_drawdown": 0.05,
        "max_total_drawdown": 0.15,
        "reward_risk_ratio": 2.5,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 3.0,
        "trailing_stop_activation": 1.5,
        "trailing_stop_distance": 1.0,
    },
    "strategies": {
        "enabled": ["pullback", "support_resistance", "dca_fibonacci"],
        "min_confluence": 1,
        "pullback": {
            "fast_ema": 21, "slow_ema": 55, "trend_ema": 200,
            "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        },
        "support_resistance": {
            "pivot_lookback": 10, "zone_tolerance": 0.003,
            "min_touches": 2, "adx_period": 14, "adx_threshold": 20,
        },
        "dca_fibonacci": {
            "fib_levels": [0.382, 0.5, 0.618, 0.786],
            "fib_weights": [0.4, 0.3, 0.2, 0.1],
            "swing_lookback": 50, "rsi_filter_long": 50, "rsi_filter_short": 50,
        },
    },
    "indicators": {
        "atr_period": 14, "bb_period": 20, "bb_std": 2,
        "stochrsi_period": 14, "stochrsi_smooth_k": 3, "stochrsi_smooth_d": 3,
        "volume_ma_period": 20,
    },
    "find_positions": {
        "leverage": 20, "max_coins": 5,
        "scan_interval_seconds": 300, "monitor_interval_seconds": 30,
    },
    "bot": {"dry_run": True, "poll_interval_seconds": 60},
}


def _make_finder(
    notifier: Optional[TelegramNotifier] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> PositionFinder:
    exchange = MagicMock()
    cfg = cfg or _BASE_CFG
    notifier = notifier or MagicMock(spec=TelegramNotifier)
    return PositionFinder(exchange=exchange, cfg=cfg, notifier=notifier)


def _make_setup(
    symbol: str = "BTC/USDT:USDT",
    direction: str = "long",
    entry: float = 80_000.0,
    sl: float = 78_000.0,
    tp: float = 86_000.0,
) -> TradeSetup:
    return TradeSetup(
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        quantity=0.01,
        risk_amount=20.0,
        risk_reward=3.0,
        approved=True,
    )


def _make_active(
    symbol: str = "BTC/USDT:USDT",
    direction: str = "long",
    entry: float = 80_000.0,
    sl: float = 78_000.0,
    tp: float = 86_000.0,
    msg_id: int = 1,
) -> ActiveSignal:
    setup = _make_setup(symbol, direction, entry, sl, tp)
    return ActiveSignal(
        symbol=symbol,
        setup=setup,
        entries=[EntryLevel(price=entry, weight=1.0)],
        message_id=msg_id,
        direction=direction,
        avg_entry=entry,
    )


class TestPositionFinderPnL:
    def test_long_pnl_positive_at_tp(self):
        finder = _make_finder()
        # entry=100, tp=110, leverage=20 → pnl = 10/100*20*100 = 200%
        pnl = finder._pnl_pct(100.0, 110.0, "long")
        assert pnl == pytest.approx(200.0)

    def test_long_pnl_negative_at_sl(self):
        finder = _make_finder()
        # entry=100, sl=90 → pnl = -10/100*20*100 = -200%
        pnl = finder._pnl_pct(100.0, 90.0, "long")
        assert pnl == pytest.approx(-200.0)

    def test_short_pnl_positive_at_tp(self):
        finder = _make_finder()
        # entry=100, tp=90 (price went down), leverage=20 → pnl = 10/100*20*100 = 200%
        pnl = finder._pnl_pct(100.0, 90.0, "short")
        assert pnl == pytest.approx(200.0)

    def test_short_pnl_negative_at_sl(self):
        finder = _make_finder()
        # entry=100, sl=110, leverage=20 → pnl = -200%
        pnl = finder._pnl_pct(100.0, 110.0, "short")
        assert pnl == pytest.approx(-200.0)

    def test_zero_entry_returns_zero(self):
        finder = _make_finder()
        assert finder._pnl_pct(0.0, 100.0, "long") == 0.0


class TestPositionFinderBuildEntries:
    def test_single_entry_fallback_when_no_dca(self):
        finder = _make_finder()
        sig = MagicMock()
        sig.meta = {}
        setup = _make_setup(entry=50_000.0)
        entries = finder._build_entries(sig, setup)
        assert len(entries) == 1
        assert entries[0].price == pytest.approx(50_000.0)
        assert entries[0].weight == pytest.approx(1.0)

    def test_dca_layers_normalised(self):
        finder = _make_finder()
        sig = MagicMock()
        sig.meta = {
            "dca_layers": [
                {"price": 80_000.0, "weight": 0.4},
                {"price": 78_000.0, "weight": 0.3},
                {"price": 76_000.0, "weight": 0.2},
            ]
        }
        setup = _make_setup(entry=80_000.0)
        entries = finder._build_entries(sig, setup)
        assert len(entries) == 3
        total_w = sum(e.weight for e in entries)
        assert total_w == pytest.approx(1.0)

    def test_dca_via_best_signal_meta(self):
        finder = _make_finder()
        sig = MagicMock()
        sig.meta = {
            "best_signal_meta": {
                "dca_layers": [
                    {"price": 50_000.0, "weight": 0.5},
                    {"price": 48_000.0, "weight": 0.5},
                ]
            }
        }
        setup = _make_setup(entry=50_000.0)
        entries = finder._build_entries(sig, setup)
        assert len(entries) == 2


class TestPositionFinderCheckHits:
    def test_long_entry_hit_sends_reply(self):
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        sig = _make_active(direction="long", entry=80_000.0, sl=78_000.0, tp=86_000.0)
        finder._active["BTC/USDT:USDT"] = sig

        # Price drops to entry level → entry hit
        finder._check_hits("BTC/USDT:USDT", sig, 79_999.0)
        notifier.reply_to.assert_called_once()
        text = notifier.reply_to.call_args[0][1]
        assert "Entry #1" in text
        assert sig.entries[0].hit is True

    def test_long_tp_hit_sends_reply(self):
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        sig = _make_active(direction="long", entry=80_000.0, sl=78_000.0, tp=86_000.0)
        # Mark entries already hit to avoid noise
        sig.entries[0].hit = True
        finder._active["BTC/USDT:USDT"] = sig

        finder._check_hits("BTC/USDT:USDT", sig, 86_001.0)
        notifier.reply_to.assert_called_once()
        text = notifier.reply_to.call_args[0][1]
        assert "Take Profit" in text
        assert sig.tp_hit is True

    def test_long_sl_hit_sends_reply_and_sets_sl_hit(self):
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        sig = _make_active(direction="long", entry=80_000.0, sl=78_000.0, tp=86_000.0)
        sig.entries[0].hit = True
        finder._active["BTC/USDT:USDT"] = sig

        finder._check_hits("BTC/USDT:USDT", sig, 77_999.0)
        notifier.reply_to.assert_called_once()
        text = notifier.reply_to.call_args[0][1]
        assert "Stop Loss" in text
        assert sig.sl_hit is True

    def test_short_entry_hit(self):
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        # short: entry hit when price rises to or above entry
        sig = _make_active(
            direction="short", entry=82_000.0, sl=84_000.0, tp=76_000.0, msg_id=5
        )
        finder._active["BTC/USDT:USDT"] = sig

        finder._check_hits("BTC/USDT:USDT", sig, 82_001.0)
        assert sig.entries[0].hit is True

    def test_no_duplicate_entry_hit(self):
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        sig = _make_active(direction="long", entry=80_000.0, sl=78_000.0, tp=86_000.0)
        sig.entries[0].hit = True  # already hit
        finder._active["BTC/USDT:USDT"] = sig

        finder._check_hits("BTC/USDT:USDT", sig, 79_000.0)
        # No entry reply expected since it was already hit; TP/SL not crossed
        notifier.reply_to.assert_not_called()

    def test_sl_crossed_without_entry_sends_expired(self):
        """Guard: SL crossed before any entry filled → 'Expired', not 'Stop Loss'.

        Uses an intentionally inverted SL (above entry for a long) so the guard
        path fires: price drops to 84k, which is ≤ SL (85k) but NOT ≤ entry (80k).
        """
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        # SL deliberately above entry to force the "no entry before SL" path.
        sig = _make_active(direction="long", entry=80_000.0, sl=85_000.0, tp=90_000.0)
        assert sig.entries[0].hit is False
        finder._active["BTC/USDT:USDT"] = sig

        # 84k ≤ 85k (SL crossed) but 84k > 80k (entry not crossed for long)
        finder._check_hits("BTC/USDT:USDT", sig, 84_000.0)
        notifier.reply_to.assert_called_once()
        text = notifier.reply_to.call_args[0][1]
        assert "Expired" in text
        assert "Stop Loss" not in text
        assert sig.sl_hit is True
        assert sig.tp_hit is False  # TP must not fire when no entry was filled

    def test_short_sl_crossed_without_entry_sends_expired(self):
        """Guard fires for short direction: inverted SL (below entry) forces the path."""
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        # SL below entry for a short → inverted (invalid) to trigger the guard.
        sig = _make_active(
            direction="short", entry=82_000.0, sl=78_000.0, tp=70_000.0, msg_id=5
        )
        assert sig.entries[0].hit is False
        finder._active["BTC/USDT:USDT"] = sig

        # 79k ≥ 78k (SL crossed for short) but 79k < 82k (entry not hit for short)
        finder._check_hits("BTC/USDT:USDT", sig, 79_000.0)
        notifier.reply_to.assert_called_once()
        text = notifier.reply_to.call_args[0][1]
        assert "Expired" in text
        assert sig.sl_hit is True

    def test_tp_not_triggered_without_entry_hit(self):
        """TP must not fire until at least one entry has been filled."""
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        # Inverted SL below entry for a short so neither entry nor SL fires at
        # this price, letting us verify TP is also suppressed.
        sig = _make_active(
            direction="short", entry=82_000.0, sl=78_000.0, tp=70_000.0
        )
        assert sig.entries[0].hit is False
        finder._active["BTC/USDT:USDT"] = sig

        # 65k < TP (70k for short: price ≤ tp would hit TP if entry was filled)
        # but 65k < SL (78k, 65k ≥ 78k? NO → SL not crossed) and 65k < entry
        # (65k ≥ 82k? NO → entry not hit) → guard returns early, TP not checked.
        finder._check_hits("BTC/USDT:USDT", sig, 65_000.0)
        assert sig.tp_hit is False
        assert sig.sl_hit is False
        notifier.reply_to.assert_not_called()


        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)
        sig = _make_active(direction="long", entry=80_000.0, sl=78_000.0, tp=86_000.0)
        sig.entries[0].hit = True
        sig.sl_hit = True  # already triggered
        finder._active["BTC/USDT:USDT"] = sig

        # monitor() should skip this signal
        finder._exchange.fetch_ticker.return_value = {"last": 77_000.0}
        finder.monitor()
        notifier.reply_to.assert_not_called()


class TestPositionFinderScanBlocking:
    def test_active_signal_blocks_rescan_until_sl_hit(self):
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)

        # Inject an active signal whose SL has NOT been hit
        sig = _make_active(symbol="BTC/USDT:USDT")
        finder._active["BTC/USDT:USDT"] = sig

        # Stub exchange so scan() would call _process_symbol if not blocked
        finder._exchange.load_markets.return_value = {}
        finder._get_top_symbols = lambda: ["BTC/USDT:USDT"]  # type: ignore[method-assign]
        finder._process_symbol = MagicMock()  # type: ignore[method-assign]

        finder.scan()
        finder._process_symbol.assert_not_called()

    def test_active_signal_allows_rescan_after_sl_hit(self):
        notifier = MagicMock(spec=TelegramNotifier)
        finder = _make_finder(notifier=notifier)

        sig = _make_active(symbol="BTC/USDT:USDT")
        sig.sl_hit = True  # SL was hit → rescan allowed
        finder._active["BTC/USDT:USDT"] = sig

        finder._get_top_symbols = lambda: ["BTC/USDT:USDT"]  # type: ignore[method-assign]
        finder._process_symbol = MagicMock()  # type: ignore[method-assign]

        finder.scan()
        finder._process_symbol.assert_called_once_with("BTC/USDT:USDT")


class TestFormatMessage:
    def test_message_contains_required_fields(self):
        finder = _make_finder()
        setup = _make_setup(
            direction="long", entry=80_000.0, sl=78_000.0, tp=86_000.0
        )
        entries = [EntryLevel(price=80_000.0, weight=1.0)]
        msg = finder._format_message("BTC/USDT:USDT", "long", setup, entries, 80_000.0)

        assert "BTCUSDT" in msg
        assert "LONG" in msg
        assert "×20" in msg
        assert "Cross" in msg
        assert "Take Profit" in msg
        assert "Stop Loss" in msg
        assert "%" in msg

    def test_short_message_shows_short(self):
        finder = _make_finder()
        setup = _make_setup(
            direction="short", entry=82_000.0, sl=84_000.0, tp=76_000.0
        )
        entries = [EntryLevel(price=82_000.0, weight=1.0)]
        msg = finder._format_message(
            "ETH/USDT:USDT", "short", setup, entries, 82_000.0
        )
        assert "SHORT" in msg
        assert "ETHUSDT" in msg


# ── State persistence ────────────────────────────────────────────────────────


def _make_finder_with_state(state_file: str) -> PositionFinder:
    """Return a PositionFinder wired to *state_file* for persistence tests."""
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["find_positions"]["state_file"] = state_file
    return _make_finder(cfg=cfg)


class TestStatePersistence:
    def test_save_creates_json_file(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        finder = _make_finder_with_state(state_file)
        finder._active["BTC/USDT:USDT"] = _make_active()
        finder._save_state()
        assert os.path.exists(state_file)
        with open(state_file) as fh:
            data = json.load(fh)
        assert "BTC/USDT:USDT" in data

    def test_save_persists_all_fields(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        finder = _make_finder_with_state(state_file)
        sig = _make_active(entry=80_000.0, sl=78_000.0, tp=86_000.0, msg_id=99)
        sig.entries[0].hit = True
        sig.tp_hit = True
        finder._active["BTC/USDT:USDT"] = sig
        finder._save_state()
        with open(state_file) as fh:
            raw = json.load(fh)["BTC/USDT:USDT"]
        assert raw["message_id"] == 99
        assert raw["avg_entry"] == pytest.approx(80_000.0)
        assert raw["tp_hit"] is True
        assert raw["entries"][0]["hit"] is True

    def test_load_restores_active_signals(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        # Save via a first finder instance
        finder1 = _make_finder_with_state(state_file)
        finder1._active["BTC/USDT:USDT"] = _make_active(msg_id=7)
        finder1._save_state()
        # Restore via a second instance
        finder2 = _make_finder_with_state(state_file)
        assert "BTC/USDT:USDT" in finder2._active
        restored = finder2._active["BTC/USDT:USDT"]
        assert restored.symbol == "BTC/USDT:USDT"
        assert restored.message_id == 7
        assert restored.direction == "long"
        assert restored.avg_entry == pytest.approx(80_000.0)

    def test_load_restores_entry_hit_state(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        finder1 = _make_finder_with_state(state_file)
        sig = _make_active()
        sig.entries[0].hit = True
        finder1._active["BTC/USDT:USDT"] = sig
        finder1._save_state()

        finder2 = _make_finder_with_state(state_file)
        assert finder2._active["BTC/USDT:USDT"].entries[0].hit is True

    def test_load_restores_sl_hit_flag(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        finder1 = _make_finder_with_state(state_file)
        sig = _make_active()
        sig.sl_hit = True
        finder1._active["BTC/USDT:USDT"] = sig
        finder1._save_state()

        finder2 = _make_finder_with_state(state_file)
        assert finder2._active["BTC/USDT:USDT"].sl_hit is True

    def test_load_missing_file_leaves_active_empty(self, tmp_path):
        state_file = str(tmp_path / "nonexistent.json")
        finder = _make_finder_with_state(state_file)
        assert finder._active == {}

    def test_load_corrupt_file_leaves_active_empty(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        with open(state_file, "w") as fh:
            fh.write("NOT VALID JSON {{{")
        finder = _make_finder_with_state(state_file)
        assert finder._active == {}

    def test_save_without_state_file_is_noop(self):
        # _BASE_CFG has no state_file → _state_file == "" → save is a no-op
        finder = _make_finder()
        finder._active["BTC/USDT:USDT"] = _make_active()
        finder._save_state()  # must not raise

    def test_scan_saves_state_to_file(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        finder = _make_finder_with_state(state_file)
        finder._get_top_symbols = lambda: []  # type: ignore[method-assign]
        finder.scan()
        assert os.path.exists(state_file)

    def test_monitor_saves_state_to_file(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        finder = _make_finder_with_state(state_file)
        finder.monitor()  # no active signals, just verifies save was called
        assert os.path.exists(state_file)

    def test_save_uses_atomic_write(self, tmp_path):
        """Verify no partial .tmp file is left after a successful save."""
        state_file = str(tmp_path / "state.json")
        finder = _make_finder_with_state(state_file)
        finder._active["BTC/USDT:USDT"] = _make_active()
        finder._save_state()
        assert not os.path.exists(state_file + ".tmp")
        assert os.path.exists(state_file)
