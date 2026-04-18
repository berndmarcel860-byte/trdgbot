"""Tests for bot.risk_manager module."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from bot.risk_manager import RiskManager, TradeSetup
from bot.strategies.base import Signal


# ── Fixtures ───────────────────────────────────────────────────────────────

_DEFAULT_CFG: Dict[str, Any] = {
    "risk": {
        "max_risk_per_trade": 0.01,
        "max_open_positions": 4,
        "max_daily_drawdown": 0.05,
        "max_total_drawdown": 0.15,
        "reward_risk_ratio": 2.5,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 3.0,
        "trailing_stop_activation": 1.5,
        "trailing_stop_distance": 1.0,
    }
}

SYMBOL = "BTC/USDT:USDT"
ATR = 500.0
PRICE = 40_000.0
EQUITY = 10_000.0


def _long_signal(sl: float | None = None, tp: float | None = None) -> Signal:
    return Signal(
        direction="long",
        strategy="test",
        symbol=SYMBOL,
        confidence=0.8,
        entry_price=PRICE,
        stop_loss=sl,
        take_profit=tp,
    )


def _short_signal(sl: float | None = None, tp: float | None = None) -> Signal:
    return Signal(
        direction="short",
        strategy="test",
        symbol=SYMBOL,
        confidence=0.8,
        entry_price=PRICE,
        stop_loss=sl,
        take_profit=tp,
    )


# ── RiskParameters ─────────────────────────────────────────────────────────

class TestRiskParameters:
    def test_defaults_loaded(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        assert rm.params.max_risk_per_trade == 0.01
        assert rm.params.reward_risk_ratio == 2.5

    def test_equity_initialised(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        assert rm.equity == EQUITY


# ── Position sizing ────────────────────────────────────────────────────────

class TestPositionSizing:
    def test_quantity_positive(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        sl = PRICE - ATR * 1.5  # 39_250
        sig = _long_signal(sl=sl)
        setup = rm.evaluate(sig, ATR, PRICE)
        assert setup.quantity > 0

    def test_risk_amount_equals_target(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        sl = PRICE - ATR * 1.5
        sig = _long_signal(sl=sl)
        setup = rm.evaluate(sig, ATR, PRICE)
        expected_risk = EQUITY * 0.01  # 1 %
        assert abs(setup.risk_amount - expected_risk) < 0.01

    def test_quantity_scales_with_equity(self):
        rm_small = RiskManager(_DEFAULT_CFG, initial_equity=1_000.0)
        rm_large = RiskManager(_DEFAULT_CFG, initial_equity=100_000.0)
        sl = PRICE - ATR * 1.5
        q_small = rm_small.evaluate(_long_signal(sl=sl), ATR, PRICE).quantity
        q_large = rm_large.evaluate(_long_signal(sl=sl), ATR, PRICE).quantity
        assert q_large > q_small


# ── R:R enforcement ────────────────────────────────────────────────────────

class TestRewardRiskRatio:
    def test_tp_adjusted_to_meet_rr(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        sl = PRICE - 500.0  # risk = 500
        tp = PRICE + 600.0  # reward = 600 → R:R = 1.2 < 2.5
        sig = _long_signal(sl=sl, tp=tp)
        setup = rm.evaluate(sig, ATR, PRICE)
        assert setup.approved
        assert setup.risk_reward >= 2.5 - 1e-9

    def test_short_tp_adjusted(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        sl = PRICE + 500.0
        tp = PRICE - 600.0  # R:R < 2.5
        sig = _short_signal(sl=sl, tp=tp)
        setup = rm.evaluate(sig, ATR, PRICE)
        assert setup.approved
        assert setup.risk_reward >= 2.5 - 1e-9


# ── Rejections ─────────────────────────────────────────────────────────────

class TestRejections:
    def test_max_positions_rejection(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        rm.open_positions = 4  # already at max
        setup = rm.evaluate(_long_signal(), ATR, PRICE)
        assert not setup.approved
        assert "max open positions" in setup.rejection_reason

    def test_halted_rejection(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        rm._halted = True
        setup = rm.evaluate(_long_signal(), ATR, PRICE)
        assert not setup.approved
        assert "halted" in setup.rejection_reason


# ── Drawdown guard ─────────────────────────────────────────────────────────

class TestDrawdownGuard:
    def test_daily_drawdown_halts_trading(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        rm.daily_start_equity = EQUITY
        # Lose 6 % – exceeds 5 % daily DD limit
        rm.update_equity(EQUITY * 0.94)
        assert rm.is_halted

    def test_total_drawdown_halts_trading(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        # Lose 16 % total
        rm.update_equity(EQUITY * 0.84)
        assert rm.is_halted

    def test_daily_reset_resumes_trading(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        rm.update_equity(EQUITY * 0.94)  # triggers daily DD halt
        assert rm.is_halted
        rm.equity = EQUITY  # restore equity for reset
        rm.reset_daily()
        assert not rm.is_halted


# ── Open/close position counters ───────────────────────────────────────────

class TestPositionCounters:
    def test_register_open_increments(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        rm.register_open()
        assert rm.open_positions == 1

    def test_register_close_decrements(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        rm.open_positions = 2
        rm.register_close()
        assert rm.open_positions == 1

    def test_register_close_never_below_zero(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        rm.open_positions = 0
        rm.register_close()
        assert rm.open_positions == 0


# ── Trailing stop ──────────────────────────────────────────────────────────

class TestTrailingStop:
    def test_trailing_stop_moves_for_long(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        entry = 40_000.0
        current_sl = entry - 1_000.0
        # Price moved 1.5× ATR above entry → trailing should activate
        current_price = entry + ATR * 1.5 + 100
        new_sl = rm.compute_trailing_stop("long", entry, current_price, ATR, current_sl)
        assert new_sl > current_sl

    def test_trailing_stop_does_not_decrease_for_long(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        entry = 40_000.0
        current_sl = entry - 200.0
        # Price just at entry – should not activate
        new_sl = rm.compute_trailing_stop("long", entry, entry, ATR, current_sl)
        assert new_sl == current_sl

    def test_trailing_stop_moves_for_short(self):
        rm = RiskManager(_DEFAULT_CFG, initial_equity=EQUITY)
        entry = 40_000.0
        current_sl = entry + 1_000.0
        current_price = entry - ATR * 1.5 - 100
        new_sl = rm.compute_trailing_stop("short", entry, current_price, ATR, current_sl)
        assert new_sl < current_sl
