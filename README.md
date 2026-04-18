# trdgbot – Crypto Futures Trading Bot

A professional-grade, fully-configurable crypto futures trading bot with built-in
risk management, multiple entry strategies (pullbacks, support/resistance, DCA with
Fibonacci), and a battery of technical indicators.

---

## Features

| Category | Details |
|---|---|
| **Strategies** | Pullback-to-EMA · Support & Resistance · DCA Fibonacci |
| **Indicators** | EMA (21/55/200) · RSI · MACD · ATR · Bollinger Bands · ADX · Stochastic RSI · Volume MA |
| **Risk Management** | ATR-based SL/TP · Position sizing (% equity risk) · Max drawdown halt · Max open positions · Trailing stops |
| **Exchange support** | Any ccxt-compatible futures exchange (Bybit, Binance USDM, OKX, …) |
| **Modes** | `dry_run: true` (paper trading) · `--live` (real orders) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and add your exchange API key and secret
```

### 3. Review config

Edit `config.yaml` to set your preferred exchange, symbols, leverage, and
strategy parameters.  The key sections are:

```yaml
exchange:
  name: bybit        # bybit | binanceusdm | okx | …
  testnet: true      # STRONGLY recommended when starting

symbols:
  - BTC/USDT:USDT
  - ETH/USDT:USDT

risk:
  max_risk_per_trade: 0.01   # 1 % of equity per trade
  max_daily_drawdown: 0.05   # halt if daily loss > 5 %

bot:
  dry_run: true              # paper trading – no real orders
```

### 4. Run (paper trading)

```bash
python main.py
```

### 5. Run (live trading – USE WITH CAUTION)

```bash
python main.py --live
```

---

## Project Structure

```
trdgbot/
├── bot/
│   ├── config.py            – YAML + env-var config loader
│   ├── exchange.py          – ccxt exchange wrapper
│   ├── indicators.py        – EMA, RSI, MACD, ATR, BB, ADX, StochRSI, Volume
│   ├── risk_manager.py      – Position sizing, SL/TP, drawdown guard, trailing stop
│   ├── order_manager.py     – Order lifecycle management
│   ├── logger.py            – Structured logging
│   └── strategies/
│       ├── base.py          – Abstract base + Signal dataclass
│       ├── pullback.py      – Pullback-to-EMA strategy
│       ├── support_resistance.py – Pivot-based S/R strategy
│       ├── dca_fibonacci.py – Fibonacci DCA layering strategy
│       └── combined.py      – Consensus signal aggregator
├── tests/
│   ├── test_indicators.py
│   ├── test_strategies.py
│   └── test_risk_manager.py
├── config.yaml
├── requirements.txt
├── .env.example
└── main.py
```

---

## Strategies

### Pullback-to-EMA

Enters on healthy pullbacks in trending markets:

- **Long**: price > EMA 200 (uptrend), price pulls back to EMA 21, RSI < 40,
  MACD histogram turning up, volume confirmation.
- **Short**: mirror conditions in downtrends.

### Support & Resistance

Detects validated S/R zones from pivot highs/lows, then enters at bounces:

- Zones require at least 2 touches within a 0.3 % price buffer.
- Entry filtered by ADX (trending market) and Stochastic RSI (oversold/overbought).

### DCA Fibonacci

Places layered limit entries at Fibonacci retracement levels (38.2 %, 50 %,
61.8 %, 78.6 %) after a significant swing:

- Each layer carries a configurable size weight (default: 40 / 30 / 20 / 10 %).
- RSI filter prevents chasing momentum.

### Combined (Consensus)

Runs all three strategies and requires at least `min_confluence` strategies to
agree on direction before entering.  The resulting signal confidence is boosted
proportionally to the number of agreeing strategies.

---

## Risk Management

| Control | Description |
|---|---|
| **Position sizing** | `quantity = (equity × risk%) / (entry − stop_loss)` |
| **ATR stop-loss** | `SL = entry ± ATR × atr_sl_multiplier` |
| **Minimum R:R** | TP is auto-adjusted if the raw R:R is below the configured minimum |
| **Max positions** | Hard cap – no new entries when the limit is reached |
| **Daily drawdown** | Trading halts when daily loss exceeds the configured % |
| **Total drawdown** | Trading halts until manually restarted |
| **Trailing stop** | Activates after price moves `trailing_stop_activation × ATR` in profit |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Disclaimer

This software is provided for educational purposes only.  Cryptocurrency futures
trading involves significant financial risk.  Past performance is not indicative
of future results.  **Never trade with money you cannot afford to lose.**
Always test thoroughly on a testnet before deploying with real funds.
