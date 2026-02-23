# fxagent — Forex AI Multi-Agent Trading System

A multi-agent AI system for forex and metals trading analysis. Uses LSTM neural networks combined with technical indicators to generate buy/sell signals with risk management.

## Supported Instruments

**Forex:** EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF
**Metals:** Gold (XAU/USD), Silver (XAG/USD), Platinum, Palladium, Copper

## Architecture

```
Market Data (Yahoo Finance)
        │
        ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Data Agent  │────▶│ Analysis Agent  │────▶│ Prediction Agent │
│ Fetch & store│     │ Indicators + S/R│     │ LSTM forecasting │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                      │
                                                      ▼
                     ┌─────────────────┐     ┌──────────────────────┐
                     │ Logging Agent   │◀────│ Recommendation Agent │
                     │ Track & improve │     │ Signals + Risk Mgmt  │
                     └─────────────────┘     └──────────────────────┘
```

All agents communicate through a shared context dictionary managed by the **Orchestrator**. Data persists in SQLite across runs.

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (fetches data, trains models, generates signals)
python -c "from pipeline.orchestrator import Orchestrator; Orchestrator().run_full_pipeline()"

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

The first run trains LSTM models for all 11 instruments (~1-2 minutes). Subsequent runs load saved models instantly.

## Project Structure

```
fxagent/
├── config.py                  # All configuration (pairs, indicator params, ML, risk)
├── requirements.txt           # Python dependencies
│
├── agents/                    # The 5 core agents
│   ├── base_agent.py          # Abstract base class with execute() template method
│   ├── data_agent.py          # Fetches OHLCV data from Yahoo Finance
│   ├── analysis_agent.py      # Applies technical indicators + support/resistance
│   ├── prediction_agent.py    # Trains/loads LSTM models, predicts prices
│   ├── recommendation_agent.py # Generates signals with SL/TP/position sizing
│   └── logging_agent.py       # Tracks outcomes, computes accuracy, flags retraining
│
├── models/                    # Machine learning
│   ├── lstm_model.py          # ForexLSTM network + LSTMTrainer
│   └── model_utils.py         # Sequence preparation and scaling
│
├── indicators/                # Technical analysis
│   ├── technical.py           # SMA, EMA, RSI, MACD, Bollinger Bands, ATR
│   └── patterns.py            # Support/resistance detection via local min/max
│
├── data/                      # Data layer
│   ├── fetcher.py             # Yahoo Finance API wrapper
│   └── storage.py             # SQLite persistence (OHLCV, signals, predictions, logs)
│
├── pipeline/                  # Orchestration
│   ├── orchestrator.py        # Chains all 5 agents in sequence
│   └── scheduler.py           # APScheduler for periodic auto-runs
│
├── backtesting/               # Strategy testing
│   ├── engine.py              # Walk-forward backtesting engine
│   └── metrics.py             # Sharpe, drawdown, win rate, profit factor
│
├── dashboard/                 # Streamlit web UI
│   ├── app.py                 # Entry point and navigation
│   └── pages/
│       ├── overview.py        # Multi-instrument summary + indicator heatmap
│       ├── pair_detail.py     # Candlestick chart + RSI + MACD + ML prediction
│       ├── signals.py         # Active signals table with P&L tracking
│       ├── backtest.py        # Backtesting interface with equity curve
│       └── logs.py            # Agent activity + prediction accuracy charts
│
├── db/                        # SQLite database (gitignored)
├── trained_models/            # Saved LSTM weights (gitignored)
└── tests/
```

## Agents

### 1. Data Agent (`agents/data_agent.py`)

Fetches historical OHLCV data from Yahoo Finance for all configured pairs. Cleans the data (drops NaN, ensures float types, sorts by date) and stores it in SQLite.

**Input:** `pairs`, `period`, `interval` (optional overrides)
**Output:** `ohlcv_data` — dict of `{pair: DataFrame}`

### 2. Analysis Agent (`agents/analysis_agent.py`)

Applies technical indicators to each pair's price data and detects support/resistance levels.

**Indicators computed:**
- **Trend:** SMA (20, 50, 200), EMA (12, 26)
- **Momentum:** RSI (14), MACD (12/26/9)
- **Volatility:** Bollinger Bands (20, 2σ), ATR (14)
- **Patterns:** Support/resistance via local min/max with price clustering

**Input:** `ohlcv_data`
**Output:** `analyzed_data` (DataFrames with 18 indicator columns), `sr_levels`

### 3. Prediction Agent (`agents/prediction_agent.py`)

Trains or loads a 2-layer LSTM neural network per instrument. Uses the last 60 time steps of Close + 10 indicator features to predict the next closing price.

**Model architecture:**
- Input: 60-step sequences × 11 features
- LSTM: 2 layers, 128 hidden units, 0.2 dropout
- Output: FC(128→64→1) with ReLU activation

**Input:** `analyzed_data`
**Output:** `predictions` — `{pair: {predicted_price, current_price, direction, confidence}}`

Models are saved to `trained_models/` and reused on subsequent runs. The Logging Agent can flag pairs for retraining if accuracy drops below 45%.

### 4. Recommendation Agent (`agents/recommendation_agent.py`)

Combines ML predictions with technical indicator confirmation to generate actionable signals. Uses a confluence scoring system — signals require a minimum score of 2.0 from multiple sources:

| Factor | Max Score | Condition |
|--------|-----------|-----------|
| ML Prediction | 1.9 | confidence > 30% |
| RSI | 1.0 | Oversold (<40) or Overbought (>60) |
| MACD | 0.8 | Bullish/bearish crossover |
| Bollinger Bands | 0.7 | Price outside bands |
| SMA 50 Trend | 0.5 | Price aligned with trend |
| S/R Proximity | 0.5 | Near support (buy) or resistance (sell) |

**Risk management:**
- **Stop Loss:** Entry ± 1.5 × ATR
- **Take Profit:** Entry ± 2.5 × ATR (risk:reward ≈ 1:1.67)
- **Position Sizing:** Max 2% account risk per trade

**Input:** `predictions`, `analyzed_data`, `sr_levels`
**Output:** `signals` — list of signal dicts with pair, type, SL, TP, size, confidence, reasons

### 5. Logging Agent (`agents/logging_agent.py`)

Runs after every pipeline cycle. Checks if open signals hit their stop loss or take profit, computes prediction accuracy per pair, and flags underperforming models for retraining.

**Input:** `ohlcv_data`, `signals`, `predictions`
**Output:** `feedback` — `{retrain_pairs, accuracy, closed_signals}`

## Pipeline

The Orchestrator (`pipeline/orchestrator.py`) chains all 5 agents:

```
DataAgent → AnalysisAgent → PredictionAgent → RecommendationAgent → LoggingAgent
```

Each agent's output is merged into a shared context dictionary. The pipeline can be run:

```python
from pipeline.orchestrator import Orchestrator

# Full pipeline (all agents including ML)
result = Orchestrator().run_full_pipeline()

# Quick analysis only (no ML — fast dashboard refresh)
result = Orchestrator().run_analysis_only()

# Specific pairs only
result = Orchestrator().run_full_pipeline(pairs=["GC=F", "SI=F"])
```

### Scheduled Runs

```python
from pipeline.scheduler import start_scheduler, stop_scheduler

start_scheduler()   # Runs full pipeline every 60 minutes
stop_scheduler()    # Stop
```

The interval is configured by `UPDATE_INTERVAL_MINUTES` in `config.py`.

## Dashboard

Launch with `streamlit run dashboard/app.py`. Five pages:

### Overview
- Price cards for all 11 instruments with 24h change
- Indicator signal heatmap (color-coded bullish/bearish/neutral)
- Recent signals table

### Pair Detail
- Interactive Plotly candlestick chart with SMA/EMA/Bollinger overlays
- RSI subplot with overbought (70) / oversold (30) zones
- MACD subplot with histogram
- ML prediction display (current → predicted price, direction, confidence)
- Current indicator values

### Active Signals
- Filterable table (by status, signal type)
- P&L color coding (green/red)
- Summary stats: total signals, open count, total P&L, win rate

### Backtesting
- Select pair, initial balance, RSI thresholds, minimum score
- Walk-forward backtest with equity curve chart
- Metrics: total return, Sharpe ratio, max drawdown, win rate, profit factor, trade log

### Agent Logs
- Prediction accuracy over time (error % by pair)
- Agent activity log table (filterable by agent)
- Signal outcome pie chart (TP hit vs SL hit)

## Backtesting

The backtesting engine (`backtesting/engine.py`) simulates trading on historical data:

```python
from data.storage import Storage
from indicators.technical import add_all_indicators
from backtesting.engine import BacktestEngine

storage = Storage()
df = storage.get_ohlcv("GC=F")
df = add_all_indicators(df)
df.dropna(inplace=True)

def my_strategy(row, prev):
    if row["RSI"] < 30 and row["MACD"] > row["MACD_signal"]:
        return {"action": "BUY", "atr": row["ATR"]}
    if row["RSI"] > 70 and row["MACD"] < row["MACD_signal"]:
        return {"action": "SELL", "atr": row["ATR"]}
    return None

engine = BacktestEngine(initial_balance=10000)
results = engine.run(df, my_strategy)
print(results["metrics"])
```

**Metrics computed:** total return, number of trades, win rate, average win/loss, profit factor, Sharpe ratio (annualized), max drawdown.

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `FOREX_PAIRS` | 6 pairs | Major forex pairs |
| `METALS_PAIRS` | 5 pairs | Gold, Silver, Platinum, Palladium, Copper |
| `DEFAULT_PERIOD` | `"2y"` | Historical data to fetch |
| `DEFAULT_INTERVAL` | `"1d"` | Candle timeframe |
| `SEQUENCE_LENGTH` | 60 | LSTM lookback window |
| `EPOCHS` | 50 | Training epochs per model |
| `LSTM_HIDDEN_SIZE` | 128 | LSTM hidden units |
| `MAX_RISK_PER_TRADE` | 0.02 | 2% account risk per trade |
| `DEFAULT_ACCOUNT_SIZE` | 10000 | Demo account balance ($) |
| `ATR_SL_MULTIPLIER` | 1.5 | Stop loss = 1.5 × ATR |
| `ATR_TP_MULTIPLIER` | 2.5 | Take profit = 2.5 × ATR |
| `UPDATE_INTERVAL_MINUTES` | 60 | Scheduler frequency |

## Database Schema

SQLite at `db/fxagent.db` with 4 tables:

- **ohlcv** — Historical price data (pair, timestamp, OHLCV, interval)
- **signals** — Generated trading signals (pair, type, SL, TP, size, status, P&L)
- **predictions** — ML predictions for accuracy tracking (pair, predicted vs actual)
- **agent_logs** — Agent activity log (agent name, level, message, metadata)

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Market data from Yahoo Finance |
| `pandas`, `numpy` | Data manipulation |
| `ta` | Technical indicators (pure Python) |
| `torch` | PyTorch for LSTM neural networks |
| `scikit-learn` | MinMaxScaler, metrics |
| `streamlit` | Web dashboard |
| `plotly` | Interactive charts |
| `apscheduler` | Periodic pipeline scheduling |
| `loguru` | Structured logging |

## Disclaimer

This system is for **educational and research purposes only**. It does not constitute financial advice. Always backtest thoroughly before risking real capital, and never trade with money you cannot afford to lose. Past performance does not guarantee future results.
