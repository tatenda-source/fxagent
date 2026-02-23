# fxagent — Forex AI Multi-Agent Trading System

A multi-agent AI system for forex and metals trading analysis. Uses LSTM neural networks combined with technical indicators to generate buy/sell signals with portfolio-level risk management and market regime detection.

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
        │                                             │
        ▼                                             ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│ Regime Detection │  │ Correlation      │  │ Recommendation Agent │
│ Trend/Range/Vol  │─▶│ Matrix           │─▶│ Signals + Risk Mgmt  │
└──────────────────┘  └──────────────────┘  └────────┬─────────────┘
                                                     │
                                                     ▼
                      ┌─────────────────┐   ┌──────────────────────┐
                      │ Logging Agent   │◀──│ Portfolio Risk Filter │
                      │ Track & improve │   │ Exposure + Correlation│
                      └─────────────────┘   └──────────────────────┘
```

All agents communicate through a shared context dictionary managed by the **Orchestrator**. Data persists in SQLite across runs. The pipeline now includes regime detection and portfolio-level risk filtering between signal generation and final output.

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

### Intraday Mode

```python
# Fetch both daily + 1h candles for faster signal generation
Orchestrator().run_full_pipeline(fetch_intraday=True)
```

## Project Structure

```
fxagent/
├── config.py                  # All configuration (pairs, indicators, ML, risk, regime)
├── requirements.txt
│
├── agents/                    # The 5 core agents
│   ├── base_agent.py          # Abstract base class with execute() template method
│   ├── data_agent.py          # Fetches OHLCV data (daily + intraday)
│   ├── analysis_agent.py      # Technical indicators + support/resistance
│   ├── prediction_agent.py    # LSTM price forecasting per instrument
│   ├── recommendation_agent.py # Regime-aware signal generation + risk management
│   └── logging_agent.py       # Outcome tracking, accuracy, retraining triggers
│
├── risk/                      # Risk management modules
│   ├── portfolio.py           # Portfolio risk manager (correlation, exposure limits)
│   └── regime.py              # Market regime detection (trending/ranging/volatile)
│
├── models/                    # Machine learning
│   ├── lstm_model.py          # ForexLSTM network + LSTMTrainer
│   └── model_utils.py         # Sequence preparation and scaling
│
├── indicators/                # Technical analysis
│   ├── technical.py           # SMA, EMA, RSI, MACD, Bollinger Bands, ATR
│   └── patterns.py            # Support/resistance via local min/max clustering
│
├── data/                      # Data layer
│   ├── fetcher.py             # Yahoo Finance API wrapper
│   └── storage.py             # SQLite persistence (OHLCV, signals, predictions, logs)
│
├── pipeline/                  # Orchestration
│   ├── orchestrator.py        # 8-stage pipeline (data → regime → correlation → analysis → predict → recommend → filter → log)
│   └── scheduler.py           # APScheduler for periodic auto-runs
│
├── backtesting/               # Strategy testing
│   ├── engine.py              # Walk-forward backtesting engine
│   └── metrics.py             # Sharpe, drawdown, win rate, profit factor
│
├── dashboard/                 # Streamlit web UI (7 pages)
│   ├── app.py                 # Entry point and navigation
│   └── pages/
│       ├── overview.py        # Multi-instrument summary + indicator heatmap
│       ├── pair_detail.py     # Candlestick chart + RSI + MACD + ML prediction
│       ├── signals.py         # Active signals table with P&L tracking
│       ├── portfolio.py       # Portfolio risk gauge + correlation heatmap
│       ├── regimes.py         # Market regime table + ADX chart + strategy impact
│       ├── backtest.py        # Backtesting interface with equity curve
│       └── logs.py            # Agent activity + prediction accuracy charts
│
├── db/                        # SQLite database (gitignored)
├── trained_models/            # Saved LSTM weights (gitignored)
└── tests/
```

## Pipeline Stages

The Orchestrator runs an 8-stage pipeline:

```
1. DataAgent         → Fetch OHLCV data (daily + optional intraday)
2. Regime Detection  → Classify each market as trending/ranging/volatile
3. Correlation       → Build cross-instrument correlation matrix + clusters
4. AnalysisAgent     → Technical indicators + support/resistance levels
5. PredictionAgent   → LSTM price forecasting per instrument
6. RecommendationAgent → Regime-aware signal generation with confluence scoring
7. Portfolio Filter  → Block signals exceeding exposure/correlation limits
8. LoggingAgent      → Track outcomes, compute accuracy, flag retraining
```

```python
from pipeline.orchestrator import Orchestrator

# Full pipeline
result = Orchestrator().run_full_pipeline()

# With intraday data
result = Orchestrator().run_full_pipeline(fetch_intraday=True)

# Specific pairs only
result = Orchestrator().run_full_pipeline(pairs=["GC=F", "SI=F"])

# Quick analysis (no ML — fast dashboard refresh)
result = Orchestrator().run_analysis_only()
```

### Scheduled Runs

```python
from pipeline.scheduler import start_scheduler, stop_scheduler

start_scheduler()   # Runs full pipeline every 60 minutes
stop_scheduler()    # Stop
```

## Agents

### 1. Data Agent (`agents/data_agent.py`)

Fetches historical OHLCV data from Yahoo Finance for all configured pairs. Supports both daily candles (for training and analysis) and intraday 1h candles (for faster signal generation).

**Input:** `pairs`, `period`, `interval`, `fetch_intraday` (optional overrides)
**Output:** `ohlcv_data` — `{pair: DataFrame}`, optionally `intraday_data`

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

Models are saved to `trained_models/` and reused on subsequent runs. The Logging Agent flags pairs for retraining if accuracy drops below 45%.

### 4. Recommendation Agent (`agents/recommendation_agent.py`)

Now **regime-aware**. Combines ML predictions with technical indicator confirmation and adjusts parameters based on market conditions. Uses a confluence scoring system — signals require a minimum score (adjusted by regime) from multiple sources:

| Factor | Max Score | Condition |
|--------|-----------|-----------|
| ML Prediction | 1.9 | confidence > 30% |
| RSI | 1.0 | Oversold (<40) or Overbought (>60) |
| MACD | 0.8 | Bullish/bearish crossover |
| Bollinger Bands | 0.7 | Price outside bands |
| SMA 50 Trend | 0.5 | Price aligned with trend |
| S/R Proximity | 0.5 | Near support (buy) or resistance (sell) |
| Trend Alignment | 0.5 | Signal matches ADX trend direction (trending markets only) |

**Regime-adjusted risk management:**
- SL/TP multipliers, position sizing, and entry thresholds all adapt to market conditions (see Regime Detection section below)

**Input:** `predictions`, `analyzed_data`, `sr_levels`, `regimes`
**Output:** `signals` — list of signal dicts with pair, type, SL, TP, size, confidence, reasons, regime

### 5. Logging Agent (`agents/logging_agent.py`)

Runs after every pipeline cycle. Checks if open signals hit their stop loss or take profit, computes prediction accuracy per pair, and flags underperforming models for retraining.

**Input:** `ohlcv_data`, `signals`, `predictions`
**Output:** `feedback` — `{retrain_pairs, accuracy, closed_signals}`

## Market Regime Detection (`risk/regime.py`)

Classifies each instrument's current market condition using ADX (trend strength) and ATR volatility analysis. The regime directly adjusts how the Recommendation Agent generates signals.

### Classification Logic

| Regime | Condition | Description |
|--------|-----------|-------------|
| **Trending** | ADX > 25 | Strong directional movement — follow the trend |
| **Ranging** | ADX < 25, normal volatility | Sideways market — mean-reversion opportunities |
| **Volatile** | ATR > 1.5× median ATR | Extreme price swings — protect capital |

### Strategy Adjustments by Regime

| Parameter | Trending | Ranging | Volatile |
|-----------|----------|---------|----------|
| Stop Loss multiplier | 1.0× (normal) | 0.8× (tighter) | 1.5× (wider) |
| Take Profit multiplier | 1.5× (let winners run) | 0.8× (quick profits) | 1.2× (slightly wider) |
| Min entry score | Lowered by 0.3 | Raised by 0.2 | Raised by 0.5 |
| Position size | 1.0× (full) | 0.8× (reduced) | 0.5× (halved) |

**Example:** In a volatile gold market, the system automatically widens stops to avoid noise, raises the entry bar so only high-confidence signals pass, and halves position size to protect capital.

### Usage

```python
from risk.regime import MarketRegime

# Single pair
regime = MarketRegime.detect(df)
# Returns: {"regime": "trending", "adx": 34.3, "trend_direction": "up",
#           "volatility_state": "normal", "confidence": 0.69, ...}

# All pairs at once
regimes = MarketRegime.detect_all(ohlcv_data)
```

## Portfolio Risk Management (`risk/portfolio.py`)

Controls portfolio-level risk to prevent correlated blowups. Applied as a filter after signal generation — the best signals (highest confidence) pass first, the rest are blocked if limits are hit.

### Risk Controls

| Control | Default | Description |
|---------|---------|-------------|
| Max portfolio risk | 6% | Total account risk across all open positions |
| Max correlated positions | 3 | Positions in same correlation cluster |
| Correlation threshold | 0.7 | Pairs with \|corr\| > 0.7 grouped together |

### Correlation Clusters

The system automatically detects correlated instrument groups. For example:
- **USD cluster:** EUR/USD, GBP/USD, USD/CAD, USD/CHF (all move inversely with USD)
- **Precious metals cluster:** Gold, Silver, Platinum (move together)

If you already have 3 open positions in the USD cluster, a new EUR/USD signal is blocked even if it's high-confidence.

### How Filtering Works

1. Signals are sorted by confidence (best first)
2. Each signal is checked against:
   - Is there already an open position on this pair?
   - Would this exceed the 6% total portfolio risk?
   - Are there too many correlated positions open?
3. Signals that pass all checks are approved; the rest are rejected with reasons

### Usage

```python
from risk.portfolio import PortfolioRiskManager

pm = PortfolioRiskManager()

# Check current exposure
status = pm.get_current_portfolio_risk()
# {"total_risk_pct": 0.04, "open_positions": 3, "risk_available": 0.02}

# Compute correlations
pm.compute_correlation_matrix(ohlcv_data)
clusters = pm.get_correlation_clusters()
# [["EURUSD=X", "GBPUSD=X", "USDCAD=X"], ["GC=F", "SI=F", "PL=F"]]

# Filter signals
approved = pm.filter_signals(signals, ohlcv_data)
```

## Dashboard

Launch with `streamlit run dashboard/app.py`. Seven pages:

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

### Portfolio Risk
- Risk gauge showing current vs. max portfolio risk
- Risk breakdown bar chart by open position
- Cross-instrument correlation heatmap (color-coded -1 to +1)
- Detected correlation clusters with explanation

### Market Regimes
- Summary cards: how many instruments are trending/ranging/volatile
- Regime table with ADX, trend direction, volatility state, confidence, and parameter adjustments
- Horizontal ADX bar chart with trending threshold line
- Strategy impact explanation for each regime type

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

### Core Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `FOREX_PAIRS` | 6 pairs | Major forex pairs |
| `METALS_PAIRS` | 5 pairs | Gold, Silver, Platinum, Palladium, Copper |
| `DEFAULT_PERIOD` | `"2y"` | Historical data to fetch |
| `DEFAULT_INTERVAL` | `"1d"` | Daily candle timeframe |
| `INTRADAY_INTERVAL` | `"1h"` | Intraday candle timeframe |

### ML Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `SEQUENCE_LENGTH` | 60 | LSTM lookback window (time steps) |
| `EPOCHS` | 50 | Training epochs per model |
| `LSTM_HIDDEN_SIZE` | 128 | LSTM hidden units |
| `LSTM_NUM_LAYERS` | 2 | LSTM layers |
| `LSTM_DROPOUT` | 0.2 | Dropout rate |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `BATCH_SIZE` | 32 | Training batch size |

### Risk Management

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_RISK_PER_TRADE` | 0.02 | 2% account risk per trade |
| `DEFAULT_ACCOUNT_SIZE` | 10000 | Demo account balance ($) |
| `ATR_SL_MULTIPLIER` | 1.5 | Base stop loss = 1.5 × ATR |
| `ATR_TP_MULTIPLIER` | 2.5 | Base take profit = 2.5 × ATR |
| `MAX_PORTFOLIO_RISK` | 0.06 | Max 6% total risk across all positions |
| `MAX_CORRELATED_POSITIONS` | 3 | Max positions in same correlation cluster |
| `CORRELATION_THRESHOLD` | 0.7 | Pairs with \|corr\| > 0.7 grouped |
| `CORRELATION_LOOKBACK` | 60 | Days used for correlation calculation |

### Regime Detection

| Setting | Default | Description |
|---------|---------|-------------|
| `REGIME_ADX_PERIOD` | 14 | ADX indicator period |
| `REGIME_ADX_TRENDING` | 25 | ADX > 25 = trending market |
| `REGIME_VOLATILITY_LOOKBACK` | 20 | Days for volatility comparison |
| `REGIME_VOLATILITY_HIGH_MULT` | 1.5 | ATR > 1.5× median = high volatility |

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
| `ta` | Technical indicators (pure Python, no C compilation) |
| `torch` | PyTorch for LSTM neural networks |
| `scikit-learn` | MinMaxScaler, metrics |
| `streamlit` | Web dashboard |
| `plotly` | Interactive charts |
| `apscheduler` | Periodic pipeline scheduling |
| `loguru` | Structured logging |

## Disclaimer

This system is for **educational and research purposes only**. It does not constitute financial advice. Always backtest thoroughly before risking real capital, and never trade with money you cannot afford to lose. Past performance does not guarantee future results.
