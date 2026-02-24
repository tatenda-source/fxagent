from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "momofx.db"
MODEL_DIR = BASE_DIR / "trained_models"

# Forex pairs (Yahoo Finance format uses "=X" suffix)
FOREX_PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
]

# Metals pairs (Yahoo Finance format: "GC=F" for futures, or ticker symbols)
METALS_PAIRS = [
    "GC=F",      # Gold (XAU/USD)
    "SI=F",      # Silver (XAG/USD)
    "PL=F",      # Platinum
    "PA=F",      # Palladium
    "HG=F",      # Copper
]

# All tradeable instruments
ALL_PAIRS = FOREX_PAIRS + METALS_PAIRS

PAIR_DISPLAY = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
    # Metals
    "GC=F": "Gold (XAU/USD)",
    "SI=F": "Silver (XAG/USD)",
    "PL=F": "Platinum",
    "PA=F": "Palladium",
    "HG=F": "Copper",
}

# Timeframes
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
INTRADAY_INTERVAL = "1h"

# Technical Indicator Defaults
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14

# ML Config
SEQUENCE_LENGTH = 60
TRAIN_TEST_SPLIT = 0.8
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32

# Enhanced ML Config
LSTM_ATTENTION_HEADS = 4
GRADIENT_CLIP_MAX_NORM = 1.0
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-6
VALIDATION_SPLIT = 0.15
MC_DROPOUT_PASSES = 30
MODEL_VERSION = "lstm_v2"

# Additional indicator periods
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
WILLIAMS_R_PERIOD = 14
CCI_PERIOD = 20
ROC_PERIOD = 10
MFI_PERIOD = 14

# Risk Management
MAX_RISK_PER_TRADE = 0.02
DEFAULT_ACCOUNT_SIZE = 10000
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

# Portfolio Risk Management
MAX_PORTFOLIO_RISK = 0.06           # Max 6% total account risk across all open positions
MAX_CORRELATED_POSITIONS = 3        # Max positions in same correlation cluster
CORRELATION_THRESHOLD = 0.7         # Pairs with |corr| > 0.7 are considered correlated
CORRELATION_LOOKBACK = 60           # Days to compute rolling correlation

# Regime Detection
REGIME_ADX_PERIOD = 14              # ADX period for trend strength
REGIME_ADX_TRENDING = 25            # ADX > 25 = trending market
REGIME_VOLATILITY_LOOKBACK = 20     # Days for volatility regime
REGIME_VOLATILITY_HIGH_MULT = 1.5   # ATR > 1.5x median = high volatility

# Scheduling
UPDATE_INTERVAL_MINUTES = 60
