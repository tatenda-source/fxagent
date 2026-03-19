"""
5-Minute Backtest Runner for V3 Ensemble

Fetches 60 days of 5m data, trains ensemble on first 70%, backtests on remaining 30%.
Uses walk-forward approach: model only sees past data at each prediction point.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from data.fetcher import ForexFetcher
from indicators.technical import add_all_indicators
from models.ensemble import EnsemblePredictor
from models.model_utils import FEATURE_COLS, log_return_to_price
from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics
from config import ALL_PAIRS, SEQUENCE_LENGTH, PAIR_DISPLAY

# 5m-specific config
WARMUP_BARS = 200          # Indicator warmup
TRAIN_PCT = 0.70           # Train on first 70%
MIN_BARS = 1000            # Minimum bars needed
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5
CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to trade
# Annualize for 5m: ~288 bars/day * 252 trading days
ANNUALIZATION_FACTOR = np.sqrt(288 * 252)


def fetch_5m_data(pairs: list) -> dict:
    """Fetch 60 days of 5-minute data for all pairs."""
    fetcher = ForexFetcher()
    data = {}
    logger.info(f"Fetching 5m data for {len(pairs)} pairs...")
    for pair in pairs:
        try:
            df = fetcher.fetch_historical(pair, period="60d", interval="5m")
            if not df.empty and len(df) >= MIN_BARS:
                data[pair] = df
                logger.info(f"  {pair}: {len(df)} bars")
            else:
                logger.warning(f"  {pair}: insufficient data ({len(df) if not df.empty else 0} bars)")
        except Exception as e:
            logger.error(f"  {pair}: fetch failed — {e}")
    return data


def prepare_5m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators and clean for 5m data."""
    df = add_all_indicators(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def train_ensemble_on_slice(df_train: pd.DataFrame, seq_length: int = SEQUENCE_LENGTH):
    """Train ensemble on a slice of 5m data. Returns (ensemble, scaler) or None."""
    from sklearn.preprocessing import RobustScaler

    available = [c for c in FEATURE_COLS if c in df_train.columns]
    data = df_train[available].values
    close_prices = df_train["Close"].values

    if len(data) < seq_length + 20:
        return None, None, None

    log_returns = np.log(close_prices[1:] / close_prices[:-1])

    scaler = RobustScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)

    X, y = [], []
    for i in range(seq_length, len(scaled) - 1):
        X.append(scaled[i - seq_length: i])
        y.append(log_returns[i])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 20:
        return None, None, None

    num_features = X.shape[2]
    ensemble = EnsemblePredictor(num_features)
    result = ensemble.train(X, y)

    return ensemble, scaler, result


def make_strategy_fn(ensemble, scaler, feature_cols, seq_length, confidence_threshold):
    """Create a strategy function that uses the trained ensemble."""

    def strategy(row, prev_rows):
        if len(prev_rows) < seq_length:
            return None

        available = [c for c in feature_cols if c in prev_rows.columns]
        window = prev_rows[available].iloc[-seq_length:].values

        # Check for NaN
        if np.isnan(window).any():
            return None

        scaled = scaler.transform(window)
        X = scaled.reshape(1, seq_length, -1)

        try:
            pred_return, direction, confidence, uncertainty = ensemble.predict_direction_confidence(X)
        except Exception:
            return None

        if confidence < confidence_threshold:
            return None

        if abs(pred_return) > 0.05:  # Sanity check for 5m
            return None

        atr = float(row["ATR"]) if "ATR" in row.index and not np.isnan(row["ATR"]) else None
        if atr is None or atr <= 0:
            return None

        action = "BUY" if direction == "UP" else "SELL"
        return {"action": action, "atr": atr}

    return strategy


def run_backtest_for_pair(pair: str, df: pd.DataFrame) -> dict:
    """Run full train + backtest cycle for one pair."""
    df = prepare_5m_features(df)

    if len(df) < WARMUP_BARS + SEQUENCE_LENGTH + 50:
        logger.warning(f"  {pair}: not enough bars after indicator warmup")
        return None

    # Split: train on first 70%, test on last 30%
    split_idx = int(len(df) * TRAIN_PCT)
    df_train = df.iloc[WARMUP_BARS:split_idx]
    df_test = df.iloc[WARMUP_BARS:]  # Engine will use all data but strategy only looks back

    available = [c for c in FEATURE_COLS if c in df.columns]

    logger.info(f"  {pair}: training on {len(df_train)} bars, testing on {len(df) - split_idx} bars")

    # Train ensemble
    ensemble, scaler, train_result = train_ensemble_on_slice(df_train)
    if ensemble is None:
        logger.warning(f"  {pair}: training failed")
        return None

    lstm_dir = train_result.get("lstm_dir_acc", 0)
    gbm_dir = train_result.get("gbm_dir_acc", 0)
    logger.info(
        f"  {pair}: LSTM dir_acc={lstm_dir:.1%}, GBM dir_acc={gbm_dir:.1%}, "
        f"weights: LSTM={train_result['lstm_weight']:.2f} GBM={train_result['gbm_weight']:.2f}"
    )

    # Create strategy function
    strategy_fn = make_strategy_fn(ensemble, scaler, available, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD)

    # Run backtest on TEST portion only
    # We need to pass the full dataframe but only count trades from split_idx onward
    engine = BacktestEngine(initial_balance=10000)

    # Custom backtest that only trades in the test window
    balance = 10000.0
    equity_curve = []
    trades = []
    position = None

    for i in range(split_idx, len(df)):
        row = df.iloc[i]
        prev = df.iloc[:i]

        if position is not None:
            pnl = engine._check_exit(position, row)
            if pnl is not None:
                balance += pnl
                position["pnl"] = pnl
                position["exit_price"] = float(row["Close"])
                position["exit_date"] = str(row.name)
                trades.append(position)
                position = None

        if position is None:
            signal = strategy_fn(row, prev)
            if signal and signal.get("action"):
                position = engine._open_position(signal, row, balance)

        equity_curve.append(balance)

    # Close remaining position
    if position is not None:
        last = df.iloc[-1]
        if position["type"] == "BUY":
            pnl = (float(last["Close"]) - position["entry"]) * position["size"]
        else:
            pnl = (position["entry"] - float(last["Close"])) * position["size"]
        balance += pnl
        position["pnl"] = pnl
        position["exit_price"] = float(last["Close"])
        trades.append(position)
        equity_curve[-1] = balance

    metrics = compute_metrics(trades, equity_curve, 10000)

    # Fix Sharpe for 5m timeframe
    if len(equity_curve) > 1:
        eq = np.array(equity_curve)
        rets = np.diff(eq) / eq[:-1]
        if np.std(rets) > 0:
            metrics["sharpe_ratio"] = round(
                float(np.mean(rets) / np.std(rets)) * ANNUALIZATION_FACTOR, 2
            )

    # Add direction accuracy from trades
    if trades:
        correct_dir = 0
        for t in trades:
            if t["type"] == "BUY" and t["pnl"] > 0:
                correct_dir += 1
            elif t["type"] == "SELL" and t["pnl"] > 0:
                correct_dir += 1
        metrics["direction_accuracy"] = round(correct_dir / len(trades) * 100, 2)
    else:
        metrics["direction_accuracy"] = 0

    return {
        "pair": pair,
        "metrics": metrics,
        "trades": trades,
        "train_result": {
            "lstm_dir_acc": lstm_dir,
            "gbm_dir_acc": gbm_dir,
        },
    }


def main():
    logger.info("=" * 60)
    logger.info("5-MINUTE BACKTEST — V3 ENSEMBLE")
    logger.info("=" * 60)

    # Fetch data
    data = fetch_5m_data(ALL_PAIRS)

    if not data:
        logger.error("No data fetched. Aborting.")
        return

    # Run backtests
    results = {}
    for pair, df in data.items():
        logger.info(f"\n--- Backtesting {PAIR_DISPLAY.get(pair, pair)} ---")
        result = run_backtest_for_pair(pair, df)
        if result:
            results[pair] = result

    # Print summary
    print("\n" + "=" * 80)
    print("5-MINUTE BACKTEST RESULTS — V3 ENSEMBLE (60-day window, 30% out-of-sample)")
    print("=" * 80)
    print(f"{'Pair':<18} {'Trades':>7} {'Win%':>7} {'Return%':>9} {'PF':>7} {'Sharpe':>8} {'MaxDD%':>8} {'DirAcc%':>9}")
    print("-" * 80)

    total_trades = 0
    total_return = 0
    win_rates = []

    for pair, r in sorted(results.items()):
        m = r["metrics"]
        display = PAIR_DISPLAY.get(pair, pair)
        print(
            f"{display:<18} {m['num_trades']:>7} {m['win_rate']:>6.1f}% {m['total_return']:>8.2f}% "
            f"{m['profit_factor']:>7.2f} {m['sharpe_ratio']:>8.2f} {m['max_drawdown']:>7.2f}% "
            f"{m['direction_accuracy']:>8.1f}%"
        )
        total_trades += m["num_trades"]
        total_return += m["total_return"]
        if m["num_trades"] > 0:
            win_rates.append(m["win_rate"])

    print("-" * 80)
    avg_win = np.mean(win_rates) if win_rates else 0
    print(f"{'AGGREGATE':<18} {total_trades:>7} {avg_win:>6.1f}% {total_return:>8.2f}%")
    print(f"\nPairs tested: {len(results)}/{len(data)}")
    print(f"Total trades: {total_trades}")
    print(f"Average win rate: {avg_win:.1f}%")
    print(f"Combined return: {total_return:.2f}%")


if __name__ == "__main__":
    main()
