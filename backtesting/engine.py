from typing import Optional

import pandas as pd

from backtesting.metrics import compute_metrics
from config import ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, MAX_RISK_PER_TRADE


class BacktestEngine:
    """Walk-forward backtesting engine."""

    def __init__(self, initial_balance: float = 10000, risk_per_trade: float = MAX_RISK_PER_TRADE):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade

    def run(self, df: pd.DataFrame, strategy_fn) -> dict:
        """
        Run backtest.

        Args:
            df: OHLCV DataFrame with indicators already applied.
            strategy_fn: callable(row, prev_rows) -> {"action": "BUY"|"SELL", "atr": float} or None

        Returns:
            {"trades": [...], "equity_curve": pd.Series, "metrics": dict}
        """
        balance = self.initial_balance
        equity_curve = []
        trades = []
        position = None

        start_idx = 200  # Skip first 200 rows for indicator warmup

        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            prev = df.iloc[:i]

            # Check if current position hit SL or TP
            if position is not None:
                pnl = self._check_exit(position, row)
                if pnl is not None:
                    balance += pnl
                    position["pnl"] = pnl
                    position["exit_price"] = float(row["Close"])
                    position["exit_date"] = str(row.name)
                    trades.append(position)
                    position = None

            # Generate new signal if no position
            if position is None:
                signal = strategy_fn(row, prev)
                if signal and signal.get("action"):
                    position = self._open_position(signal, row, balance)

            equity_curve.append(balance)

        # Close any remaining position at last price
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

        metrics = compute_metrics(trades, equity_curve, self.initial_balance)

        return {
            "trades": trades,
            "equity_curve": pd.Series(equity_curve, index=df.index[start_idx:]),
            "metrics": metrics,
        }

    def _check_exit(self, position: dict, row) -> Optional[float]:
        """Check if row's High/Low hit SL or TP. Returns P&L or None."""
        high = float(row["High"])
        low = float(row["Low"])

        if position["type"] == "BUY":
            if low <= position["sl"]:
                return (position["sl"] - position["entry"]) * position["size"]
            if high >= position["tp"]:
                return (position["tp"] - position["entry"]) * position["size"]
        else:  # SELL
            if high >= position["sl"]:
                return (position["entry"] - position["sl"]) * position["size"]
            if low <= position["tp"]:
                return (position["entry"] - position["tp"]) * position["size"]

        return None

    def _open_position(self, signal: dict, row, balance: float) -> dict:
        """Create a position dict from signal."""
        entry = float(row["Close"])
        atr = signal.get("atr", 0.001)

        if signal["action"] == "BUY":
            sl = entry - (atr * ATR_SL_MULTIPLIER)
            tp = entry + (atr * ATR_TP_MULTIPLIER)
        else:
            sl = entry + (atr * ATR_SL_MULTIPLIER)
            tp = entry - (atr * ATR_TP_MULTIPLIER)

        risk_per_unit = abs(entry - sl)
        max_loss = balance * self.risk_per_trade
        size = max_loss / risk_per_unit if risk_per_unit > 0 else 0

        return {
            "type": signal["action"],
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": round(size, 2),
            "entry_date": str(row.name),
        }
