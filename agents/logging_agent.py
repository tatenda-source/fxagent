import pandas as pd

from agents.base_agent import BaseAgent
from data.storage import Storage

_REQUIRED_SIGNAL_FIELDS = ("entry_price", "predicted_price", "stop_loss", "take_profit", "signal_type")


class LoggingAgent(BaseAgent):
    """Agent 5: Evaluates open signals, tracks accuracy, flags retraining needs."""

    def __init__(self):
        super().__init__(name="LoggingAgent")
        self.storage = Storage()

    def run(self, input_data: dict) -> dict:
        try:
            ohlcv = input_data.get("ohlcv_data", {})
            open_signals = self.storage.get_open_signals()
            closed_signals = self._evaluate_open_signals(ohlcv, open_signals)
            accuracy = self._compute_prediction_accuracy()
            retrain_pairs = [pair for pair, acc in accuracy.items() if acc < 0.45]

            self.logger.info(
                f"Logging complete: {len(open_signals)} signals evaluated, "
                f"{len(closed_signals)} closed, "
                f"{len(retrain_pairs)} flagged for retraining"
            )

            return {
                "feedback": {
                    "retrain_pairs": retrain_pairs,
                    "accuracy": accuracy,
                    "closed_signals": closed_signals,
                }
            }
        except Exception as exc:
            self.logger.error(f"LoggingAgent failed: {exc}")
            return {"feedback": {"retrain_pairs": [], "accuracy": {}, "closed_signals": []}}

    def _evaluate_open_signals(self, ohlcv: dict, open_signals: pd.DataFrame) -> list:
        """Check if any open signal hit SL or TP."""
        closed = []

        for _, sig in open_signals.iterrows():
            if any(pd.isna(sig.get(f)) for f in _REQUIRED_SIGNAL_FIELDS):
                self.logger.warning(f"Skipping malformed signal id={sig.get('id')}: missing required fields")
                continue

            pair = sig["pair"]
            if pair not in ohlcv:
                continue

            current_price = float(ohlcv[pair]["Close"].iloc[-1])

            if sig["signal_type"] == "BUY":
                if current_price <= sig["stop_loss"]:
                    pnl = sig["stop_loss"] - sig["entry_price"]
                    self.storage.update_signal_outcome(sig["id"], "SL_HIT", pnl)
                    closed.append({"id": sig["id"], "status": "SL_HIT", "pnl": pnl})
                elif current_price >= sig["take_profit"]:
                    pnl = sig["take_profit"] - sig["entry_price"]
                    self.storage.update_signal_outcome(sig["id"], "TP_HIT", pnl)
                    closed.append({"id": sig["id"], "status": "TP_HIT", "pnl": pnl})
            elif sig["signal_type"] == "SELL":
                if current_price >= sig["stop_loss"]:
                    pnl = sig["entry_price"] - sig["stop_loss"]
                    self.storage.update_signal_outcome(sig["id"], "SL_HIT", pnl)
                    closed.append({"id": sig["id"], "status": "SL_HIT", "pnl": pnl})
                elif current_price <= sig["take_profit"]:
                    pnl = sig["entry_price"] - sig["take_profit"]
                    self.storage.update_signal_outcome(sig["id"], "TP_HIT", pnl)
                    closed.append({"id": sig["id"], "status": "TP_HIT", "pnl": pnl})

        return closed

    def _compute_prediction_accuracy(self) -> dict:
        """Compare stored predictions to actual prices. Returns {pair: directional_accuracy}."""
        predictions = self.storage.get_predictions(limit=500)
        if predictions.empty:
            return {}

        accuracy = {}
        for pair in predictions["pair"].unique():
            pair_preds = predictions[predictions["pair"] == pair]
            if pair_preds.empty or pair_preds["actual_price"].isna().all():
                continue
            valid = pair_preds.dropna(subset=["actual_price"])
            if valid.empty:
                continue
            correct = sum(
                1 for _, row in valid.iterrows()
                if (row["predicted_price"] > row["actual_price"]) == (row["error"] is None or row["error"] >= 0)
            )
            accuracy[pair] = correct / len(valid) if len(valid) > 0 else 0

        return accuracy
