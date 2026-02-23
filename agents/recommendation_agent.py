from typing import Optional

from agents.base_agent import BaseAgent
from data.storage import Storage
from config import (
    MAX_RISK_PER_TRADE, DEFAULT_ACCOUNT_SIZE,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
)


class RecommendationAgent(BaseAgent):
    """Agent 4: Generates buy/sell signals with SL, TP, and position sizing."""

    def __init__(self, account_size: float = DEFAULT_ACCOUNT_SIZE):
        super().__init__(name="RecommendationAgent")
        self.storage = Storage()
        self.account_size = account_size

    def run(self, input_data: dict) -> dict:
        predictions = input_data["predictions"]
        analyzed_data = input_data["analyzed_data"]
        sr_levels = input_data["sr_levels"]

        signals = []
        for pair, pred in predictions.items():
            if pair not in analyzed_data:
                continue

            df = analyzed_data[pair]
            latest = df.iloc[-1]
            levels = sr_levels.get(pair, {"support": [], "resistance": []})

            signal = self._evaluate(pair, pred, latest, levels)
            if signal:
                signals.append(signal)
                self.storage.save_signal(signal)
                self.logger.info(
                    f"{pair}: {signal['signal_type']} signal | "
                    f"confidence={signal['confidence']:.2f} | "
                    f"SL={signal['stop_loss']:.5f} TP={signal['take_profit']:.5f}"
                )

        return {"signals": signals}

    def _evaluate(self, pair: str, pred: dict, latest, levels: dict) -> Optional[dict]:
        score = 0.0
        reasons = []

        # 1. ML prediction direction
        if pred["confidence"] > 0.3:
            score += pred["confidence"] * 2
            reasons.append(f"ML predicts {pred['direction']} ({pred['confidence']:.0%})")

        # 2. RSI confirmation
        if pred["direction"] == "UP" and latest["RSI"] < 40:
            score += 1.0
            reasons.append(f"RSI oversold ({latest['RSI']:.1f})")
        elif pred["direction"] == "DOWN" and latest["RSI"] > 60:
            score += 1.0
            reasons.append(f"RSI overbought ({latest['RSI']:.1f})")

        # 3. MACD confirmation
        if pred["direction"] == "UP" and latest["MACD"] > latest["MACD_signal"]:
            score += 0.8
            reasons.append("MACD bullish crossover")
        elif pred["direction"] == "DOWN" and latest["MACD"] < latest["MACD_signal"]:
            score += 0.8
            reasons.append("MACD bearish crossover")

        # 4. Bollinger Band confirmation
        if pred["direction"] == "UP" and latest["Close"] < latest["BB_lower"]:
            score += 0.7
            reasons.append("Price below lower Bollinger Band")
        elif pred["direction"] == "DOWN" and latest["Close"] > latest["BB_upper"]:
            score += 0.7
            reasons.append("Price above upper Bollinger Band")

        # 5. SMA trend alignment
        if pred["direction"] == "UP" and latest["Close"] > latest["SMA_50"]:
            score += 0.5
            reasons.append("Price above SMA 50 (uptrend)")
        elif pred["direction"] == "DOWN" and latest["Close"] < latest["SMA_50"]:
            score += 0.5
            reasons.append("Price below SMA 50 (downtrend)")

        # 6. Support/Resistance proximity
        current = pred["current_price"]
        for sup in levels.get("support", []):
            if pred["direction"] == "UP" and abs(current - sup) / current < 0.005:
                score += 0.5
                reasons.append(f"Near support level {sup:.5f}")
                break
        for res in levels.get("resistance", []):
            if pred["direction"] == "DOWN" and abs(current - res) / current < 0.005:
                score += 0.5
                reasons.append(f"Near resistance level {res:.5f}")
                break

        # Minimum confluence threshold
        if score < 2.0:
            return None

        signal_type = "BUY" if pred["direction"] == "UP" else "SELL"
        atr = latest["ATR"]

        # Risk management
        if signal_type == "BUY":
            stop_loss = current - (atr * ATR_SL_MULTIPLIER)
            take_profit = current + (atr * ATR_TP_MULTIPLIER)
        else:
            stop_loss = current + (atr * ATR_SL_MULTIPLIER)
            take_profit = current - (atr * ATR_TP_MULTIPLIER)

        risk_per_unit = abs(current - stop_loss)
        max_loss = self.account_size * MAX_RISK_PER_TRADE
        position_size = max_loss / risk_per_unit if risk_per_unit > 0 else 0

        return {
            "pair": pair,
            "signal_type": signal_type,
            "confidence": min(score / 5.0, 0.95),
            "entry_price": current,
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "position_size": round(position_size, 2),
            "reasons": reasons,
            "predicted_price": pred["predicted_price"],
        }
