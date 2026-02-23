from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base_agent import BaseAgent
from data.storage import Storage
from config import (
    MAX_RISK_PER_TRADE, DEFAULT_ACCOUNT_SIZE,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
)


class RecommendationAgent(BaseAgent):
    """Agent 4: Generates buy/sell signals with SL, TP, and position sizing.
    Now regime-aware — adjusts parameters based on market conditions.
    Evaluates signals in parallel, then batch-writes to DB."""

    def __init__(self, account_size: float = DEFAULT_ACCOUNT_SIZE):
        super().__init__(name="RecommendationAgent")
        self.storage = Storage()
        self.account_size = account_size

    def run(self, input_data: dict) -> dict:
        predictions = input_data["predictions"]
        analyzed_data = input_data["analyzed_data"]
        sr_levels = input_data["sr_levels"]
        regimes = input_data.get("regimes", {})

        # Evaluate signals in parallel
        signals = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for pair, pred in predictions.items():
                if pair not in analyzed_data:
                    continue
                df = analyzed_data[pair]
                latest = df.iloc[-1]
                levels = sr_levels.get(pair, {"support": [], "resistance": []})
                regime = regimes.get(pair, {})
                future = executor.submit(self._evaluate, pair, pred, latest, levels, regime)
                futures[future] = pair

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    signal = future.result()
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    self.logger.error(f"Signal evaluation failed for {pair}: {e}")

        # Batch write signals to DB
        for signal in signals:
            self.storage.save_signal(signal)
            regime = regimes.get(signal["pair"], {})
            self.logger.info(
                f"{signal['pair']}: {signal['signal_type']} signal | "
                f"confidence={signal['confidence']:.2f} | "
                f"SL={signal['stop_loss']:.5f} TP={signal['take_profit']:.5f} | "
                f"regime={regime.get('regime', 'unknown')}"
            )

        return {"signals": signals}

    def _evaluate(self, pair: str, pred: dict, latest, levels: dict,
                  regime: dict) -> Optional[dict]:
        score = 0.0
        reasons = []

        # Get regime adjustments (default to neutral if no regime data)
        adjustments = regime.get("strategy_adjustments", {})
        min_score_adj = adjustments.get("min_score_adj", 0.0)
        sl_mult_adj = adjustments.get("sl_multiplier_adj", 1.0)
        tp_mult_adj = adjustments.get("tp_multiplier_adj", 1.0)
        size_adj = adjustments.get("position_size_adj", 1.0)

        # Tag regime in reasons
        regime_name = regime.get("regime", "unknown")
        if regime_name != "unknown":
            reasons.append(f"Regime: {regime_name} (ADX={regime.get('adx', 0):.1f})")

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

        # 7. Regime-specific bonus: trend alignment in trending markets
        if regime_name == "trending":
            trend_dir = regime.get("trend_direction", "flat")
            if (pred["direction"] == "UP" and trend_dir == "up") or \
               (pred["direction"] == "DOWN" and trend_dir == "down"):
                score += 0.5
                reasons.append(f"Aligned with {trend_dir}trend (ADX={regime.get('adx', 0):.0f})")

        # Minimum confluence threshold — adjusted by regime
        min_score = 2.0 + min_score_adj
        if score < min_score:
            return None

        signal_type = "BUY" if pred["direction"] == "UP" else "SELL"
        atr = latest["ATR"]

        # Apply regime-adjusted SL/TP multipliers
        effective_sl_mult = ATR_SL_MULTIPLIER * sl_mult_adj
        effective_tp_mult = ATR_TP_MULTIPLIER * tp_mult_adj

        if signal_type == "BUY":
            stop_loss = current - (atr * effective_sl_mult)
            take_profit = current + (atr * effective_tp_mult)
        else:
            stop_loss = current + (atr * effective_sl_mult)
            take_profit = current - (atr * effective_tp_mult)

        risk_per_unit = abs(current - stop_loss)
        max_loss = self.account_size * MAX_RISK_PER_TRADE
        position_size = max_loss / risk_per_unit if risk_per_unit > 0 else 0

        # Apply regime position size adjustment
        position_size *= size_adj

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
            "regime": regime_name,
        }
