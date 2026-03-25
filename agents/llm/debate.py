"""Adversarial debate system for trading decisions.

Implements TradingAgents' key innovation: Bull vs Bear researchers debate
each trade, then a Judge (Research Manager) makes the final call.

This catches blind spots that a single-path pipeline misses. A signal
that survives adversarial scrutiny is more robust.

Flow:
  1. Bull Researcher: Makes the strongest case FOR the trade
  2. Bear Researcher: Makes the strongest case AGAINST the trade
  3. (Optional rounds of rebuttal)
  4. Judge: Weighs both arguments and renders a verdict
"""

import logging
from typing import Dict, Optional, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base_agent import BaseAgent
from agents.llm.base_llm import LLMClient

logger = logging.getLogger(__name__)


BULL_SYSTEM = """You are a senior bull researcher at a trading firm. Your job is to make
the STRONGEST possible case FOR taking this trade. You must be specific, cite the data
provided, and address potential counterarguments preemptively.

Be aggressive but honest. If the data genuinely doesn't support the trade, acknowledge
weaknesses but argue why they're manageable. Think like a conviction-driven fund manager
who needs to convince the risk committee.

Structure your argument:
1. THESIS: One-sentence bull case
2. SUPPORTING EVIDENCE: 3-5 specific data points
3. RISK MITIGATION: Why the risks are manageable
4. EXPECTED OUTCOME: Specific price target / return expectation"""

BEAR_SYSTEM = """You are a senior bear researcher at a trading firm. Your job is to make
the STRONGEST possible case AGAINST taking this trade. You must be specific, cite the data
provided, and poke holes in every bullish argument.

Be ruthless but honest. If the data genuinely supports the trade, acknowledge strengths
but argue why they're insufficient or misleading. Think like a risk manager who needs to
protect the firm's capital.

Structure your argument:
1. COUNTER-THESIS: One-sentence bear case
2. RED FLAGS: 3-5 specific concerns from the data
3. HIDDEN RISKS: What the bull case is ignoring
4. WORST CASE: What happens if this trade goes wrong"""

JUDGE_SYSTEM = """You are the Research Manager at a trading firm. You have just read a debate
between your Bull and Bear researchers about a proposed trade.

Your job is to render a FINAL VERDICT. You are not trying to be balanced — you are trying
to be CORRECT. Sometimes the bull is right, sometimes the bear is right.

Consider:
- Quality of arguments on each side
- Which side's evidence is more compelling
- Whether the risk/reward is asymmetric in either direction
- Whether the ML signals and fundamental analysis agree or disagree

You MUST output your verdict in this exact format:
CONVICTION: <strong_buy/buy/hold/sell/strong_sell>
CONFIDENCE: <0.0 to 1.0>
REASONING: <2-3 sentences explaining your decision>
KEY_RISK: <the single biggest risk to this trade>"""


class DebateAgent(BaseAgent):
    """Runs adversarial bull/bear debates for each signal, judged by a research manager.

    Uses two LLM tiers:
    - quick_llm: For bull/bear researchers (speed)
    - deep_llm: For the judge (accuracy)
    """

    def __init__(self, quick_llm: LLMClient, deep_llm: LLMClient,
                 max_debate_rounds: int = 1):
        super().__init__(name="DebateAgent")
        self.quick_llm = quick_llm
        self.deep_llm = deep_llm
        self.max_debate_rounds = max_debate_rounds

    def run(self, input_data: dict) -> dict:
        predictions = input_data.get("predictions", {})
        signals = input_data.get("signals", [])
        llm_analyses = input_data.get("llm_analyses", {})
        regimes = input_data.get("regimes", {})
        past_reflections = input_data.get("past_reflections", {})

        # Only debate pairs that have generated signals
        signal_pairs = {s["pair"] for s in signals}
        debates = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for pair in signal_pairs:
                if pair not in predictions:
                    continue
                context = {
                    "prediction": predictions[pair],
                    "signal": next(s for s in signals if s["pair"] == pair),
                    "llm_analysis": llm_analyses.get(pair, {}),
                    "regime": regimes.get(pair, {}),
                    "past_lessons": past_reflections.get(pair, []),
                }
                futures[executor.submit(self._debate_pair, pair, context)] = pair

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    if result:
                        debates[pair] = result
                except Exception as e:
                    self.logger.error(f"Debate failed for {pair}: {e}")

        # Filter signals based on debate outcomes
        filtered_signals = self._apply_debate_filter(signals, debates)

        return {"debates": debates, "signals": filtered_signals}

    def _debate_pair(self, pair: str, context: dict) -> Optional[Dict[str, Any]]:
        """Run a full bull/bear/judge debate for one pair."""
        brief = self._build_brief(pair, context)

        # Round 1: Bull makes case
        bull_prompt = f"Here is the trade proposal:\n\n{brief}\n\nMake your bull case."
        bull_case = self.quick_llm.generate(bull_prompt, system=BULL_SYSTEM, temperature=0.4)

        # Round 1: Bear makes countercase
        bear_prompt = (
            f"Here is the trade proposal:\n\n{brief}\n\n"
            f"The Bull Researcher argues:\n{bull_case}\n\n"
            f"Make your bear case. Address the bull's specific arguments."
        )
        bear_case = self.quick_llm.generate(bear_prompt, system=BEAR_SYSTEM, temperature=0.4)

        # Additional rebuttal rounds
        for _ in range(self.max_debate_rounds - 1):
            # Bull rebuttal
            bull_rebuttal_prompt = (
                f"The Bear counters:\n{bear_case}\n\n"
                f"Respond to the bear's specific points. Defend your thesis."
            )
            bull_case += "\n\nREBUTTAL:\n" + self.quick_llm.generate(
                bull_rebuttal_prompt, system=BULL_SYSTEM, temperature=0.4
            )

            # Bear rebuttal
            bear_rebuttal_prompt = (
                f"The Bull responds:\n{bull_case}\n\n"
                f"Counter the bull's rebuttal. Press on the weakest points."
            )
            bear_case += "\n\nREBUTTAL:\n" + self.quick_llm.generate(
                bear_rebuttal_prompt, system=BEAR_SYSTEM, temperature=0.4
            )

        # Judge renders verdict (using deep_think LLM)
        judge_prompt = (
            f"TRADE PROPOSAL:\n{brief}\n\n"
            f"BULL CASE:\n{bull_case}\n\n"
            f"BEAR CASE:\n{bear_case}\n\n"
            f"Render your verdict."
        )
        verdict_raw = self.deep_llm.generate(judge_prompt, system=JUDGE_SYSTEM, temperature=0.1)
        verdict = self._parse_verdict(verdict_raw)

        self.logger.info(
            f"{pair}: debate result = {verdict.get('conviction', 'hold')} "
            f"(confidence={verdict.get('confidence', 0):.2f})"
        )

        return {
            "bull_case": bull_case,
            "bear_case": bear_case,
            "judge_verdict": verdict_raw,
            "debate_rounds": self.max_debate_rounds,
            "conviction": verdict.get("conviction", "hold"),
            "confidence": verdict.get("confidence", 0.5),
            "reasoning": verdict.get("reasoning", ""),
            "key_risk": verdict.get("key_risk", ""),
        }

    def _build_brief(self, pair: str, context: dict) -> str:
        """Build the trade brief that both bull and bear will analyze."""
        signal = context["signal"]
        pred = context["prediction"]
        regime = context.get("regime", {})
        llm = context.get("llm_analysis", {})
        lessons = context.get("past_lessons", [])

        parts = [
            f"PAIR: {pair}",
            f"PROPOSED ACTION: {signal['signal_type']}",
            f"ENTRY: {signal['entry_price']:.5f}",
            f"STOP LOSS: {signal['stop_loss']:.5f}",
            f"TAKE PROFIT: {signal['take_profit']:.5f}",
            f"POSITION SIZE: {signal['position_size']:.2f}",
            f"CONFIDENCE: {signal['confidence']:.2f}",
            "",
            f"ML PREDICTION: {pred['direction']} (confidence={pred['confidence']:.0%})",
            f"PREDICTED RETURN: {pred.get('predicted_return', 0):.4%}",
            f"MODEL UNCERTAINTY: {pred.get('uncertainty', 0):.6f}",
            "",
            f"REGIME: {regime.get('regime', 'unknown')}",
            f"ADX: {regime.get('adx', 0):.1f}",
            f"TREND: {regime.get('trend_direction', 'flat')}",
            f"VOLATILITY: {regime.get('volatility_state', 'normal')}",
            "",
            f"SIGNAL REASONS: {', '.join(signal.get('reasons', []))}",
        ]

        if llm:
            parts.append(f"\nNEWS SENTIMENT: {llm.get('sentiment_score', 0):.2f}")
            parts.append(f"MACRO SCORE: {llm.get('macro_score', 0):.2f}")
            parts.append(f"FUNDAMENTAL BIAS: {llm.get('fundamental_bias', 'neutral')}")
            events = llm.get("key_events", [])
            if events:
                parts.append("UPCOMING EVENTS:")
                for e in events[:3]:
                    parts.append(f"  - {e}")
            risks = llm.get("risk_factors", [])
            if risks:
                parts.append("IDENTIFIED RISKS:")
                for r in risks[:3]:
                    parts.append(f"  - {r}")

        if lessons:
            parts.append("\nLESSONS FROM PAST TRADES ON THIS PAIR:")
            for l in lessons[-3:]:
                parts.append(f"  - {l}")

        return "\n".join(parts)

    def _parse_verdict(self, verdict_raw: str) -> Dict[str, Any]:
        result = {
            "conviction": "hold",
            "confidence": 0.5,
            "reasoning": "",
            "key_risk": "",
        }

        for line in verdict_raw.split("\n"):
            line = line.strip()
            if line.startswith("CONVICTION:"):
                conv = line.split(":", 1)[1].strip().lower()
                valid = ("strong_buy", "buy", "hold", "sell", "strong_sell")
                if conv in valid:
                    result["conviction"] = conv
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_RISK:"):
                result["key_risk"] = line.split(":", 1)[1].strip()

        return result

    def _apply_debate_filter(self, signals: List[Dict], debates: Dict[str, Dict]) -> List[Dict]:
        """Remove or downgrade signals based on debate outcomes."""
        filtered = []
        for signal in signals:
            pair = signal["pair"]
            debate = debates.get(pair)

            if debate is None:
                # No debate happened — keep signal as-is
                filtered.append(signal)
                continue

            conviction = debate.get("conviction", "hold")

            # Kill signals where the judge says sell/strong_sell
            if signal["signal_type"] == "BUY" and conviction in ("sell", "strong_sell"):
                self.logger.warning(
                    f"DEBATE VETOED {pair} BUY: judge says {conviction}. "
                    f"Reason: {debate.get('reasoning', 'N/A')}"
                )
                continue
            if signal["signal_type"] == "SELL" and conviction in ("buy", "strong_buy"):
                self.logger.warning(
                    f"DEBATE VETOED {pair} SELL: judge says {conviction}. "
                    f"Reason: {debate.get('reasoning', 'N/A')}"
                )
                continue

            # Hold = reduce position size by 50%
            if conviction == "hold":
                signal = signal.copy()
                signal["position_size"] = round(signal["position_size"] * 0.5, 2)
                signal["reasons"] = signal["reasons"] + [f"Debate: reduced size (judge={conviction})"]
                self.logger.info(f"{pair}: debate judge says hold — halving position")

            # Strong conviction = boost confidence
            if conviction in ("strong_buy", "strong_sell"):
                signal = signal.copy()
                signal["confidence"] = min(signal["confidence"] * 1.2, 0.95)
                signal["reasons"] = signal["reasons"] + [f"Debate: high conviction ({conviction})"]

            filtered.append(signal)

        vetoed = len(signals) - len(filtered)
        if vetoed > 0:
            self.logger.info(f"Debate vetoed {vetoed} of {len(signals)} signals")

        return filtered
