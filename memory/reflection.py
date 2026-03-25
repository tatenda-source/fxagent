"""Post-trade reflection system.

After each trade closes (TP or SL hit), generates an LLM-powered post-mortem
that identifies what worked, what failed, and stores lessons for future trades.

This is the "learning loop" that TradingAgents implements — the system gets
smarter over time by remembering its mistakes and successes.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from agents.base_agent import BaseAgent
from agents.llm.base_llm import LLMClient
from memory.bm25_memory import TradingMemory

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM = """You are a trading performance analyst. Your job is to analyze
a completed trade and extract actionable lessons for future trades.

Be specific and practical. Don't give generic advice like "always use stop losses."
Instead, focus on what THIS specific trade tells us about THIS specific instrument
and market condition.

Your output MUST follow this format:
WHAT_WORKED:
- <specific thing that went right>
WHAT_FAILED:
- <specific thing that went wrong>
KEY_LESSON: <one-sentence actionable lesson for future trades on this pair>
PATTERN_IDENTIFIED: <any recurring pattern you notice, or "none">"""


class ReflectionAgent(BaseAgent):
    """Generates post-trade reflections and stores them in memory."""

    def __init__(self, llm: LLMClient, memory: TradingMemory):
        super().__init__(name="ReflectionAgent")
        self.llm = llm
        self.memory = memory

    def run(self, input_data: dict) -> dict:
        feedback = input_data.get("feedback", {})
        closed_signals = feedback.get("closed_signals", [])
        regimes = input_data.get("regimes", {})
        debates = input_data.get("debates", {})

        if not closed_signals:
            return {"reflections": []}

        reflections = []
        for closed in closed_signals:
            try:
                reflection = self._reflect_on_trade(closed, input_data)
                if reflection:
                    reflections.append(reflection)
            except Exception as e:
                self.logger.error(f"Reflection failed for signal {closed.get('id')}: {e}")

        self.logger.info(f"Generated {len(reflections)} reflections from {len(closed_signals)} closed trades")
        return {"reflections": reflections}

    def _reflect_on_trade(self, closed_signal: dict, context: dict) -> Optional[Dict[str, Any]]:
        """Generate a reflection for a single closed trade."""
        signal_id = closed_signal.get("id")
        outcome = closed_signal.get("status", "UNKNOWN")
        pnl = closed_signal.get("pnl", 0)
        pair = closed_signal.get("pair", "UNKNOWN")

        # Build the situation description from available context
        regime = context.get("regimes", {}).get(pair, {})
        debate = context.get("debates", {}).get(pair, {})
        llm_analysis = context.get("llm_analyses", {}).get(pair, {})

        situation = self._build_situation(pair, closed_signal, regime, debate, llm_analysis)

        # Get past lessons for context
        past = self.memory.get_lessons_for_pair(pair, limit=3)

        prompt = self._build_prompt(situation, outcome, pnl, past)

        try:
            response = self.llm.generate(prompt, system=REFLECTION_SYSTEM, temperature=0.3)
            parsed = self._parse_reflection(response)

            # Store in memory
            self.memory.add_memory(
                pair=pair,
                situation=situation,
                outcome=f"{outcome} (PnL: {pnl:.5f})",
                lessons=parsed.get("key_lesson", response[:200]),
                metadata={
                    "signal_id": signal_id,
                    "regime": regime.get("regime", "unknown"),
                    "pnl": pnl,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            reflection = {
                "pair": pair,
                "signal_id": signal_id,
                "outcome": outcome,
                "pnl": pnl,
                "lessons": parsed.get("key_lesson", ""),
                "what_worked": parsed.get("what_worked", []),
                "what_failed": parsed.get("what_failed", []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.logger.info(
                f"{pair}: {outcome} (PnL={pnl:.5f}) — lesson: {parsed.get('key_lesson', 'N/A')[:80]}"
            )

            return reflection

        except Exception as e:
            self.logger.error(f"LLM reflection failed for {pair}: {e}")
            # Store basic memory even without LLM
            self.memory.add_memory(
                pair=pair,
                situation=situation,
                outcome=f"{outcome} (PnL: {pnl:.5f})",
                lessons=f"Trade {'won' if pnl > 0 else 'lost'} — no detailed analysis available",
                metadata={"signal_id": signal_id, "pnl": pnl,
                          "timestamp": datetime.now(timezone.utc).isoformat()},
            )
            return None

    def _build_situation(self, pair: str, signal: dict, regime: dict,
                         debate: dict, llm_analysis: dict) -> str:
        parts = [f"Trade on {pair}"]

        if regime:
            parts.append(
                f"Regime: {regime.get('regime', '?')}, "
                f"ADX={regime.get('adx', 0):.1f}, "
                f"trend={regime.get('trend_direction', '?')}"
            )

        if debate:
            parts.append(f"Debate verdict: {debate.get('conviction', '?')}")

        if llm_analysis:
            parts.append(f"News sentiment: {llm_analysis.get('sentiment_score', 0):.2f}")
            parts.append(f"Macro bias: {llm_analysis.get('fundamental_bias', '?')}")

        return ". ".join(parts)

    def _build_prompt(self, situation: str, outcome: str, pnl: float,
                      past_lessons: List[str]) -> str:
        parts = [
            f"TRADE SITUATION:\n{situation}\n",
            f"OUTCOME: {outcome}",
            f"PnL: {pnl:.5f}",
        ]

        if past_lessons:
            parts.append("\nPAST LESSONS FOR THIS PAIR:")
            for lesson in past_lessons:
                parts.append(f"- {lesson}")

        parts.append(
            "\nAnalyze this trade. What can we learn for next time? "
            "Are there any patterns emerging from the past lessons?"
        )

        return "\n".join(parts)

    def _parse_reflection(self, response: str) -> Dict[str, Any]:
        result = {
            "what_worked": [],
            "what_failed": [],
            "key_lesson": "",
            "pattern_identified": "",
        }

        current_section = None
        for line in response.split("\n"):
            line = line.strip()

            if "WHAT_WORKED:" in line:
                current_section = "worked"
            elif "WHAT_FAILED:" in line:
                current_section = "failed"
            elif line.startswith("KEY_LESSON:"):
                result["key_lesson"] = line.split(":", 1)[1].strip()
                current_section = None
            elif line.startswith("PATTERN_IDENTIFIED:"):
                result["pattern_identified"] = line.split(":", 1)[1].strip()
                current_section = None
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "worked":
                    result["what_worked"].append(item)
                elif current_section == "failed":
                    result["what_failed"].append(item)

        return result


def load_past_reflections(memory: TradingMemory, pairs: list) -> Dict[str, List[str]]:
    """Load relevant past lessons for each pair. Used at pipeline start."""
    reflections = {}
    for pair in pairs:
        lessons = memory.get_lessons_for_pair(pair, limit=5)
        if lessons:
            reflections[pair] = lessons
    return reflections
