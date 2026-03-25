"""LLM-powered macro/fundamental analysis agent.

Analyzes macroeconomic context for FX pairs and metals:
- Central bank policy divergence
- Interest rate differentials
- Economic calendar events
- Cross-asset correlations (DXY, yields, equities)

For FX: focuses on monetary policy, rate expectations, economic data
For metals: focuses on real yields, USD strength, industrial demand
"""

import logging
from typing import Dict, Optional, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base_agent import BaseAgent
from agents.llm.base_llm import LLMClient

logger = logging.getLogger(__name__)

# Macro context for each pair
PAIR_MACRO_CONTEXT = {
    "EURUSD=X": {
        "base": "EUR", "quote": "USD",
        "central_banks": "ECB vs Federal Reserve",
        "key_data": "NFP, CPI, PMI, ECB/Fed rate decisions, German Ifo",
        "drivers": "Rate differential, EU-US growth divergence, risk sentiment",
    },
    "GBPUSD=X": {
        "base": "GBP", "quote": "USD",
        "central_banks": "Bank of England vs Federal Reserve",
        "key_data": "UK CPI, employment, BoE/Fed decisions, UK GDP",
        "drivers": "Rate expectations, Brexit effects, UK fiscal policy",
    },
    "USDJPY=X": {
        "base": "USD", "quote": "JPY",
        "central_banks": "Federal Reserve vs Bank of Japan",
        "key_data": "US yields, BoJ policy, Japan CPI, US NFP",
        "drivers": "US-Japan rate gap, BoJ yield curve control, risk-on/off",
    },
    "AUDUSD=X": {
        "base": "AUD", "quote": "USD",
        "central_banks": "RBA vs Federal Reserve",
        "key_data": "Australia employment, RBA decisions, China PMI, iron ore",
        "drivers": "Commodity prices, China outlook, risk appetite",
    },
    "USDCAD=X": {
        "base": "USD", "quote": "CAD",
        "central_banks": "Federal Reserve vs Bank of Canada",
        "key_data": "Oil prices, BoC decisions, Canada employment, US NFP",
        "drivers": "Oil prices, US-Canada rate spread, trade balance",
    },
    "USDCHF=X": {
        "base": "USD", "quote": "CHF",
        "central_banks": "Federal Reserve vs Swiss National Bank",
        "key_data": "SNB policy, Swiss CPI, safe-haven flows, US data",
        "drivers": "Risk sentiment, SNB intervention, gold correlation",
    },
    "GC=F": {
        "asset": "Gold",
        "key_data": "Real yields, DXY, Fed policy, geopolitical risk",
        "drivers": "Real interest rates (inverse), USD strength (inverse), safe-haven demand",
    },
    "SI=F": {
        "asset": "Silver",
        "key_data": "Gold price, industrial demand, solar panel demand",
        "drivers": "Gold correlation, industrial/green energy demand, USD",
    },
    "PL=F": {
        "asset": "Platinum",
        "key_data": "Auto sector demand, hydrogen economy, supply from SA/Russia",
        "drivers": "Auto catalytic converter demand, mining supply disruptions",
    },
    "PA=F": {
        "asset": "Palladium",
        "key_data": "Auto sector, Russia supply (40%), EV transition threat",
        "drivers": "Gasoline auto demand, Russian supply risk, EV substitution",
    },
    "HG=F": {
        "asset": "Copper",
        "key_data": "China PMI, construction, EV/electrification demand",
        "drivers": "China industrial demand, global construction, green transition",
    },
}

SYSTEM_PROMPT = """You are a senior macro strategist at a global trading firm specializing in FX and commodities.

Your job is to analyze the macroeconomic backdrop for a specific instrument and assess whether
the fundamental picture supports or contradicts the technical/ML signals.

Be SPECIFIC. Reference actual policy rates, recent data prints, and upcoming events.
Do NOT give generic advice. Think like a prop desk macro analyst.

Your report MUST include:
1. MACRO_SCORE: A number from -1.0 (strong fundamental headwind) to 1.0 (strong fundamental tailwind)
2. KEY_DRIVERS: The 2-3 most important macro factors right now
3. UPCOMING_RISKS: Events in the next 1-2 weeks that could invalidate the trade
4. FUNDAMENTAL_BIAS: bullish/bearish/neutral
5. CONVICTION: high/medium/low — how confident are you in the fundamental picture

Format:
MACRO_SCORE: <number>
KEY_DRIVERS:
- <driver 1>
- <driver 2>
UPCOMING_RISKS:
- <risk 1>
- <risk 2>
FUNDAMENTAL_BIAS: <bullish/bearish/neutral>
CONVICTION: <high/medium/low>
ANALYSIS:
<your detailed macro analysis>"""


class MacroAgent(BaseAgent):
    """Analyzes macroeconomic fundamentals for each tradeable pair using LLM."""

    def __init__(self, llm: LLMClient):
        super().__init__(name="MacroAgent")
        self.llm = llm

    def run(self, input_data: dict) -> dict:
        pairs = list(input_data.get("predictions", {}).keys())
        if not pairs:
            pairs = list(input_data.get("ohlcv_data", {}).keys())

        analyses = input_data.get("llm_analyses", {})

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._analyze_pair, pair, input_data): pair
                for pair in pairs
            }
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    if result:
                        # Merge with existing news analysis
                        if pair in analyses:
                            analyses[pair]["macro_report"] = result.get("macro_report", "")
                            analyses[pair]["macro_score"] = result.get("macro_score", 0.0)
                            analyses[pair]["fundamental_bias"] = result.get("fundamental_bias", "neutral")
                        else:
                            analyses[pair] = result
                except Exception as e:
                    self.logger.error(f"Macro analysis failed for {pair}: {e}")

        return {"llm_analyses": analyses}

    def _analyze_pair(self, pair: str, context: dict) -> Optional[Dict[str, Any]]:
        macro_ctx = PAIR_MACRO_CONTEXT.get(pair, {})
        if not macro_ctx:
            return None

        regime = context.get("regimes", {}).get(pair, {})
        prediction = context.get("predictions", {}).get(pair, {})
        news_analysis = context.get("llm_analyses", {}).get(pair, {})

        prompt = self._build_prompt(pair, macro_ctx, regime, prediction, news_analysis)

        try:
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.2)
            parsed = self._parse_response(response)
            self.logger.info(
                f"{pair}: macro_score={parsed.get('macro_score', 0):.2f}, "
                f"bias={parsed.get('fundamental_bias', 'neutral')}, "
                f"conviction={parsed.get('conviction', 'low')}"
            )
            return parsed
        except Exception as e:
            self.logger.error(f"LLM macro analysis failed for {pair}: {e}")
            return None

    def _build_prompt(self, pair: str, macro_ctx: dict, regime: dict,
                      prediction: dict, news_analysis: dict) -> str:
        parts = [f"Analyze the macroeconomic outlook for {pair}.\n"]

        parts.append("INSTRUMENT CONTEXT:")
        for k, v in macro_ctx.items():
            parts.append(f"  {k}: {v}")
        parts.append("")

        if regime:
            parts.append(
                f"TECHNICAL REGIME: {regime.get('regime', 'unknown')}, "
                f"trend={regime.get('trend_direction', 'flat')}"
            )

        if prediction:
            parts.append(
                f"ML SIGNAL: {prediction.get('direction', '?')} "
                f"(confidence={prediction.get('confidence', 0):.0%})"
            )

        if news_analysis:
            sentiment = news_analysis.get("sentiment_score", 0)
            parts.append(f"NEWS SENTIMENT: {sentiment:.2f}")
            events = news_analysis.get("key_events", [])
            if events:
                parts.append("RECENT NEWS EVENTS:")
                for e in events[:5]:
                    parts.append(f"  - {e}")

        parts.append(
            "\nAssess the fundamental picture. Does it support or contradict "
            "the technical/ML signals? What macro risks should we be aware of?"
        )

        return "\n".join(parts)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        result = {
            "macro_report": response,
            "macro_score": 0.0,
            "fundamental_bias": "neutral",
            "conviction": "low",
            "key_drivers": [],
            "upcoming_risks": [],
        }

        sections = response.split("\n")
        current_section = None

        for line in sections:
            line = line.strip()

            if line.startswith("MACRO_SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    result["macro_score"] = max(-1.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("FUNDAMENTAL_BIAS:"):
                bias = line.split(":", 1)[1].strip().lower()
                if bias in ("bullish", "bearish", "neutral"):
                    result["fundamental_bias"] = bias
            elif line.startswith("CONVICTION:"):
                conv = line.split(":", 1)[1].strip().lower()
                if conv in ("high", "medium", "low"):
                    result["conviction"] = conv
            elif "KEY_DRIVERS:" in line:
                current_section = "drivers"
            elif "UPCOMING_RISKS:" in line:
                current_section = "risks"
            elif "ANALYSIS:" in line or "FUNDAMENTAL_BIAS:" in line:
                current_section = None
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "drivers":
                    result["key_drivers"].append(item)
                elif current_section == "risks":
                    result["upcoming_risks"].append(item)

        return result
