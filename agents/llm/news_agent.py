"""LLM-powered news and sentiment analysis agent.

Fetches recent news for FX pairs and metals, then uses an LLM to interpret
the sentiment and identify market-moving events. This fills a critical gap
in the original pipeline which was purely technical.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base_agent import BaseAgent
from agents.llm.base_llm import LLMClient

logger = logging.getLogger(__name__)

# Map Yahoo tickers to search-friendly names
PAIR_NEWS_NAMES = {
    "EURUSD=X": "EUR/USD euro dollar",
    "GBPUSD=X": "GBP/USD british pound",
    "USDJPY=X": "USD/JPY japanese yen",
    "AUDUSD=X": "AUD/USD australian dollar",
    "USDCAD=X": "USD/CAD canadian dollar",
    "USDCHF=X": "USD/CHF swiss franc",
    "GC=F": "gold XAU/USD",
    "SI=F": "silver XAG/USD",
    "PL=F": "platinum",
    "PA=F": "palladium",
    "HG=F": "copper",
}

SYSTEM_PROMPT = """You are a senior FX and commodities market analyst at a trading firm.
Your job is to analyze recent news and economic events for a specific currency pair or
commodity and produce a concise, actionable report.

You must be specific and quantitative where possible. Avoid vague statements.
Focus on what matters for SHORT-TERM (1-5 day) directional trading.

Your report MUST include:
1. SENTIMENT SCORE: A number from -1.0 (extremely bearish) to 1.0 (extremely bullish)
2. KEY EVENTS: Bullet list of upcoming or recent events that could move this pair
3. RISK FACTORS: What could go wrong with a trade in either direction
4. DIRECTIONAL BIAS: Your overall view (bullish/bearish/neutral) with reasoning

Format your response as:
SENTIMENT_SCORE: <number>
KEY_EVENTS:
- <event 1>
- <event 2>
RISK_FACTORS:
- <risk 1>
- <risk 2>
ANALYSIS:
<your detailed analysis paragraph>
DIRECTIONAL_BIAS: <bullish/bearish/neutral>"""


class NewsAgent(BaseAgent):
    """Fetches news and uses LLM to analyze sentiment for each tradeable pair."""

    def __init__(self, llm: LLMClient):
        super().__init__(name="NewsAgent")
        self.llm = llm

    def run(self, input_data: dict) -> dict:
        pairs = list(input_data.get("predictions", {}).keys())
        if not pairs:
            pairs = list(input_data.get("ohlcv_data", {}).keys())

        analyses = {}

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
                        analyses[pair] = result
                except Exception as e:
                    self.logger.error(f"News analysis failed for {pair}: {e}")

        return {"llm_analyses": analyses}

    def _analyze_pair(self, pair: str, context: dict) -> Optional[Dict[str, Any]]:
        """Analyze a single pair using news data + LLM interpretation."""
        news_name = PAIR_NEWS_NAMES.get(pair, pair)

        # Gather context from the quantitative pipeline
        regime = context.get("regimes", {}).get(pair, {})
        prediction = context.get("predictions", {}).get(pair, {})
        past_lessons = context.get("past_reflections", {}).get(pair, [])

        # Fetch news via yfinance
        news_items = self._fetch_news(pair)
        news_text = self._format_news(news_items)

        # Build the prompt with quantitative context
        prompt = self._build_prompt(pair, news_name, news_text, regime, prediction, past_lessons)

        try:
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.2)
            parsed = self._parse_response(response)
            self.logger.info(
                f"{pair}: sentiment={parsed.get('sentiment_score', 0):.2f}, "
                f"bias={parsed.get('directional_bias', 'unknown')}"
            )
            return parsed
        except Exception as e:
            self.logger.error(f"LLM analysis failed for {pair}: {e}")
            return None

    def _fetch_news(self, pair: str) -> List[Dict]:
        """Fetch recent news for a pair via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(pair)
            news = ticker.news or []
            return news[:10]  # Last 10 articles
        except Exception as e:
            self.logger.warning(f"Failed to fetch news for {pair}: {e}")
            return []

    def _format_news(self, news_items: List[Dict]) -> str:
        if not news_items:
            return "No recent news available."

        lines = []
        for item in news_items:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            lines.append(f"- [{publisher}] {title}")
        return "\n".join(lines)

    def _build_prompt(self, pair: str, news_name: str, news_text: str,
                      regime: dict, prediction: dict, past_lessons: List[str]) -> str:
        parts = [f"Analyze the current outlook for {news_name} ({pair}).\n"]

        parts.append(f"RECENT NEWS:\n{news_text}\n")

        if regime:
            parts.append(
                f"CURRENT REGIME: {regime.get('regime', 'unknown')}, "
                f"ADX={regime.get('adx', 0):.1f}, "
                f"trend={regime.get('trend_direction', 'flat')}, "
                f"volatility={regime.get('volatility_state', 'normal')}\n"
            )

        if prediction:
            parts.append(
                f"ML PREDICTION: {prediction.get('direction', '?')} "
                f"(confidence={prediction.get('confidence', 0):.0%}, "
                f"predicted_return={prediction.get('predicted_return', 0):.4%})\n"
            )

        if past_lessons:
            parts.append("LESSONS FROM PAST TRADES:\n")
            for lesson in past_lessons[-3:]:  # Last 3 lessons
                parts.append(f"- {lesson}\n")

        parts.append(
            "\nProvide your analysis considering both the news/fundamental backdrop "
            "and the quantitative signals above. Where do they agree or disagree?"
        )

        return "\n".join(parts)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured LLM response into a dict."""
        result = {
            "news_report": response,
            "sentiment_score": 0.0,
            "key_events": [],
            "risk_factors": [],
            "directional_bias": "neutral",
        }

        for line in response.split("\n"):
            line = line.strip()

            if line.startswith("SENTIMENT_SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    result["sentiment_score"] = max(-1.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass

            elif line.startswith("DIRECTIONAL_BIAS:"):
                bias = line.split(":", 1)[1].strip().lower()
                if bias in ("bullish", "bearish", "neutral"):
                    result["directional_bias"] = bias

            elif line.startswith("- "):
                # Collect bullet points — we'll assign them based on section
                pass

        # Extract key events and risk factors from sections
        sections = response.split("\n")
        current_section = None
        for line in sections:
            line = line.strip()
            if "KEY_EVENTS:" in line:
                current_section = "events"
            elif "RISK_FACTORS:" in line:
                current_section = "risks"
            elif "ANALYSIS:" in line or "DIRECTIONAL_BIAS:" in line or "SENTIMENT_SCORE:" in line:
                current_section = None
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "events":
                    result["key_events"].append(item)
                elif current_section == "risks":
                    result["risk_factors"].append(item)

        return result
