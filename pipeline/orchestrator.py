"""Pipeline orchestrator — now delegates to the hybrid TradingGraph.

The Orchestrator class is preserved for backward compatibility.
New code should use TradingGraph directly.
"""

from loguru import logger

from graph.trading_graph import TradingGraph
from config import (
    LLM_PROVIDER, LLM_QUICK_MODEL, LLM_DEEP_MODEL,
    ENABLE_LLM, MAX_DEBATE_ROUNDS,
)


class Orchestrator:
    """Coordinates all agents + risk modules in the hybrid pipeline.

    This is a thin wrapper around TradingGraph for backward compatibility.
    The TradingGraph handles all orchestration including the new LLM agents.
    """

    def __init__(self, enable_llm: bool = None):
        llm_enabled = enable_llm if enable_llm is not None else ENABLE_LLM
        self.graph = TradingGraph(
            llm_provider=LLM_PROVIDER,
            quick_model=LLM_QUICK_MODEL,
            deep_model=LLM_DEEP_MODEL,
            enable_llm=llm_enabled,
            max_debate_rounds=MAX_DEBATE_ROUNDS,
        )

    def run_full_pipeline(self, pairs: list = None, fetch_intraday: bool = False) -> dict:
        """Execute the complete hybrid pipeline. Returns merged context."""
        return self.graph.propagate(pairs, fetch_intraday)

    def run_analysis_only(self, pairs: list = None) -> dict:
        """Quick run: fetch + analyze + regime detection (no ML, no LLM)."""
        return self.graph.run_analysis_only(pairs)

    def run_quantitative_only(self, pairs: list = None) -> dict:
        """Run without LLM calls — pure ML + technical pipeline."""
        return self.graph.run_quantitative_only(pairs)
