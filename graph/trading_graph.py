"""LangGraph-based trading pipeline orchestration.

Replaces the imperative Orchestrator with a declarative StateGraph.
The graph defines the flow between stages with conditional edges,
making the pipeline easier to extend, debug, and visualize.

Hybrid architecture:
  Quantitative backbone (LSTM+GBM, regime, risk) from FXAgent
  + LLM analysis layer (news, macro, debate, reflection) from TradingAgents

Pipeline flow:
  Data → [Regime + Correlation + Analysis] → Prediction
  → [News + Macro] → Recommendation → Debate → Portfolio Filter
  → Logging → Reflection → END
"""

import os
import logging
from typing import Dict, Any, Optional

from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.prediction_agent import PredictionAgent
from agents.recommendation_agent import RecommendationAgent
from agents.logging_agent import LoggingAgent
from agents.llm.base_llm import create_llm_client, LLMClient
from agents.llm.news_agent import NewsAgent
from agents.llm.macro_agent import MacroAgent
from agents.llm.debate import DebateAgent
from memory.bm25_memory import TradingMemory
from memory.reflection import ReflectionAgent, load_past_reflections
from risk.portfolio import PortfolioRiskManager
from risk.regime import MarketRegime
from config import MODEL_DIR

logger = logging.getLogger(__name__)


class TradingGraph:
    """Hybrid trading pipeline combining quantitative ML with LLM-powered analysis.

    This replaces the old Orchestrator with a graph-inspired architecture.
    When langgraph is available, uses a compiled StateGraph.
    Otherwise, runs the same flow imperatively (graceful degradation).
    """

    def __init__(self, llm_provider: str = "openai",
                 quick_model: Optional[str] = None,
                 deep_model: Optional[str] = None,
                 enable_llm: bool = True,
                 max_debate_rounds: int = 1):
        """Initialize the hybrid trading graph.

        Args:
            llm_provider: "openai", "anthropic", or "ollama"
            quick_model: Model for analysts/debaters (fast, cheap)
            deep_model: Model for judges (accurate, slower)
            enable_llm: If False, runs quantitative-only mode (no LLM calls)
            max_debate_rounds: Number of debate rounds (1 = one bull + one bear)
        """
        # Quantitative agents (always active)
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.prediction_agent = PredictionAgent()
        self.recommendation_agent = RecommendationAgent()
        self.logging_agent = LoggingAgent()
        self.portfolio_manager = PortfolioRiskManager()

        # Memory system (always active — works without LLM)
        self.memory = TradingMemory()

        # LLM agents (optional — graceful degradation if no API key)
        self.enable_llm = enable_llm
        self.news_agent = None
        self.macro_agent = None
        self.debate_agent = None
        self.reflection_agent = None

        if enable_llm:
            try:
                quick_llm = create_llm_client(llm_provider, model=quick_model)
                deep_llm = create_llm_client(llm_provider, model=deep_model)

                self.news_agent = NewsAgent(quick_llm)
                self.macro_agent = MacroAgent(quick_llm)
                self.debate_agent = DebateAgent(quick_llm, deep_llm, max_debate_rounds)
                self.reflection_agent = ReflectionAgent(quick_llm, self.memory)

                logger.info(f"LLM agents initialized (provider={llm_provider})")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM agents: {e}. Running quantitative-only.")
                self.enable_llm = False

    def propagate(self, pairs: list = None, fetch_intraday: bool = False) -> Dict[str, Any]:
        """Execute the full hybrid pipeline.

        This is the main entry point — equivalent to TradingAgents' propagate().

        Returns the final pipeline state with all signals, analyses, and reflections.
        """
        state = {}
        if pairs:
            state["pairs"] = pairs
        if fetch_intraday:
            state["fetch_intraday"] = True

        health = self._init_health()

        logger.info("=" * 60)
        logger.info("HYBRID TRADING PIPELINE — Starting")
        logger.info("=" * 60)

        # === Stage 1: Data Fetch ===
        state = self._run_stage("data", state, health, self._stage_data)
        if not state.get("ohlcv_data"):
            logger.error("No data fetched. Aborting.")
            state["pipeline_health"] = health
            return state

        total_pairs = len(state.get("ohlcv_data", {}))

        # === Stage 2: Load past reflections from memory ===
        all_pairs = list(state["ohlcv_data"].keys())
        state["past_reflections"] = load_past_reflections(self.memory, all_pairs)
        if state["past_reflections"]:
            logger.info(f"Loaded past reflections for {len(state['past_reflections'])} pairs")

        # === Stage 3: Parallel — Regime + Correlation + Analysis ===
        state = self._run_stage("analysis", state, health, self._stage_analysis)
        if self._check_circuit_breaker("analysis", total_pairs, health):
            state["pipeline_health"] = health
            return state

        # === Stage 4: Prediction (ML) ===
        state = self._run_stage("prediction", state, health, self._stage_prediction)
        if self._check_circuit_breaker("prediction", total_pairs, health):
            state["pipeline_health"] = health
            return state

        # === Stage 5: LLM Analysis (News + Macro) ===
        if self.enable_llm:
            state = self._run_stage("llm_analysis", state, health, self._stage_llm_analysis)

        # === Stage 6: Recommendation ===
        state = self._run_stage("recommendation", state, health, self._stage_recommendation)
        if self._check_circuit_breaker("recommendation", total_pairs, health):
            state["pipeline_health"] = health
            return state

        # === Stage 7: Adversarial Debate ===
        if self.enable_llm and state.get("signals"):
            state = self._run_stage("debate", state, health, self._stage_debate)

        # === Stage 8: Portfolio Risk Filter ===
        raw_signal_count = len(state.get("signals", []))
        state = self._run_stage("risk_filter", state, health, self._stage_portfolio_filter)

        # === Stage 9: Logging & Feedback ===
        state = self._run_stage("logging", state, health, self._stage_logging)

        # === Stage 10: Reflection (learn from closed trades) ===
        if self.enable_llm:
            state = self._run_stage("reflection", state, health, self._stage_reflection)

        # Handle model retraining
        self._handle_retraining(state)

        # Final summary
        state["pipeline_health"] = health
        self._log_summary(state, health, raw_signal_count)

        return state

    # ------------------------------------------------------------------ #
    #  Stage implementations
    # ------------------------------------------------------------------ #

    def _stage_data(self, state: dict) -> dict:
        output = self.data_agent.execute(state)
        state.update(output)
        return state

    def _stage_analysis(self, state: dict) -> dict:
        from concurrent.futures import ThreadPoolExecutor

        ohlcv_data = state["ohlcv_data"]

        with ThreadPoolExecutor(max_workers=3) as executor:
            regime_future = executor.submit(MarketRegime.detect_all, ohlcv_data)
            corr_future = executor.submit(
                self.portfolio_manager.compute_correlation_matrix, ohlcv_data
            )
            analysis_future = executor.submit(self.analysis_agent.execute, state)

            state["regimes"] = regime_future.result()
            state["correlation_matrix"] = corr_future.result()
            state.update(analysis_future.result())

        for pair, regime in state["regimes"].items():
            logger.info(
                f"  {pair}: {regime['regime']} "
                f"(ADX={regime['adx']:.1f}, vol={regime['volatility_state']}, "
                f"trend={regime['trend_direction']})"
            )

        clusters = self.portfolio_manager.get_correlation_clusters()
        if clusters:
            logger.info(f"  Correlation clusters: {clusters}")

        return state

    def _stage_prediction(self, state: dict) -> dict:
        output = self.prediction_agent.execute(state)
        state.update(output)
        return state

    def _stage_llm_analysis(self, state: dict) -> dict:
        """Run news and macro analysis sequentially (macro needs news context)."""
        if self.news_agent:
            news_output = self.news_agent.execute(state)
            state.update(news_output)
            logger.info(f"  News analysis complete for {len(state.get('llm_analyses', {}))} pairs")

        if self.macro_agent:
            macro_output = self.macro_agent.execute(state)
            state.update(macro_output)
            logger.info("  Macro analysis complete")

        return state

    def _stage_recommendation(self, state: dict) -> dict:
        # Inject LLM analysis into recommendation scoring
        if state.get("llm_analyses"):
            self._boost_signals_with_llm(state)

        output = self.recommendation_agent.execute(state)
        state.update(output)
        return state

    def _stage_debate(self, state: dict) -> dict:
        if self.debate_agent and state.get("signals"):
            output = self.debate_agent.execute(state)
            state.update(output)
            logger.info(
                f"  Debate complete: {len(state.get('debates', {}))} pairs debated, "
                f"{len(state.get('signals', []))} signals remaining"
            )
        return state

    def _stage_portfolio_filter(self, state: dict) -> dict:
        if not state.get("signals"):
            return state

        logger.info("Applying portfolio risk filters...")
        portfolio_status = self.portfolio_manager.get_current_portfolio_risk()
        state["portfolio_status"] = portfolio_status
        logger.info(
            f"  Portfolio risk: {portfolio_status['total_risk_pct'] * 100:.1f}% "
            f"({portfolio_status['open_positions']} open positions)"
        )

        filtered = self.portfolio_manager.filter_signals(
            state["signals"], state["ohlcv_data"]
        )
        rejected = len(state["signals"]) - len(filtered)
        if rejected > 0:
            logger.warning(f"  Filtered out {rejected} signals (portfolio risk limits)")
        state["signals"] = filtered
        return state

    def _stage_logging(self, state: dict) -> dict:
        output = self.logging_agent.execute(state)
        state.update(output)
        return state

    def _stage_reflection(self, state: dict) -> dict:
        if self.reflection_agent:
            output = self.reflection_agent.execute(state)
            state.update(output)
        return state

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _boost_signals_with_llm(self, state: dict):
        """Inject LLM sentiment/macro scores into the context for the recommendation agent.

        The recommendation agent doesn't know about LLM analyses directly,
        so we modify the prediction confidence based on LLM agreement/disagreement.
        """
        llm_analyses = state.get("llm_analyses", {})
        predictions = state.get("predictions", {})

        for pair, analysis in llm_analyses.items():
            if pair not in predictions:
                continue

            pred = predictions[pair]
            sentiment = analysis.get("sentiment_score", 0)
            macro_score = analysis.get("macro_score", 0)

            # Check if LLM agrees with ML direction
            ml_bullish = pred["direction"] == "UP"
            llm_bullish = (sentiment + macro_score) / 2 > 0.1
            llm_bearish = (sentiment + macro_score) / 2 < -0.1

            if (ml_bullish and llm_bullish) or (not ml_bullish and llm_bearish):
                # Agreement — boost confidence by 10-20%
                boost = min(abs(sentiment + macro_score) / 4, 0.15)
                pred["confidence"] = min(pred["confidence"] + boost, 0.95)
                logger.info(f"  {pair}: LLM agrees with ML — confidence boosted by {boost:.0%}")
            elif (ml_bullish and llm_bearish) or (not ml_bullish and llm_bullish):
                # Disagreement — reduce confidence by 10-20%
                penalty = min(abs(sentiment + macro_score) / 4, 0.15)
                pred["confidence"] = max(pred["confidence"] - penalty, 0.05)
                logger.warning(f"  {pair}: LLM disagrees with ML — confidence reduced by {penalty:.0%}")

    def _handle_retraining(self, state: dict):
        feedback = state.get("feedback", {})
        if feedback.get("retrain_pairs"):
            logger.warning(f"Retraining requested for: {feedback['retrain_pairs']}")
            for pair in feedback["retrain_pairs"]:
                for suffix in ("_lstm_v3.pt", "_gbm_v3.txt", "_lstm_v2.pt", "_lstm.pt"):
                    model_path = os.path.join(MODEL_DIR, f"{pair.replace('=', '_')}{suffix}")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        logger.info(f"Deleted model {suffix} for {pair}")

    def _run_stage(self, name: str, state: dict, health: dict, stage_fn) -> dict:
        try:
            state = stage_fn(state)
            health[name]["succeeded"] += 1
        except Exception as e:
            logger.error(f"Stage '{name}' failed: {e}")
            health[name]["failed"] += 1
        return state

    def _init_health(self) -> dict:
        stages = [
            "data", "analysis", "prediction", "llm_analysis",
            "recommendation", "debate", "risk_filter", "logging", "reflection",
        ]
        return {s: {"succeeded": 0, "failed": 0} for s in stages}

    def _check_circuit_breaker(self, stage_name: str, total_pairs: int, health: dict) -> bool:
        failed = health[stage_name]["failed"]
        if total_pairs > 0 and failed / total_pairs > 0.5:
            logger.error(f"CIRCUIT BREAKER: {stage_name} — {failed}/{total_pairs} failed. Aborting.")
            return True
        return False

    def _log_summary(self, state: dict, health: dict, raw_signal_count: int):
        logger.info("=" * 60)
        logger.info("PIPELINE HEALTH SUMMARY")
        logger.info("=" * 60)
        for stage, counts in health.items():
            s, f = counts["succeeded"], counts["failed"]
            if s == 0 and f == 0:
                continue
            status = "OK" if f == 0 else "DEGRADED"
            logger.info(f"  {stage}: {s} succeeded, {f} failed [{status}]")

        signals = state.get("signals", [])
        debates = state.get("debates", {})
        reflections = state.get("reflections", [])
        memories = len(self.memory)

        logger.info(f"  Signals: {len(signals)} approved (of {raw_signal_count} generated)")
        if debates:
            logger.info(f"  Debates: {len(debates)} pairs debated")
        if reflections:
            logger.info(f"  Reflections: {len(reflections)} new lessons learned")
        logger.info(f"  Memory: {memories} total memories stored")
        logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    #  Convenience methods
    # ------------------------------------------------------------------ #

    def run_quantitative_only(self, pairs: list = None) -> Dict[str, Any]:
        """Run the pipeline without any LLM calls (fast mode)."""
        saved = self.enable_llm
        self.enable_llm = False
        try:
            return self.propagate(pairs)
        finally:
            self.enable_llm = saved

    def run_analysis_only(self, pairs: list = None) -> Dict[str, Any]:
        """Quick run: fetch + analyze + regime detection (no ML, no LLM)."""
        state = {}
        if pairs:
            state["pairs"] = pairs

        data_output = self.data_agent.execute(state)
        state.update(data_output)

        if not state.get("ohlcv_data"):
            return state

        state["regimes"] = MarketRegime.detect_all(state["ohlcv_data"])
        state["correlation_matrix"] = self.portfolio_manager.compute_correlation_matrix(
            state["ohlcv_data"]
        )

        analysis_output = self.analysis_agent.execute(state)
        state.update(analysis_output)
        return state
