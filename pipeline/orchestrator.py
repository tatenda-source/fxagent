import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.prediction_agent import PredictionAgent
from agents.recommendation_agent import RecommendationAgent
from agents.logging_agent import LoggingAgent
from risk.portfolio import PortfolioRiskManager
from risk.regime import MarketRegime
from config import MODEL_DIR


class Orchestrator:
    """Coordinates all agents + risk modules in a sequential pipeline."""

    def __init__(self):
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.prediction_agent = PredictionAgent()
        self.recommendation_agent = RecommendationAgent()
        self.logging_agent = LoggingAgent()
        self.portfolio_manager = PortfolioRiskManager()

    def run_full_pipeline(self, pairs: list = None, fetch_intraday: bool = False) -> dict:
        """Execute the complete agent pipeline with risk management. Returns merged context."""
        context = {}
        if pairs:
            context["pairs"] = pairs
        if fetch_intraday:
            context["fetch_intraday"] = True

        logger.info("=== Starting full pipeline ===")

        # Stage 1: Fetch data
        data_output = self.data_agent.execute(context)
        context.update(data_output)

        if not context.get("ohlcv_data"):
            logger.error("No data fetched. Aborting pipeline.")
            return context

        # Stages 2-4: Run regime detection, correlation, and analysis in parallel
        logger.info("Running regime detection, correlation, and analysis in parallel...")
        ohlcv_data = context["ohlcv_data"]

        with ThreadPoolExecutor(max_workers=3) as executor:
            regime_future = executor.submit(MarketRegime.detect_all, ohlcv_data)
            corr_future = executor.submit(
                self.portfolio_manager.compute_correlation_matrix, ohlcv_data
            )
            analysis_future = executor.submit(self.analysis_agent.execute, context)

            regimes = regime_future.result()
            corr_matrix = corr_future.result()
            analysis_output = analysis_future.result()

        context["regimes"] = regimes
        for pair, regime in regimes.items():
            logger.info(
                f"  {pair}: {regime['regime']} "
                f"(ADX={regime['adx']:.1f}, vol={regime['volatility_state']}, "
                f"trend={regime['trend_direction']})"
            )

        context["correlation_matrix"] = corr_matrix
        clusters = self.portfolio_manager.get_correlation_clusters()
        if clusters:
            logger.info(f"  Correlation clusters: {clusters}")

        context.update(analysis_output)

        # Stage 5: Predict
        prediction_output = self.prediction_agent.execute(context)
        context.update(prediction_output)

        # Stage 6: Recommend (now regime-aware)
        recommendation_output = self.recommendation_agent.execute(context)
        context.update(recommendation_output)

        # Stage 7: Portfolio risk filter
        raw_signal_count = len(context.get("signals", []))
        if context.get("signals"):
            logger.info("Applying portfolio risk filters...")
            portfolio_status = self.portfolio_manager.get_current_portfolio_risk()
            context["portfolio_status"] = portfolio_status
            logger.info(
                f"  Portfolio risk: {portfolio_status['total_risk_pct'] * 100:.1f}% "
                f"({portfolio_status['open_positions']} open positions, "
                f"{portfolio_status['risk_available'] * 100:.1f}% available)"
            )

            filtered = self.portfolio_manager.filter_signals(
                context["signals"], context["ohlcv_data"]
            )
            rejected_count = raw_signal_count - len(filtered)
            if rejected_count > 0:
                logger.warning(f"  Filtered out {rejected_count} signals (portfolio risk limits)")
            context["signals"] = filtered

        # Stage 8: Log & feedback
        logging_output = self.logging_agent.execute(context)
        context.update(logging_output)

        # Handle feedback: delete models for pairs needing retraining
        feedback = context.get("feedback", {})
        if feedback.get("retrain_pairs"):
            logger.warning(f"Retraining requested for: {feedback['retrain_pairs']}")
            for pair in feedback["retrain_pairs"]:
                for suffix in ("_lstm_v2.pt", "_lstm.pt"):
                    model_path = os.path.join(MODEL_DIR, f"{pair.replace('=', '_')}{suffix}")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        logger.info(f"Deleted model {suffix} for {pair} — will retrain on next run")

        logger.info(
            f"=== Pipeline complete. "
            f"{len(context.get('signals', []))} signals approved "
            f"(of {raw_signal_count} generated) ==="
        )
        return context

    def run_analysis_only(self, pairs: list = None) -> dict:
        """Quick run: fetch + analyze + regime detection (no ML)."""
        context = {}
        if pairs:
            context["pairs"] = pairs

        data_output = self.data_agent.execute(context)
        context.update(data_output)

        if not context.get("ohlcv_data"):
            return context

        context["regimes"] = MarketRegime.detect_all(context["ohlcv_data"])
        context["correlation_matrix"] = self.portfolio_manager.compute_correlation_matrix(
            context["ohlcv_data"]
        )

        analysis_output = self.analysis_agent.execute(context)
        context.update(analysis_output)
        return context
