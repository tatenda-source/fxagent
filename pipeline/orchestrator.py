import os

from loguru import logger

from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.prediction_agent import PredictionAgent
from agents.recommendation_agent import RecommendationAgent
from agents.logging_agent import LoggingAgent
from config import MODEL_DIR


class Orchestrator:
    """Coordinates all 5 agents in a sequential pipeline."""

    def __init__(self):
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.prediction_agent = PredictionAgent()
        self.recommendation_agent = RecommendationAgent()
        self.logging_agent = LoggingAgent()

    def run_full_pipeline(self, pairs: list = None) -> dict:
        """Execute the complete agent pipeline. Returns merged context."""
        context = {}
        if pairs:
            context["pairs"] = pairs

        logger.info("=== Starting full pipeline ===")

        # Stage 1: Fetch data
        data_output = self.data_agent.execute(context)
        context.update(data_output)

        if not context.get("ohlcv_data"):
            logger.error("No data fetched. Aborting pipeline.")
            return context

        # Stage 2: Analyze
        analysis_output = self.analysis_agent.execute(context)
        context.update(analysis_output)

        # Stage 3: Predict
        prediction_output = self.prediction_agent.execute(context)
        context.update(prediction_output)

        # Stage 4: Recommend
        recommendation_output = self.recommendation_agent.execute(context)
        context.update(recommendation_output)

        # Stage 5: Log & feedback
        logging_output = self.logging_agent.execute(context)
        context.update(logging_output)

        # Handle feedback: delete models for pairs needing retraining
        feedback = context.get("feedback", {})
        if feedback.get("retrain_pairs"):
            logger.warning(f"Retraining requested for: {feedback['retrain_pairs']}")
            for pair in feedback["retrain_pairs"]:
                model_path = os.path.join(MODEL_DIR, f"{pair.replace('=', '_')}_lstm.pt")
                if os.path.exists(model_path):
                    os.remove(model_path)
                    logger.info(f"Deleted model for {pair} — will retrain on next run")

        logger.info(f"=== Pipeline complete. {len(context.get('signals', []))} signals generated ===")
        return context

    def run_analysis_only(self, pairs: list = None) -> dict:
        """Quick run: fetch + analyze only (no ML). Useful for dashboard refresh."""
        context = {}
        if pairs:
            context["pairs"] = pairs

        data_output = self.data_agent.execute(context)
        context.update(data_output)

        if not context.get("ohlcv_data"):
            return context

        analysis_output = self.analysis_agent.execute(context)
        context.update(analysis_output)
        return context
