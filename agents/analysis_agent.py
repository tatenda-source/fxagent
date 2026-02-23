from agents.base_agent import BaseAgent
from indicators.technical import add_all_indicators
from indicators.patterns import find_support_resistance


class AnalysisAgent(BaseAgent):
    """Agent 2: Applies technical indicators and detects support/resistance."""

    def __init__(self):
        super().__init__(name="AnalysisAgent")

    def run(self, input_data: dict) -> dict:
        ohlcv_data = input_data["ohlcv_data"]
        analyzed = {}
        sr_levels = {}

        for pair, df in ohlcv_data.items():
            df_with_indicators = add_all_indicators(df)
            df_with_indicators.dropna(inplace=True)
            analyzed[pair] = df_with_indicators
            sr_levels[pair] = find_support_resistance(df)
            self.logger.info(
                f"{pair}: {len(df_with_indicators)} rows with indicators, "
                f"{len(sr_levels[pair]['support'])} support, "
                f"{len(sr_levels[pair]['resistance'])} resistance levels"
            )

        return {"analyzed_data": analyzed, "sr_levels": sr_levels}
