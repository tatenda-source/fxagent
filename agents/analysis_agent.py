from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base_agent import BaseAgent
from indicators.technical import add_all_indicators
from indicators.patterns import find_support_resistance


class AnalysisAgent(BaseAgent):
    """Agent 2: Applies technical indicators and detects support/resistance in parallel."""

    def __init__(self):
        super().__init__(name="AnalysisAgent")

    def run(self, input_data: dict) -> dict:
        ohlcv_data = input_data["ohlcv_data"]
        analyzed = {}
        sr_levels = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._analyze_pair, pair, df): pair
                for pair, df in ohlcv_data.items()
            }
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    df_result, sr = future.result()
                    analyzed[pair] = df_result
                    sr_levels[pair] = sr
                    self.logger.info(
                        f"{pair}: {len(df_result)} rows with indicators, "
                        f"{len(sr['support'])} support, "
                        f"{len(sr['resistance'])} resistance levels"
                    )
                except Exception as e:
                    self.logger.error(f"Analysis failed for {pair}: {e}")

        return {"analyzed_data": analyzed, "sr_levels": sr_levels}

    def _analyze_pair(self, pair: str, df):
        df_with_indicators = add_all_indicators(df)
        df_with_indicators.dropna(inplace=True)
        sr = find_support_resistance(df)
        return df_with_indicators, sr
