import pandas as pd

from agents.base_agent import BaseAgent
from data.fetcher import ForexFetcher
from data.storage import Storage
from config import ALL_PAIRS, DEFAULT_PERIOD, DEFAULT_INTERVAL


class DataAgent(BaseAgent):
    """Agent 1: Fetches forex data, cleans it, and stores in SQLite."""

    def __init__(self):
        super().__init__(name="DataAgent")
        self.fetcher = ForexFetcher()
        self.storage = Storage()

    def run(self, input_data: dict) -> dict:
        pairs = input_data.get("pairs", ALL_PAIRS)
        period = input_data.get("period", DEFAULT_PERIOD)
        interval = input_data.get("interval", DEFAULT_INTERVAL)

        output = {}
        for pair in pairs:
            df = self.fetcher.fetch_historical(pair, period, interval)
            if df.empty:
                self.logger.warning(f"No data returned for {pair}")
                continue
            df = self._clean(df)
            self.storage.save_ohlcv(pair, df, interval)
            output[pair] = df
            self.logger.info(f"Fetched {len(df)} rows for {pair}")

        return {"ohlcv_data": output}

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        df = df.sort_index()
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        return df
