import pandas as pd

from agents.base_agent import BaseAgent
from data.fetcher import ForexFetcher
from data.storage import Storage
from config import ALL_PAIRS, DEFAULT_PERIOD, DEFAULT_INTERVAL, INTRADAY_INTERVAL


class DataAgent(BaseAgent):
    """Agent 1: Fetches forex data, cleans it, and stores in SQLite.
    Uses batch/parallel fetching for speed."""

    def __init__(self):
        super().__init__(name="DataAgent")
        self.fetcher = ForexFetcher()
        self.storage = Storage()

    def run(self, input_data: dict) -> dict:
        pairs = input_data.get("pairs", ALL_PAIRS)
        period = input_data.get("period", DEFAULT_PERIOD)
        interval = input_data.get("interval", DEFAULT_INTERVAL)
        fetch_intraday = input_data.get("fetch_intraday", False)

        # Batch fetch all daily data at once
        raw_data = self.fetcher.fetch_all_pairs(pairs, period, interval)

        output = {}
        for pair, df in raw_data.items():
            df = self._clean(df)
            if df.empty:
                self.logger.warning(f"No data returned for {pair}")
                continue
            self.storage.save_ohlcv(pair, df, interval)
            output[pair] = df
            self.logger.info(f"Fetched {len(df)} daily rows for {pair}")

        # Batch fetch intraday data in parallel
        intraday_output = {}
        if fetch_intraday:
            raw_intraday = self.fetcher.fetch_all_latest(pairs, interval=INTRADAY_INTERVAL)
            for pair, intra_df in raw_intraday.items():
                intra_df = self._clean(intra_df)
                if not intra_df.empty:
                    self.storage.save_ohlcv(pair, intra_df, INTRADAY_INTERVAL)
                    intraday_output[pair] = intra_df
                    self.logger.info(f"Fetched {len(intra_df)} intraday rows for {pair}")

        result = {"ohlcv_data": output}
        if intraday_output:
            result["intraday_data"] = intraday_output
        return result

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        df = df.sort_index()
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        return df
