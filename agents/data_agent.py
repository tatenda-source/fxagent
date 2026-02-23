import pandas as pd

from agents.base_agent import BaseAgent
from data.fetcher import ForexFetcher
from data.storage import Storage
from config import ALL_PAIRS, DEFAULT_PERIOD, DEFAULT_INTERVAL, INTRADAY_INTERVAL


class DataAgent(BaseAgent):
    """Agent 1: Fetches forex data, cleans it, and stores in SQLite.
    Supports both daily and intraday data fetching."""

    def __init__(self):
        super().__init__(name="DataAgent")
        self.fetcher = ForexFetcher()
        self.storage = Storage()

    def run(self, input_data: dict) -> dict:
        pairs = input_data.get("pairs", ALL_PAIRS)
        period = input_data.get("period", DEFAULT_PERIOD)
        interval = input_data.get("interval", DEFAULT_INTERVAL)
        fetch_intraday = input_data.get("fetch_intraday", False)

        output = {}
        intraday_output = {}

        for pair in pairs:
            # Daily data (for training + indicators)
            df = self.fetcher.fetch_historical(pair, period, interval)
            if df.empty:
                self.logger.warning(f"No data returned for {pair}")
                continue
            df = self._clean(df)
            self.storage.save_ohlcv(pair, df, interval)
            output[pair] = df
            self.logger.info(f"Fetched {len(df)} daily rows for {pair}")

            # Intraday data (for faster signal generation)
            if fetch_intraday:
                intra_df = self.fetcher.fetch_latest(pair, interval=INTRADAY_INTERVAL)
                if not intra_df.empty:
                    intra_df = self._clean(intra_df)
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
