import time

import pandas as pd

from agents.base_agent import BaseAgent
from data.fetcher import ForexFetcher
from data.storage import Storage
from config import ALL_PAIRS, DEFAULT_PERIOD, DEFAULT_INTERVAL, INTRADAY_INTERVAL

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2
STALE_DATA_DAYS = 3
MAX_NAN_RATIO = 0.10
MIN_ROW_COUNT = 200


class DataAgent(BaseAgent):
    """Agent 1: Fetches forex data, cleans it, and stores in SQLite.
    Uses batch/parallel fetching for speed."""

    def __init__(self):
        super().__init__(name="DataAgent")
        self.fetcher = ForexFetcher()
        self.storage = Storage()

    def _fetch_with_retry(self, fetch_fn, *args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fetch_fn(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES:
                    self.logger.error(f"Fetch failed after {MAX_RETRIES} attempts: {e}")
                    return {}
                wait = RETRY_BACKOFF_BASE ** attempt
                self.logger.warning(f"Fetch attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return {}

    def _validate_pair(self, pair: str, df: pd.DataFrame) -> bool:
        if df.empty:
            self.logger.warning(f"No data returned for {pair}")
            return False

        if len(df) < MIN_ROW_COUNT:
            self.logger.warning(f"Skipping {pair}: only {len(df)} rows (minimum {MIN_ROW_COUNT})")
            return False

        latest_ts = df.index.max()
        if hasattr(latest_ts, 'tz') and latest_ts.tz is not None:
            now = pd.Timestamp.now(tz=latest_ts.tz)
        else:
            now = pd.Timestamp.now()
        days_old = (now - latest_ts).days
        if days_old > STALE_DATA_DAYS:
            self.logger.warning(f"Stale data for {pair}: latest point is {days_old} days old")

        return True

    def run(self, input_data: dict) -> dict:
        pairs = input_data.get("pairs", ALL_PAIRS)
        period = input_data.get("period", DEFAULT_PERIOD)
        interval = input_data.get("interval", DEFAULT_INTERVAL)
        fetch_intraday = input_data.get("fetch_intraday", False)

        # Batch fetch all daily data at once (with retry)
        raw_data = self._fetch_with_retry(self.fetcher.fetch_all_pairs, pairs, period, interval)

        output = {}
        for pair, df in raw_data.items():
            df = self._clean(pair, df)
            if not self._validate_pair(pair, df):
                continue
            self.storage.save_ohlcv(pair, df, interval)
            output[pair] = df
            self.logger.info(f"Fetched {len(df)} daily rows for {pair}")

        # Batch fetch intraday data in parallel (with retry)
        intraday_output = {}
        if fetch_intraday:
            raw_intraday = self._fetch_with_retry(
                self.fetcher.fetch_all_latest, pairs, interval=INTRADAY_INTERVAL
            )
            for pair, intra_df in raw_intraday.items():
                intra_df = self._clean(pair, intra_df)
                if not intra_df.empty:
                    self.storage.save_ohlcv(pair, intra_df, INTRADAY_INTERVAL)
                    intraday_output[pair] = intra_df
                    self.logger.info(f"Fetched {len(intra_df)} intraday rows for {pair}")

        result = {"ohlcv_data": output}
        if intraday_output:
            result["intraday_data"] = intraday_output
        return result

    def _clean(self, pair: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        ohlc_cols = ["Open", "High", "Low", "Close"]
        nan_count = df[ohlc_cols].isna().sum().sum()
        total_values = df[ohlc_cols].size
        if total_values > 0:
            nan_ratio = nan_count / total_values
            if nan_ratio > MAX_NAN_RATIO:
                self.logger.error(
                    f"Rejecting {pair}: {nan_ratio:.1%} NaN values exceeds {MAX_NAN_RATIO:.0%} threshold"
                )
                return pd.DataFrame()
            if nan_count > 0:
                self.logger.warning(f"{pair}: forward-filling {nan_count} NaN values")
                df[ohlc_cols] = df[ohlc_cols].ffill()
                df = df.dropna(subset=ohlc_cols)
        for col in ohlc_cols:
            df[col] = df[col].astype(float)
        return df
