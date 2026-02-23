import yfinance as yf
import pandas as pd

from config import ALL_PAIRS, DEFAULT_PERIOD, DEFAULT_INTERVAL


class ForexFetcher:
    """Wraps yfinance for forex data retrieval."""

    def fetch_historical(
        self,
        pair: str,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV for a single pair. Returns DataFrame with Open/High/Low/Close/Volume."""
        ticker = yf.Ticker(pair)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return df
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df

    def fetch_all_pairs(
        self,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> dict:
        """Fetch data for all configured pairs. Returns {pair: DataFrame}."""
        results = {}
        for pair in ALL_PAIRS:
            results[pair] = self.fetch_historical(pair, period, interval)
        return results

    def fetch_latest(self, pair: str, interval: str = "1h") -> pd.DataFrame:
        """Fetch the most recent data (last 5 days)."""
        ticker = yf.Ticker(pair)
        df = ticker.history(period="5d", interval=interval)
        if df.empty:
            return df
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df
