from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd

from config import ALL_PAIRS, DEFAULT_PERIOD, DEFAULT_INTERVAL


class ForexFetcher:
    """Wraps yfinance for forex data retrieval with batch + parallel support."""

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
        pairs: list = None,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> dict:
        """Fetch data for all pairs using batch download. Falls back to parallel individual fetches."""
        if pairs is None:
            pairs = ALL_PAIRS

        # Try batch download first (single HTTP request for all tickers)
        try:
            results = self._batch_download(pairs, period, interval)
            if results:
                return results
        except Exception:
            pass

        # Fallback: parallel individual fetches
        return self._parallel_fetch(pairs, period, interval)

    def _batch_download(self, pairs: list, period: str, interval: str) -> dict:
        """Use yf.download() to fetch all tickers in one bulk request."""
        tickers_str = " ".join(pairs)
        raw = yf.download(tickers_str, period=period, interval=interval, group_by="ticker", threads=True)

        if raw.empty:
            return {}

        results = {}
        for pair in pairs:
            try:
                if len(pairs) == 1:
                    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                else:
                    df = raw[pair][["Open", "High", "Low", "Close", "Volume"]].copy()
                df.dropna(inplace=True)
                if not df.empty:
                    results[pair] = df
            except (KeyError, TypeError):
                continue

        return results

    def _parallel_fetch(self, pairs: list, period: str, interval: str) -> dict:
        """Fetch pairs in parallel using threads."""
        results = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self.fetch_historical, pair, period, interval): pair
                for pair in pairs
            }
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[pair] = df
                except Exception:
                    continue
        return results

    def fetch_all_latest(self, pairs: list = None, interval: str = "1h") -> dict:
        """Fetch the most recent data for all pairs in parallel."""
        if pairs is None:
            pairs = ALL_PAIRS

        results = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self.fetch_latest, pair, interval): pair
                for pair in pairs
            }
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[pair] = df
                except Exception:
                    continue
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
