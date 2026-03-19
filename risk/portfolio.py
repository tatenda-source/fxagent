import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from data.storage import Storage
from config import (
    PAIR_DISPLAY, MAX_PORTFOLIO_RISK, MAX_CORRELATED_POSITIONS,
    CORRELATION_THRESHOLD, CORRELATION_LOOKBACK, MAX_RISK_PER_TRADE,
    DEFAULT_ACCOUNT_SIZE,
)

MAX_DRAWDOWN_PCT = 0.15
logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """
    Controls portfolio-level risk:
    - Tracks total exposure across all open positions
    - Computes cross-pair correlation matrix
    - Blocks new trades that would exceed portfolio risk limits
    - Prevents correlated position clustering (e.g., 5 USD-short trades)
    """

    def __init__(self, account_size: float = DEFAULT_ACCOUNT_SIZE):
        self.storage = Storage()
        self.account_size = account_size
        self._correlation_matrix = None

    def compute_correlation_matrix(self, ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build a correlation matrix from close prices across all pairs.
        Uses last CORRELATION_LOOKBACK days of returns.
        """
        closes = {}
        for pair, df in ohlcv_data.items():
            if len(df) >= CORRELATION_LOOKBACK:
                closes[pair] = df["Close"].tail(CORRELATION_LOOKBACK).pct_change().dropna()

        if len(closes) < 2:
            self._correlation_matrix = pd.DataFrame()
            return self._correlation_matrix

        returns_df = pd.DataFrame(closes)
        self._correlation_matrix = returns_df.corr()
        return self._correlation_matrix

    def get_correlated_pairs(self, pair: str) -> List[str]:
        """Return pairs that are highly correlated (|corr| > threshold) with the given pair."""
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return []
        if pair not in self._correlation_matrix.columns:
            return []

        corr_series = self._correlation_matrix[pair].drop(pair, errors="ignore")
        correlated = corr_series[corr_series.abs() > CORRELATION_THRESHOLD]
        return list(correlated.index)

    def get_current_portfolio_risk(self) -> Dict:
        """
        Calculate current portfolio risk from open positions.
        Returns:
            {
                "total_risk_pct": float,    # Total % of account at risk
                "open_positions": int,
                "risk_by_pair": {pair: risk_pct},
                "risk_available": float,    # Remaining risk budget
            }
        """
        open_signals = self.storage.get_open_signals()

        if open_signals.empty:
            return {
                "total_risk_pct": 0.0,
                "open_positions": 0,
                "risk_by_pair": {},
                "risk_available": MAX_PORTFOLIO_RISK,
            }

        risk_by_pair = {}
        total_risk = 0.0

        for _, sig in open_signals.iterrows():
            risk_amount = abs(sig["entry_price"] - sig["stop_loss"]) * sig.get("position_size", 0)
            risk_pct = risk_amount / self.account_size if self.account_size > 0 else 0
            risk_by_pair[sig["pair"]] = risk_pct
            total_risk += risk_pct

        return {
            "total_risk_pct": total_risk,
            "open_positions": len(open_signals),
            "risk_by_pair": risk_by_pair,
            "risk_available": max(0, MAX_PORTFOLIO_RISK - total_risk),
        }

    def check_drawdown_limit(self, current_equity: float, peak_equity: float) -> bool:
        if peak_equity <= 0:
            return False
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown > MAX_DRAWDOWN_PCT:
            logger.critical(
                f"Max drawdown breached: {drawdown:.1%} drawdown "
                f"(current={current_equity:.2f}, peak={peak_equity:.2f}). "
                f"All new signals rejected."
            )
            return True
        return False

    def filter_signals(self, signals: List[Dict], ohlcv_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Filter signals through portfolio risk constraints.
        Removes signals that would:
        1. Exceed total portfolio risk limit
        2. Add too many correlated positions
        3. Duplicate an already open pair

        Returns filtered list + attaches rejection reasons to removed signals.
        """
        if self._correlation_matrix is None:
            self.compute_correlation_matrix(ohlcv_data)

        portfolio = self.get_current_portfolio_risk()
        open_signals = self.storage.get_open_signals()
        open_pairs = set(open_signals["pair"].tolist()) if not open_signals.empty else set()

        open_pair_directions: Dict[str, str] = {}
        if not open_signals.empty:
            for _, sig in open_signals.iterrows():
                open_pair_directions[sig["pair"]] = sig.get("signal_type", "")

        approved = []
        rejected = []

        # Sort by confidence descending — best signals get through first
        sorted_signals = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)

        running_risk = portfolio["total_risk_pct"]

        for signal in sorted_signals:
            pair = signal["pair"]
            reasons = []

            # Check 1: Already have an open position on this pair in the same direction
            if pair in open_pairs:
                existing_dir = open_pair_directions.get(pair, "")
                if existing_dir == signal.get("signal_type", ""):
                    reasons.append(
                        f"Duplicate signal: already have open {existing_dir} on "
                        f"{PAIR_DISPLAY.get(pair, pair)}"
                    )
                else:
                    reasons.append(f"Already have open position on {PAIR_DISPLAY.get(pair, pair)}")

            # Check 2: Portfolio risk limit
            signal_risk = abs(signal["entry_price"] - signal["stop_loss"]) * signal.get("position_size", 0)
            signal_risk_pct = signal_risk / self.account_size if self.account_size > 0 else 0

            if running_risk + signal_risk_pct > MAX_PORTFOLIO_RISK:
                reasons.append(
                    f"Would exceed portfolio risk limit "
                    f"({(running_risk + signal_risk_pct) * 100:.1f}% > {MAX_PORTFOLIO_RISK * 100:.0f}%)"
                )

            # Check 3: Correlated positions
            correlated = self.get_correlated_pairs(pair)
            correlated_open_count = sum(1 for p in correlated if p in open_pairs)
            correlated_approved_count = sum(1 for s in approved if s["pair"] in correlated)
            total_correlated = correlated_open_count + correlated_approved_count

            if total_correlated >= MAX_CORRELATED_POSITIONS:
                corr_names = [PAIR_DISPLAY.get(p, p) for p in correlated if p in open_pairs]
                reasons.append(
                    f"Too many correlated positions ({total_correlated} already: {', '.join(corr_names)})"
                )

            if reasons:
                signal["rejection_reasons"] = reasons
                rejected.append(signal)
            else:
                approved.append(signal)
                running_risk += signal_risk_pct
                open_pairs.add(pair)

        return approved

    def get_correlation_clusters(self) -> List[List[str]]:
        """Group pairs into correlation clusters for display."""
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return []

        pairs = list(self._correlation_matrix.columns)
        visited = set()
        clusters = []

        for pair in pairs:
            if pair in visited:
                continue
            cluster = [pair]
            visited.add(pair)
            correlated = self.get_correlated_pairs(pair)
            for cp in correlated:
                if cp not in visited:
                    cluster.append(cp)
                    visited.add(cp)
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters
