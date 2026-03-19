import logging
import numpy as np
import pandas as pd
import ta

from config import (
    REGIME_ADX_PERIOD, REGIME_ADX_TRENDING,
    REGIME_VOLATILITY_LOOKBACK, REGIME_VOLATILITY_HIGH_MULT,
)

logger = logging.getLogger(__name__)


class MarketRegime:
    """Identifies what type of market we're in: trending, ranging, or volatile."""

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"

    @staticmethod
    def detect(df: pd.DataFrame) -> dict:
        """
        Analyze the current market regime for a given pair.

        Returns:
            {
                "regime": "trending" | "ranging" | "volatile",
                "adx": float,              # Trend strength (0-100)
                "trend_direction": "up" | "down" | "flat",
                "volatility_state": "normal" | "high" | "extreme",
                "volatility_ratio": float,  # Current ATR / median ATR
                "confidence": float,        # How confident we are in the classification
                "strategy_adjustments": {   # Recommended parameter tweaks
                    "sl_multiplier_adj": float,
                    "tp_multiplier_adj": float,
                    "min_score_adj": float,
                    "position_size_adj": float,
                }
            }
        """
        required_rows = max(REGIME_ADX_PERIOD * 2, REGIME_VOLATILITY_LOOKBACK + 10)
        if not isinstance(df, pd.DataFrame) or len(df) < required_rows:
            logger.warning(
                f"Insufficient data for regime detection ({len(df) if isinstance(df, pd.DataFrame) else 0} rows, "
                f"need {required_rows}). Returning unknown regime."
            )
            return MarketRegime._unknown_regime()

        for col in ("High", "Low", "Close"):
            if col not in df.columns:
                logger.warning(f"Missing column '{col}' in DataFrame. Returning unknown regime.")
                return MarketRegime._unknown_regime()

        result = {}

        # --- ADX for trend strength ---
        try:
            adx_indicator = ta.trend.ADXIndicator(
                df["High"], df["Low"], df["Close"], window=REGIME_ADX_PERIOD
            )
            adx = adx_indicator.adx().iloc[-1]
            plus_di = adx_indicator.adx_pos().iloc[-1]
            minus_di = adx_indicator.adx_neg().iloc[-1]
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}. Returning unknown regime.")
            return MarketRegime._unknown_regime()

        adx_val = float(adx) if not np.isnan(adx) else 0.0
        result["adx"] = float(np.clip(adx_val, 0.0, 100.0))

        # Trend direction from DI+/DI-
        if np.isnan(plus_di) or np.isnan(minus_di):
            result["trend_direction"] = "flat"
        elif plus_di > minus_di:
            result["trend_direction"] = "up"
        elif minus_di > plus_di:
            result["trend_direction"] = "down"
        else:
            result["trend_direction"] = "flat"

        # --- Volatility analysis ---
        try:
            atr_series = ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=14
            )
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}. Returning unknown regime.")
            return MarketRegime._unknown_regime()

        current_atr = atr_series.iloc[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            current_atr = 0.0
        else:
            current_atr = max(current_atr, 0.0)
        median_atr = atr_series.tail(REGIME_VOLATILITY_LOOKBACK).median()
        if np.isnan(median_atr) or median_atr <= 0:
            median_atr = 0.0
        vol_ratio = current_atr / median_atr if median_atr > 0 else 1.0

        result["volatility_ratio"] = float(vol_ratio)

        if vol_ratio > REGIME_VOLATILITY_HIGH_MULT * 1.5:
            result["volatility_state"] = "extreme"
        elif vol_ratio > REGIME_VOLATILITY_HIGH_MULT:
            result["volatility_state"] = "high"
        else:
            result["volatility_state"] = "normal"

        # --- Regime classification ---
        if result["volatility_state"] == "extreme":
            result["regime"] = MarketRegime.VOLATILE
            result["confidence"] = min(0.95, vol_ratio / 3.0)
        elif result["adx"] >= REGIME_ADX_TRENDING:
            result["regime"] = MarketRegime.TRENDING
            result["confidence"] = min(0.95, result["adx"] / 50.0)
        else:
            result["regime"] = MarketRegime.RANGING
            result["confidence"] = min(0.95, (REGIME_ADX_TRENDING - result["adx"]) / REGIME_ADX_TRENDING)

        # --- Strategy adjustments based on regime ---
        result["strategy_adjustments"] = MarketRegime._get_adjustments(result)

        return result

    @staticmethod
    def _get_adjustments(regime_info: dict) -> dict:
        """
        Return multiplier adjustments for the Recommendation Agent.
        These modify the base parameters from config.py.
        """
        regime = regime_info["regime"]
        vol_state = regime_info["volatility_state"]

        if regime == MarketRegime.TRENDING:
            return {
                "sl_multiplier_adj": 1.0,     # Normal stops in trends
                "tp_multiplier_adj": 1.5,     # Let winners run — wider TP
                "min_score_adj": -0.3,        # Lower entry bar — trends are forgiving
                "position_size_adj": 1.0,     # Normal size
            }
        elif regime == MarketRegime.VOLATILE:
            return {
                "sl_multiplier_adj": 1.5,     # Wider stops to avoid noise
                "tp_multiplier_adj": 1.2,     # Slightly wider TP
                "min_score_adj": 0.5,         # Higher bar — need more confirmation
                "position_size_adj": 0.5,     # Half size — protect capital
            }
        else:  # RANGING
            return {
                "sl_multiplier_adj": 0.8,     # Tighter stops — less room to move
                "tp_multiplier_adj": 0.8,     # Tighter targets — take quick profits
                "min_score_adj": 0.2,         # Slightly higher bar
                "position_size_adj": 0.8,     # Slightly smaller size
            }

    @staticmethod
    def _default_regime() -> dict:
        return {
            "regime": MarketRegime.RANGING,
            "adx": 0.0,
            "trend_direction": "flat",
            "volatility_state": "normal",
            "volatility_ratio": 1.0,
            "confidence": 0.0,
            "strategy_adjustments": {
                "sl_multiplier_adj": 1.0,
                "tp_multiplier_adj": 1.0,
                "min_score_adj": 0.0,
                "position_size_adj": 1.0,
            },
        }

    @staticmethod
    def _unknown_regime() -> dict:
        return {
            "regime": "unknown",
            "adx": 0.0,
            "trend_direction": "flat",
            "volatility_state": "normal",
            "volatility_ratio": 1.0,
            "confidence": 0.0,
            "strategy_adjustments": {
                "sl_multiplier_adj": 1.0,
                "tp_multiplier_adj": 1.0,
                "min_score_adj": 0.0,
                "position_size_adj": 1.0,
            },
        }

    @staticmethod
    def is_tradeable_regime(regime_info: dict) -> tuple:
        """Determine if a regime is suitable for trading.

        Returns (tradeable: bool, reason: str)

        Rules:
        - VOLATILE regime with vol_ratio > 2.0: NOT tradeable ("extreme volatility")
        - RANGING regime with ADX < 12: NOT tradeable ("dead market")
        - Any regime with confidence < 0.3: NOT tradeable ("unclear regime")
        - Everything else: tradeable
        """
        regime = regime_info.get("regime", "unknown")
        vol_ratio = regime_info.get("volatility_ratio", 1.0)
        adx = regime_info.get("adx", 0.0)
        confidence = regime_info.get("confidence", 0.0)

        if regime == MarketRegime.VOLATILE and vol_ratio > 2.0:
            return False, "extreme volatility"

        if regime == MarketRegime.RANGING and adx < 12:
            return False, "dead market"

        if confidence < 0.3:
            return False, "unclear regime"

        return True, ""

    @staticmethod
    def detect_all(ohlcv_data: dict) -> dict:
        """Detect regime for all pairs. Returns {pair: regime_dict}."""
        regimes = {}
        for pair, df in ohlcv_data.items():
            regimes[pair] = MarketRegime.detect(df)
        return regimes
