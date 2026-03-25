import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Session windows (UTC hours)
LONDON_OPEN = 7
LONDON_CLOSE = 16
NEW_YORK_OPEN = 12
NEW_YORK_CLOSE = 21

# ATR expansion bounds
ATR_RATIO_MIN = 0.8
ATR_RATIO_MAX = 2.0

# Minimum technical indicators that must agree
MIN_CONFLUENCE = 3

# Noise filter: predicted return must exceed this multiple of the noise floor
NOISE_MULTIPLIER = 0.5


class TradeFilter:
    """Pre-trade quality filter. Rejects trades that fail quality checks.
    This is the single biggest improvement for reducing noise trades."""

    # ------------------------------------------------------------------ #
    #  1. Session Filter
    # ------------------------------------------------------------------ #
    @staticmethod
    def passes_session_filter(pair: str, timestamp) -> bool:
        """Only allow trades during high-liquidity sessions.

        FX pairs (=X): London OR New York session.
        Metals (=F): New York session only.
        """
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                hour = timestamp.hour  # assume UTC
            else:
                hour = timestamp.astimezone(timezone.utc).hour
        else:
            # Fallback: try to pull .hour from whatever was passed
            hour = getattr(timestamp, "hour", None)
            if hour is None:
                logger.warning("Cannot extract hour from timestamp %s, allowing trade", timestamp)
                return True

        in_london = LONDON_OPEN <= hour < LONDON_CLOSE
        in_new_york = NEW_YORK_OPEN <= hour < NEW_YORK_CLOSE

        if pair.endswith("=X"):
            return in_london or in_new_york
        elif pair.endswith("=F"):
            return in_new_york
        else:
            # Unknown instrument type — don't block
            return True

    # ------------------------------------------------------------------ #
    #  2. ATR Expansion Filter
    # ------------------------------------------------------------------ #
    @staticmethod
    def passes_volatility_filter(atr_current: float, atr_median: float) -> bool:
        """Allow if volatility is alive but not exploding.

        ATR ratio must be between 0.8 and 2.0.
        Dead markets (< 0.8) and spikes (> 2.0) are rejected.
        """
        if atr_median <= 0:
            logger.warning("ATR median is zero or negative, skipping volatility filter")
            return True

        ratio = atr_current / atr_median
        return ATR_RATIO_MIN <= ratio <= ATR_RATIO_MAX

    # ------------------------------------------------------------------ #
    #  3. Confluence Score
    # ------------------------------------------------------------------ #
    @staticmethod
    def count_confluence(direction: str, latest_row) -> int:
        """Count how many technical indicators agree with the predicted direction.

        Checks: RSI, MACD, Bollinger Bands, SMA 50, Stochastic %K.
        Returns the count (0-5).
        """
        count = 0
        is_up = direction == "UP"

        # RSI
        rsi = latest_row.get("RSI") if isinstance(latest_row, dict) else getattr(latest_row, "RSI", None)
        if rsi is not None:
            if is_up and rsi < 40:
                count += 1
            elif not is_up and rsi > 60:
                count += 1

        # MACD vs Signal
        macd = latest_row.get("MACD") if isinstance(latest_row, dict) else getattr(latest_row, "MACD", None)
        macd_sig = latest_row.get("MACD_signal") if isinstance(latest_row, dict) else getattr(latest_row, "MACD_signal", None)
        if macd is not None and macd_sig is not None:
            if is_up and macd > macd_sig:
                count += 1
            elif not is_up and macd < macd_sig:
                count += 1

        # Bollinger Bands
        close = latest_row.get("Close") if isinstance(latest_row, dict) else getattr(latest_row, "Close", None)
        bb_lower = latest_row.get("BB_lower") if isinstance(latest_row, dict) else getattr(latest_row, "BB_lower", None)
        bb_upper = latest_row.get("BB_upper") if isinstance(latest_row, dict) else getattr(latest_row, "BB_upper", None)
        if close is not None and bb_lower is not None and bb_upper is not None:
            if is_up and close < bb_lower:
                count += 1
            elif not is_up and close > bb_upper:
                count += 1

        # SMA 50
        sma_50 = latest_row.get("SMA_50") if isinstance(latest_row, dict) else getattr(latest_row, "SMA_50", None)
        if close is not None and sma_50 is not None:
            if is_up and close > sma_50:
                count += 1
            elif not is_up and close < sma_50:
                count += 1

        # Stochastic %K
        stoch_k = latest_row.get("STOCH_K") if isinstance(latest_row, dict) else getattr(latest_row, "STOCH_K", None)
        if stoch_k is not None:
            if is_up and stoch_k < 20:
                count += 1
            elif not is_up and stoch_k > 80:
                count += 1

        return count

    # ------------------------------------------------------------------ #
    #  4. Spread / Noise Filter
    # ------------------------------------------------------------------ #
    @staticmethod
    def passes_noise_filter(predicted_return: float, atr: float, close: float) -> bool:
        """Reject if predicted return is too small relative to ATR-implied noise.

        noise_floor = ATR / close
        Reject if abs(predicted_return) < noise_floor * 0.5
        """
        if close <= 0:
            logger.warning("Close price is zero or negative, skipping noise filter")
            return True

        noise_floor = atr / close
        return abs(predicted_return) >= noise_floor * NOISE_MULTIPLIER

    # ------------------------------------------------------------------ #
    #  5. Main Filter
    # ------------------------------------------------------------------ #
    def filter_trade(self, pair: str, direction: str, predicted_return: float,
                     confidence: float, latest_row, timestamp) -> tuple:
        """Run all filters. Returns (passes, rejection_reasons).

        An empty rejection_reasons list means all filters passed.
        """
        rejections = []

        # --- Session filter ---
        if not self.passes_session_filter(pair, timestamp):
            rejections.append(f"Outside high-liquidity session (hour={_extract_hour(timestamp)})")

        # --- Volatility filter ---
        atr_ratio = _get(latest_row, "ATR_RATIO")
        if atr_ratio is not None:
            if not (ATR_RATIO_MIN <= atr_ratio <= ATR_RATIO_MAX):
                rejections.append(f"ATR ratio out of bounds ({atr_ratio:.2f}, need {ATR_RATIO_MIN}-{ATR_RATIO_MAX})")

        # --- Confluence filter ---
        confluence = self.count_confluence(direction, latest_row)
        if confluence < MIN_CONFLUENCE:
            rejections.append(f"Insufficient confluence ({confluence}/{MIN_CONFLUENCE} indicators agree)")

        # --- Noise filter ---
        close = _get(latest_row, "Close")
        atr = _get(latest_row, "ATR")
        if close is not None and atr is not None:
            if not self.passes_noise_filter(predicted_return, atr, close):
                noise_floor = atr / close if close > 0 else 0
                rejections.append(
                    f"Predicted return too small (|{predicted_return:.6f}| < noise floor {noise_floor * NOISE_MULTIPLIER:.6f})"
                )

        passes = len(rejections) == 0

        if not passes:
            for reason in rejections:
                logger.info(f"[TradeFilter] {pair} REJECTED: {reason}")
        else:
            logger.debug(f"[TradeFilter] {pair} PASSED all filters (confluence={confluence})")

        return passes, rejections


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _get(row, key):
    """Safely extract a value from a pandas Series or dict."""
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _extract_hour(timestamp) -> int:
    """Best-effort hour extraction for logging."""
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return timestamp.hour
        return timestamp.astimezone(timezone.utc).hour
    return getattr(timestamp, "hour", -1)
