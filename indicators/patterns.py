import numpy as np
import pandas as pd


def find_support_resistance(df: pd.DataFrame, window: int = 20, num_levels: int = 5) -> dict:
    """Identify support and resistance levels using local min/max."""
    highs = df["High"].values
    lows = df["Low"].values

    resistance_levels = []
    support_levels = []

    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            resistance_levels.append(highs[i])
        if lows[i] == min(lows[i - window: i + window + 1]):
            support_levels.append(lows[i])

    resistance_levels = _cluster_levels(resistance_levels, threshold=0.001)
    support_levels = _cluster_levels(support_levels, threshold=0.001)

    return {
        "support": sorted(support_levels)[:num_levels],
        "resistance": sorted(resistance_levels, reverse=True)[:num_levels],
    }


def _cluster_levels(levels: list, threshold: float) -> list:
    """Merge nearby price levels into clusters, return mean of each."""
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for price in levels[1:]:
        if clusters[-1][-1] != 0 and (price - clusters[-1][-1]) / clusters[-1][-1] < threshold:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    clusters.sort(key=len, reverse=True)
    return [float(np.mean(c)) for c in clusters]
