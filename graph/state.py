"""Typed state management for the trading pipeline.

Replaces the untyped context dict with a structured TypedDict that enforces
contracts between pipeline stages. Inspired by LangGraph's AgentState pattern.
"""

from typing import TypedDict, Optional, List, Dict, Any
import pandas as pd


class RegimeInfo(TypedDict):
    regime: str                     # "trending" | "ranging" | "volatile" | "unknown"
    adx: float
    trend_direction: str            # "up" | "down" | "flat"
    volatility_state: str           # "normal" | "high" | "extreme"
    volatility_ratio: float
    confidence: float
    strategy_adjustments: Dict[str, float]


class PredictionInfo(TypedDict):
    predicted_price: float
    current_price: float
    direction: str                  # "UP" | "DOWN"
    confidence: float
    change_pct: float
    predicted_return: float
    uncertainty: float


class SignalInfo(TypedDict):
    pair: str
    signal_type: str                # "BUY" | "SELL"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasons: List[str]
    predicted_price: float
    regime: str


class LLMAnalysis(TypedDict, total=False):
    news_report: str                # News agent's analysis
    macro_report: str               # Macro/fundamental analysis
    sentiment_score: float          # -1.0 (bearish) to 1.0 (bullish)
    key_events: List[str]           # Upcoming events that could move the pair
    risk_factors: List[str]         # Identified risk factors


class DebateResult(TypedDict, total=False):
    bull_case: str                  # Bull researcher's argument
    bear_case: str                  # Bear researcher's argument
    judge_verdict: str              # Research manager's decision
    debate_rounds: int              # Number of debate rounds
    conviction: str                 # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    reasoning: str                  # Judge's explanation


class ReflectionEntry(TypedDict, total=False):
    pair: str
    signal_id: int
    outcome: str                    # "TP_HIT" | "SL_HIT"
    pnl: float
    lessons: str                    # LLM-generated post-mortem
    what_worked: List[str]
    what_failed: List[str]
    timestamp: str


class PipelineState(TypedDict, total=False):
    # Input
    pairs: List[str]
    fetch_intraday: bool

    # Stage 1: Data
    ohlcv_data: Dict[str, Any]          # pair -> DataFrame
    intraday_data: Dict[str, Any]       # pair -> DataFrame

    # Stage 2-4: Analysis
    regimes: Dict[str, RegimeInfo]
    correlation_matrix: Any             # pd.DataFrame
    analyzed_data: Dict[str, Any]       # pair -> DataFrame with indicators
    sr_levels: Dict[str, Dict[str, list]]

    # Stage 5: Prediction
    predictions: Dict[str, PredictionInfo]

    # Stage 6: LLM Analysis (NEW)
    llm_analyses: Dict[str, LLMAnalysis]    # pair -> LLM analysis

    # Stage 7: Debate (NEW)
    debates: Dict[str, DebateResult]        # pair -> debate result

    # Stage 8: Recommendation
    signals: List[SignalInfo]

    # Stage 9: Portfolio Filter
    portfolio_status: Dict[str, Any]

    # Stage 10: Logging & Feedback
    feedback: Dict[str, Any]

    # Stage 11: Reflection (NEW)
    reflections: List[ReflectionEntry]

    # Meta
    pipeline_health: Dict[str, Dict[str, int]]
    past_reflections: Dict[str, List[str]]  # pair -> relevant past lessons
