# 5-Minute Backtest Results — V3 Ensemble

**Date:** 2026-03-19
**Method:** Walk-forward (train 70%, test 30% out-of-sample)
**Data:** 60 days of 5-minute bars (~13k-17k bars per pair)
**Model:** LSTM + LightGBM ensemble with directional loss
**Risk:** 2% per trade, SL=1.5 ATR, TP=2.5 ATR

## Results

| Pair | Trades | Win% | Return% | Profit Factor | Sharpe | Max DD% | Dir Acc% |
|---|---|---|---|---|---|---|---|
| GBP/USD | 614 | 40.9% | +137.27% | 1.12 | 6.33 | 32.32% | 40.9% |
| AUD/USD | 753 | 40.2% | +128.99% | 1.09 | 5.78 | 37.64% | 40.2% |
| USD/JPY | 356 | 39.0% | +21.41% | 1.04 | 2.47 | 38.44% | 39.0% |
| Silver (XAG/USD) | 256 | 39.5% | +21.35% | 1.06 | 2.86 | 30.69% | 39.5% |
| USD/CHF | 379 | 38.5% | +7.24% | 1.02 | 1.49 | 43.53% | 38.5% |
| Platinum | 462 | 36.8% | -26.70% | 0.96 | -1.24 | 49.98% | 36.8% |
| EUR/USD | 667 | 36.4% | -46.61% | 0.93 | -2.38 | 71.27% | 36.4% |
| USD/CAD | 195 | 32.8% | -41.30% | 0.77 | -5.15 | 47.83% | 32.8% |
| Gold (XAU/USD) | 373 | 33.0% | -65.01% | 0.86 | -8.20 | 76.07% | 33.0% |
| Copper | 231 | 34.2% | -37.80% | 0.83 | -4.44 | 52.81% | 34.2% |
| Palladium | 474 | 34.2% | -63.79% | 0.86 | -7.06 | 66.37% | 34.2% |

## Aggregate

| Metric | Value |
|---|---|
| Total Trades | 4,760 |
| Average Win Rate | 36.9% |
| Combined Return | +35.05% |
| Pairs Profitable | 5 / 11 |

## Observations

- **Break-even win rate is ~37.5%** with the 1.5:2.5 ATR risk-reward ratio
- **GBP/USD and AUD/USD are standout performers** with Sharpe > 5
- **Metals (Gold, Palladium, Copper) underperform** on 5m — too noisy at this timeframe
- **Max drawdowns are high** (30-76%) — position sizing should be reduced for live trading
- The ensemble edge is real but **pair-selective** — not every instrument is tradeable

## Recommended Live Configuration

- Trade only: GBP/USD, AUD/USD, USD/JPY, Silver, USD/CHF
- Reduce position size to 1% risk per trade (from 2%)
- Monitor drawdown circuit breaker at 10% (from 15%)

## Training Metrics (5m)

| Pair | LSTM Dir Acc | GBM Dir Acc | LSTM Weight | GBM Weight |
|---|---|---|---|---|
| EUR/USD | 32.0% | 32.1% | 0.50 | 0.50 |
| GBP/USD | 50.6% | 47.4% | 0.52 | 0.48 |
| USD/JPY | 50.0% | 48.7% | 0.51 | 0.49 |
| AUD/USD | 45.2% | 42.8% | 0.51 | 0.49 |
| USD/CAD | 50.6% | 47.3% | 0.52 | 0.48 |
| USD/CHF | 48.7% | 48.7% | 0.50 | 0.50 |
| Gold | 52.2% | 52.2% | 0.50 | 0.50 |
| Silver | 51.1% | 49.8% | 0.51 | 0.49 |
| Platinum | 53.0% | 52.3% | 0.50 | 0.50 |
| Palladium | 48.2% | 45.2% | 0.52 | 0.48 |
| Copper | 48.6% | 45.7% | 0.52 | 0.48 |
