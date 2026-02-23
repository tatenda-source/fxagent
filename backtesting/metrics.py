import numpy as np


def compute_metrics(trades: list, equity_curve: list, initial_balance: float) -> dict:
    """Compute standard trading performance metrics."""
    if not trades:
        return {
            "total_return": 0, "num_trades": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "sharpe_ratio": 0, "max_drawdown": 0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (equity_curve[-1] - initial_balance) / initial_balance if initial_balance > 0 else 0
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = abs(float(np.mean(losses))) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    # Sharpe ratio (annualized from daily)
    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    return {
        "total_return": round(total_return * 100, 2),
        "num_trades": len(trades),
        "win_rate": round(win_rate * 100, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown": round(max_drawdown * 100, 2),
    }
