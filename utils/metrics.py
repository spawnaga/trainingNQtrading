"""
Trading performance metrics and analysis utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradingMetrics:
    """
    Comprehensive trading performance metrics.
    """

    # Returns metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Trade metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0

    # Time metrics
    avg_holding_time: float = 0.0
    avg_bars_in_trade: int = 0
    time_in_market: float = 0.0

    # Recovery metrics
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'expectancy': self.expectancy,
            'avg_holding_time': self.avg_holding_time
        }

    def __str__(self) -> str:
        return f"""
Trading Performance Metrics
===========================
Returns:
  Total Return:     {self.total_return:>10.2%}
  Annualized:       {self.annualized_return:>10.2%}
  Sharpe Ratio:     {self.sharpe_ratio:>10.2f}
  Sortino Ratio:    {self.sortino_ratio:>10.2f}
  Calmar Ratio:     {self.calmar_ratio:>10.2f}

Risk:
  Max Drawdown:     {self.max_drawdown:>10.2%}
  Volatility:       {self.volatility:>10.2%}
  VaR (95%):        {self.var_95:>10.2%}

Trades:
  Total Trades:     {self.total_trades:>10d}
  Win Rate:         {self.win_rate:>10.2%}
  Profit Factor:    {self.profit_factor:>10.2f}
  Avg Win:          ${self.avg_win:>9.2f}
  Avg Loss:         ${self.avg_loss:>9.2f}
  Expectancy:       ${self.expectancy:>9.2f}
"""


def calculate_returns(equity_curve: np.ndarray) -> np.ndarray:
    """Calculate returns from equity curve."""
    return np.diff(equity_curve) / equity_curve[:-1]


def calculate_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0
) -> float:
    """
    Calculate annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / annualization
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return < 1e-8:
        return 0.0

    return float(mean_return / std_return * np.sqrt(annualization))


def calculate_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0
) -> float:
    """
    Calculate annualized Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / annualization
    mean_return = np.mean(excess_returns)

    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2:
        return mean_return * np.sqrt(annualization) * 100

    downside_std = np.std(downside_returns)
    if downside_std < 1e-8:
        return 0.0

    return float(mean_return / downside_std * np.sqrt(annualization))


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Returns:
        (max_drawdown, peak_idx, trough_idx)
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak

    max_dd = np.max(drawdown)
    trough_idx = np.argmax(drawdown)

    # Find peak before trough
    peak_idx = np.argmax(equity_curve[:trough_idx + 1])

    return float(max_dd), int(peak_idx), int(trough_idx)


def calculate_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """Calculate drawdown series."""
    peak = np.maximum.accumulate(equity_curve)
    return (peak - equity_curve) / peak


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk.
    """
    if len(returns) < 10:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    """
    if len(returns) < 10:
        return 0.0
    var = calculate_var(returns, confidence)
    return float(np.mean(returns[returns <= var]))


def calculate_ulcer_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Ulcer Index (measures depth and duration of drawdowns).
    """
    drawdowns = calculate_drawdown_series(equity_curve)
    return float(np.sqrt(np.mean(drawdowns ** 2)))


def calculate_recovery_factor(
    total_return: float,
    max_drawdown: float
) -> float:
    """Calculate recovery factor (return / max drawdown)."""
    if max_drawdown < 1e-8:
        return total_return * 100
    return total_return / max_drawdown


def calculate_profit_factor(winning_pnl: float, losing_pnl: float) -> float:
    """Calculate profit factor."""
    if abs(losing_pnl) < 1e-8:
        return winning_pnl * 100 if winning_pnl > 0 else 0.0
    return abs(winning_pnl / losing_pnl)


def calculate_expectancy(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate trading expectancy.

    E = (Win% * Avg Win) - (Loss% * Avg Loss)
    """
    return win_rate * avg_win - (1 - win_rate) * abs(avg_loss)


def calculate_comprehensive_metrics(
    equity_curve: np.ndarray,
    trades: Optional[List[Dict]] = None,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0
) -> TradingMetrics:
    """
    Calculate comprehensive trading metrics.

    Args:
        equity_curve: Array of equity values
        trades: Optional list of trade dicts with 'pnl', 'holding_time' keys
        annualization_factor: Factor for annualizing metrics
        risk_free_rate: Risk-free rate for Sharpe calculation

    Returns:
        TradingMetrics object
    """
    metrics = TradingMetrics()

    if len(equity_curve) < 2:
        return metrics

    # Calculate returns
    returns = calculate_returns(equity_curve)

    # Returns metrics
    metrics.total_return = (equity_curve[-1] / equity_curve[0]) - 1

    years = len(returns) / annualization_factor
    if years > 0:
        metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1

    metrics.sharpe_ratio = calculate_sharpe(returns, risk_free_rate, annualization_factor)
    metrics.sortino_ratio = calculate_sortino(returns, risk_free_rate, annualization_factor)

    # Risk metrics
    metrics.max_drawdown, _, _ = calculate_max_drawdown(equity_curve)
    metrics.avg_drawdown = np.mean(calculate_drawdown_series(equity_curve))
    metrics.volatility = np.std(returns) * np.sqrt(annualization_factor)
    metrics.var_95 = calculate_var(returns, 0.95)
    metrics.cvar_95 = calculate_cvar(returns, 0.95)
    metrics.ulcer_index = calculate_ulcer_index(equity_curve)

    # Calmar ratio
    if metrics.max_drawdown > 0:
        metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

    # Recovery factor
    metrics.recovery_factor = calculate_recovery_factor(
        metrics.total_return, metrics.max_drawdown
    )

    # Trade metrics (if trades provided)
    if trades:
        metrics.total_trades = len(trades)

        pnls = [t.get('pnl', 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        metrics.win_rate = len(wins) / len(pnls) if pnls else 0
        metrics.avg_win = np.mean(wins) if wins else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        metrics.largest_win = max(wins) if wins else 0
        metrics.largest_loss = min(losses) if losses else 0
        metrics.avg_trade = np.mean(pnls) if pnls else 0

        winning_pnl = sum(wins)
        losing_pnl = sum(losses)
        metrics.profit_factor = calculate_profit_factor(winning_pnl, losing_pnl)

        metrics.expectancy = calculate_expectancy(
            metrics.win_rate, metrics.avg_win, abs(metrics.avg_loss)
        )

        # Time metrics
        holding_times = [t.get('holding_time', 0) for t in trades]
        metrics.avg_holding_time = np.mean(holding_times) if holding_times else 0

    return metrics


def rolling_sharpe(
    returns: np.ndarray,
    window: int = 63,  # ~3 months
    annualization: float = 252.0
) -> np.ndarray:
    """Calculate rolling Sharpe ratio."""
    rolling = pd.Series(returns).rolling(window)
    mean = rolling.mean()
    std = rolling.std()
    return (mean / std * np.sqrt(annualization)).values


def rolling_drawdown(equity_curve: np.ndarray, window: int = 63) -> np.ndarray:
    """Calculate rolling maximum drawdown."""
    result = np.zeros(len(equity_curve))

    for i in range(window, len(equity_curve)):
        window_equity = equity_curve[i - window:i + 1]
        dd, _, _ = calculate_max_drawdown(window_equity)
        result[i] = dd

    return result
