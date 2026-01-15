"""
Fitness evaluation functions for genetic algorithm optimization.

Implements various risk-adjusted return metrics for evaluating
trading model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradeResult:
    """Container for individual trade results."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float  # Positive for long, negative for short
    pnl: float
    pnl_pct: float
    holding_time_minutes: float
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    avg_holding_time: float
    expectancy: float


class FitnessEvaluator:
    """
    Evaluates trading strategy fitness using multiple metrics.

    Supports:
    - Sharpe Ratio
    - Sortino Ratio
    - Calmar Ratio
    - Profit Factor
    - Custom weighted combinations
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor for annualizing returns (252 for daily)
            weights: Custom weights for combined fitness
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

        # Default weights for combined fitness
        self.weights = weights or {
            'sharpe': 0.4,
            'calmar': 0.3,
            'profit_factor': 0.2,
            'win_rate': 0.1
        }

    def calculate_returns(
        self,
        equity_curve: np.ndarray
    ) -> np.ndarray:
        """Calculate returns from equity curve."""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array of returns

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.annualization_factor
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return < 1e-8:
            return 0.0

        sharpe = mean_return / std_return * np.sqrt(self.annualization_factor)
        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Calculate annualized Sortino ratio.
        Uses downside deviation instead of standard deviation.

        Args:
            returns: Array of returns

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.annualization_factor
        mean_return = np.mean(excess_returns)

        # Downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            downside_std = 1e-8
        else:
            downside_std = np.std(downside_returns)

        if downside_std < 1e-8:
            return 0.0

        sortino = mean_return / downside_std * np.sqrt(self.annualization_factor)
        return float(sortino)

    def calculate_max_drawdown(
        self,
        equity_curve: np.ndarray
    ) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Array of equity values

        Returns:
            Maximum drawdown as positive decimal
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown)

        return float(max_dd)

    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray
    ) -> float:
        """
        Calculate Calmar ratio (CAGR / Max Drawdown).

        Args:
            returns: Array of returns
            equity_curve: Array of equity values

        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        # Calculate CAGR
        total_return = equity_curve[-1] / equity_curve[0] - 1
        years = len(returns) / self.annualization_factor

        # Handle edge case: total_return < -1 (lost more than 100%)
        if total_return <= -1.0:
            return -10.0  # Very negative calmar for complete loss

        if years < 0.1:
            cagr = total_return * (1 / years)
        else:
            cagr = (1 + total_return) ** (1 / years) - 1

        # Max drawdown
        max_dd = self.calculate_max_drawdown(equity_curve)

        if max_dd < 1e-8:
            return min(cagr * 100, 10.0)  # Cap at reasonable value

        calmar = cagr / max_dd
        # Cap calmar to reasonable range
        return float(max(min(calmar, 10.0), -10.0))

    def calculate_profit_factor(
        self,
        trades: List[TradeResult]
    ) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of trade results

        Returns:
            Profit factor
        """
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss < 1e-8:
            return gross_profit * 100 if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    def calculate_win_rate(
        self,
        trades: List[TradeResult]
    ) -> float:
        """
        Calculate win rate.

        Args:
            trades: List of trade results

        Returns:
            Win rate as decimal
        """
        if not trades:
            return 0.0

        wins = sum(1 for t in trades if t.pnl > 0)
        return float(wins / len(trades))

    def calculate_expectancy(
        self,
        trades: List[TradeResult]
    ) -> float:
        """
        Calculate expectancy (average P&L per trade).

        Args:
            trades: List of trade results

        Returns:
            Expectancy
        """
        if not trades:
            return 0.0

        return float(np.mean([t.pnl for t in trades]))

    def evaluate_trades(
        self,
        trades: List[TradeResult],
        initial_capital: float = 100000.0
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics from trade list.

        Args:
            trades: List of trade results
            initial_capital: Starting capital

        Returns:
            PerformanceMetrics object
        """
        if not trades:
            return PerformanceMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                total_trades=0,
                avg_holding_time=0.0,
                expectancy=0.0
            )

        # Build equity curve
        equity = [initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)
        equity_curve = np.array(equity)

        returns = self.calculate_returns(equity_curve)

        # Win/loss statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0

        return PerformanceMetrics(
            total_return=(equity_curve[-1] / equity_curve[0]) - 1,
            sharpe_ratio=self.calculate_sharpe_ratio(returns),
            sortino_ratio=self.calculate_sortino_ratio(returns),
            calmar_ratio=self.calculate_calmar_ratio(returns, equity_curve),
            max_drawdown=self.calculate_max_drawdown(equity_curve),
            win_rate=self.calculate_win_rate(trades),
            profit_factor=self.calculate_profit_factor(trades),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            total_trades=len(trades),
            avg_holding_time=float(np.mean([t.holding_time_minutes for t in trades])),
            expectancy=self.calculate_expectancy(trades)
        )

    def evaluate_equity_curve(
        self,
        equity_curve: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate metrics from equity curve.

        Args:
            equity_curve: Array of equity values

        Returns:
            Dict of metrics
        """
        returns = self.calculate_returns(equity_curve)

        return {
            'total_return': float(equity_curve[-1] / equity_curve[0] - 1),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns, equity_curve),
            'max_drawdown': self.calculate_max_drawdown(equity_curve)
        }

    def calculate_fitness(
        self,
        metrics: PerformanceMetrics
    ) -> float:
        """
        Calculate combined fitness score.

        Key insight: During early training, models may not trade much.
        We should encourage ANY trading activity while still rewarding
        good performance.

        Args:
            metrics: Performance metrics

        Returns:
            Combined fitness score
        """
        # =========================================================
        # TRADE ACTIVITY BONUS (Critical for early training)
        # =========================================================
        # Give a base score just for making trades - this encourages
        # the model to learn to trade rather than stay flat
        if metrics.total_trades == 0:
            # No trades = small negative fitness (not -1000)
            # This allows models that at least vary their signals to survive
            return -0.5

        # Handle NaN values in metrics
        import math
        if math.isnan(metrics.sharpe_ratio):
            metrics = PerformanceMetrics(
                total_return=metrics.total_return,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                avg_win=metrics.avg_win,
                avg_loss=metrics.avg_loss,
                total_trades=metrics.total_trades,
                avg_holding_time=metrics.avg_holding_time,
                expectancy=metrics.expectancy
            )

        # Base fitness for making trades (scaled by number of trades)
        trade_bonus = min(metrics.total_trades / 50.0, 1.0) * 0.3
        fitness = trade_bonus

        # =========================================================
        # PERFORMANCE METRICS
        # =========================================================
        # Sharpe component (target: 2.0)
        if metrics.sharpe_ratio > 0:
            sharpe_score = min(metrics.sharpe_ratio / 2.0, 1.0)
        else:
            sharpe_score = max(metrics.sharpe_ratio / 2.0, -1.0)  # Cap negative
        fitness += self.weights.get('sharpe', 0.4) * sharpe_score

        # Calmar component (target: 3.0)
        if metrics.calmar_ratio > 0:
            calmar_score = min(metrics.calmar_ratio / 3.0, 1.0)
        else:
            calmar_score = max(metrics.calmar_ratio / 3.0, -1.0)
        fitness += self.weights.get('calmar', 0.3) * calmar_score

        # Profit factor component (target: 2.0)
        if metrics.profit_factor > 0:
            pf_score = min(metrics.profit_factor / 2.0, 1.0)
        else:
            pf_score = 0.0
        fitness += self.weights.get('profit_factor', 0.2) * pf_score

        # Win rate component (target: 0.55)
        if metrics.win_rate > 0:
            wr_score = min(metrics.win_rate / 0.55, 1.2)
        else:
            wr_score = 0.0
        fitness += self.weights.get('win_rate', 0.1) * wr_score

        # =========================================================
        # PENALTIES (Gradual, not multiplicative)
        # =========================================================
        # Penalty for too few trades (gradual, not zero-out)
        if metrics.total_trades < 20:
            trade_penalty = 0.2 * (1.0 - metrics.total_trades / 20.0)
            fitness -= trade_penalty

        # Penalty for excessive drawdown
        if metrics.max_drawdown > 0.15:
            dd_penalty = 0.3 * min((metrics.max_drawdown - 0.15) / 0.35, 1.0)
            fitness -= dd_penalty

        return float(fitness)

    def calculate_fitness_from_returns(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray
    ) -> float:
        """
        Quick fitness calculation from returns and equity curve.

        Args:
            returns: Array of returns
            equity_curve: Array of equity values

        Returns:
            Fitness score
        """
        sharpe = self.calculate_sharpe_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, equity_curve)
        max_dd = self.calculate_max_drawdown(equity_curve)

        # Simple combined score
        fitness = 0.5 * sharpe + 0.3 * calmar - 0.2 * max_dd * 10

        return float(fitness)


def simulate_trades_from_signals(
    signals: np.ndarray,
    prices: np.ndarray,
    timestamps: pd.DatetimeIndex,
    initial_capital: float = 100000.0,
    commission: float = 2.25,
    slippage: float = 0.25,
    point_value: float = 20.0,
    use_signal_changes: bool = True
) -> Tuple[List[TradeResult], np.ndarray]:
    """
    Simulate trades from model signals.

    Two modes:
    1. use_signal_changes=True (default): Trade based on CHANGES in signal
       - More robust to constant-bias models
       - Signal crossing thresholds triggers trades
    2. use_signal_changes=False: Trade based on absolute signal value
       - Traditional approach

    Args:
        signals: Array of position signals (-1 to 1)
        prices: Array of prices
        timestamps: Array of timestamps
        initial_capital: Starting capital
        commission: Commission per contract per side
        slippage: Slippage in points
        point_value: Dollar value per point
        use_signal_changes: If True, use signal changes for trading

    Returns:
        (list of trades, equity curve)
    """
    trades = []
    equity = [initial_capital]
    current_equity = initial_capital

    position = 0
    entry_price = 0
    entry_time = None
    entry_idx = 0
    high_since_entry = 0
    low_since_entry = float('inf')

    # Precompute signal statistics for adaptive thresholds
    signal_mean = np.mean(signals)
    signal_std = np.std(signals)

    # If signals have very low variance, use centered signals
    if signal_std < 0.1:
        # Center signals around mean to detect relative changes
        centered_signals = signals - signal_mean
    else:
        centered_signals = signals

    for i in range(1, len(signals)):
        signal = signals[i]
        centered_signal = centered_signals[i]
        price = prices[i]
        timestamp = timestamps[i]

        # Track high/low since entry
        if position != 0:
            high_since_entry = max(high_since_entry, price)
            low_since_entry = min(low_since_entry, price)

        # Determine new position
        prev_position = position

        if use_signal_changes and signal_std < 0.1:
            # Low variance mode: use centered signals with adaptive threshold
            # This helps when model outputs constant-ish values
            threshold = max(0.01, signal_std * 0.5)
            if centered_signal > threshold:
                new_position = 1
            elif centered_signal < -threshold:
                new_position = -1
            else:
                new_position = 0
        else:
            # Normal mode: use absolute signal value
            # Threshold at 0.1 to require some conviction
            if signal > 0.1:
                new_position = 1
            elif signal < -0.1:
                new_position = -1
            else:
                new_position = 0

        if new_position != prev_position:
            # Close existing position
            if position != 0:
                exit_price = price - slippage * np.sign(position)
                pnl = (exit_price - entry_price) * position * point_value - commission * 2

                # MAE/MFE
                if position > 0:
                    mae = (entry_price - low_since_entry) / entry_price if entry_price > 0 else 0
                    mfe = (high_since_entry - entry_price) / entry_price if entry_price > 0 else 0
                else:
                    mae = (high_since_entry - entry_price) / entry_price if entry_price > 0 else 0
                    mfe = (entry_price - low_since_entry) / entry_price if entry_price > 0 else 0

                holding_time = (timestamp - entry_time).total_seconds() / 60.0

                trades.append(TradeResult(
                    entry_time=entry_time,
                    exit_time=timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size=float(position),
                    pnl=pnl,
                    pnl_pct=pnl / current_equity if current_equity > 0 else 0,
                    holding_time_minutes=holding_time,
                    mae=mae,
                    mfe=mfe
                ))

                current_equity += pnl

            # Open new position
            if new_position != 0:
                entry_price = price + slippage * np.sign(new_position)
                entry_time = timestamp
                entry_idx = i
                high_since_entry = price
                low_since_entry = price

            position = new_position

        equity.append(current_equity)

    return trades, np.array(equity)
