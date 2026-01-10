"""
Unit tests for utils module.

Tests:
- TradingMetrics
- Logger utilities
- GPU utilities
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.metrics import (
    TradingMetrics,
    calculate_returns,
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_var,
    calculate_cvar,
    calculate_ulcer_index,
    calculate_recovery_factor,
    calculate_profit_factor,
    calculate_expectancy,
    calculate_comprehensive_metrics,
    rolling_sharpe
)
from utils.logger import setup_logger, get_logger, log_trade, TradeLogger
from utils.gpu_utils import (
    get_device,
    get_gpu_memory_info,
    clear_gpu_memory,
    GPUMemoryMonitor,
    optimize_model_for_inference
)


class TestTradingMetrics:
    """Tests for TradingMetrics dataclass."""

    def test_creation(self):
        """Test creating metrics object."""
        metrics = TradingMetrics(
            total_return=0.25,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.6
        )

        assert metrics.total_return == 0.25
        assert metrics.sharpe_ratio == 1.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TradingMetrics(total_return=0.25, sharpe_ratio=1.5)
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d['total_return'] == 0.25

    def test_str_representation(self):
        """Test string representation."""
        metrics = TradingMetrics(
            total_return=0.25,
            sharpe_ratio=1.5,
            max_drawdown=0.15
        )
        s = str(metrics)

        assert 'Total Return' in s
        assert 'Sharpe Ratio' in s


class TestCalculateReturns:
    """Tests for calculate_returns function."""

    def test_basic(self):
        """Test basic returns calculation."""
        equity = np.array([100, 110, 105, 115])
        returns = calculate_returns(equity)

        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.10, rel=0.01)
        assert returns[1] == pytest.approx(-0.0454545, rel=0.01)  # (105-110)/110

    def test_constant_equity(self):
        """Test with constant equity."""
        equity = np.array([100, 100, 100])
        returns = calculate_returns(equity)

        assert np.allclose(returns, 0)


class TestCalculateSharpe:
    """Tests for calculate_sharpe function."""

    def test_positive_returns(self):
        """Test with positive returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)

        sharpe = calculate_sharpe(returns)

        assert sharpe > 0

    def test_negative_returns(self):
        """Test with negative returns."""
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.01, 252)

        sharpe = calculate_sharpe(returns)

        assert sharpe < 0

    def test_zero_volatility(self):
        """Test with zero volatility."""
        returns = np.ones(100) * 0.001

        sharpe = calculate_sharpe(returns)

        assert sharpe == 0.0  # Undefined, returns 0

    def test_empty_returns(self):
        """Test with empty returns."""
        returns = np.array([])

        sharpe = calculate_sharpe(returns)

        assert sharpe == 0.0


class TestCalculateSortino:
    """Tests for calculate_sortino function."""

    def test_positive_returns(self):
        """Test with positive returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)

        sortino = calculate_sortino(returns)

        assert sortino > 0

    def test_no_downside(self):
        """Test with no downside returns."""
        returns = np.abs(np.random.randn(100)) * 0.01

        sortino = calculate_sortino(returns)

        # Should be very high or capped
        assert sortino > 10


class TestCalculateMaxDrawdown:
    """Tests for calculate_max_drawdown function."""

    def test_basic(self):
        """Test basic drawdown calculation."""
        equity = np.array([100, 110, 90, 95, 105])

        max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        # Max drawdown is from 110 to 90 = 18.18%
        assert max_dd == pytest.approx(0.1818, rel=0.01)
        assert peak_idx == 1
        assert trough_idx == 2

    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity = np.array([100, 105, 110, 115, 120])

        max_dd, _, _ = calculate_max_drawdown(equity)

        assert max_dd == 0.0


class TestCalculateDrawdownSeries:
    """Tests for calculate_drawdown_series function."""

    def test_basic(self):
        """Test drawdown series calculation."""
        equity = np.array([100, 110, 90, 95, 100])

        dd_series = calculate_drawdown_series(equity)

        assert len(dd_series) == len(equity)
        assert dd_series[0] == 0.0  # No drawdown at start
        assert dd_series[2] == pytest.approx(0.1818, rel=0.01)


class TestCalculateVaR:
    """Tests for calculate_var function."""

    def test_basic(self):
        """Test VaR calculation."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)

        var_95 = calculate_var(returns, confidence=0.95)

        # 95% VaR should be negative (left tail)
        assert var_95 < 0
        # Should be approximately -1.645 * 0.01 = -0.01645
        assert var_95 == pytest.approx(-0.0165, rel=0.2)


class TestCalculateCVaR:
    """Tests for calculate_cvar function."""

    def test_basic(self):
        """Test CVaR calculation."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)

        cvar_95 = calculate_cvar(returns, confidence=0.95)

        # CVaR should be more negative than VaR
        var_95 = calculate_var(returns, confidence=0.95)
        assert cvar_95 < var_95


class TestCalculateUlcerIndex:
    """Tests for calculate_ulcer_index function."""

    def test_basic(self):
        """Test Ulcer Index calculation."""
        equity = np.array([100, 110, 90, 95, 100, 105])

        ui = calculate_ulcer_index(equity)

        assert ui > 0


class TestCalculateRecoveryFactor:
    """Tests for calculate_recovery_factor function."""

    def test_basic(self):
        """Test recovery factor calculation."""
        rf = calculate_recovery_factor(
            total_return=0.30,
            max_drawdown=0.10
        )

        assert rf == pytest.approx(3.0, rel=0.01)

    def test_zero_drawdown(self):
        """Test with zero drawdown."""
        rf = calculate_recovery_factor(
            total_return=0.20,
            max_drawdown=0.0
        )

        assert rf > 0  # Should be high value


class TestCalculateProfitFactor:
    """Tests for calculate_profit_factor function."""

    def test_basic(self):
        """Test profit factor calculation."""
        pf = calculate_profit_factor(
            winning_pnl=1000,
            losing_pnl=-500
        )

        assert pf == 2.0

    def test_no_losses(self):
        """Test with no losses."""
        pf = calculate_profit_factor(
            winning_pnl=1000,
            losing_pnl=0
        )

        assert pf > 0  # Should be high value


class TestCalculateExpectancy:
    """Tests for calculate_expectancy function."""

    def test_basic(self):
        """Test expectancy calculation."""
        # 60% win rate, $500 avg win, $300 avg loss
        # E = 0.6 * 500 - 0.4 * 300 = 300 - 120 = 180
        expectancy = calculate_expectancy(
            win_rate=0.6,
            avg_win=500,
            avg_loss=300
        )

        assert expectancy == pytest.approx(180, rel=0.01)

    def test_negative_expectancy(self):
        """Test negative expectancy."""
        expectancy = calculate_expectancy(
            win_rate=0.4,
            avg_win=200,
            avg_loss=300
        )

        assert expectancy < 0


class TestCalculateComprehensiveMetrics:
    """Tests for calculate_comprehensive_metrics function."""

    def test_with_equity_only(self, sample_equity_curve):
        """Test with just equity curve."""
        metrics = calculate_comprehensive_metrics(sample_equity_curve)

        assert isinstance(metrics, TradingMetrics)
        assert metrics.total_return > 0
        assert metrics.sharpe_ratio > 0
        assert 0 < metrics.max_drawdown < 1

    def test_with_trades(self, sample_equity_curve, sample_trades):
        """Test with trades provided."""
        metrics = calculate_comprehensive_metrics(
            sample_equity_curve,
            trades=sample_trades
        )

        assert metrics.total_trades == 10
        assert metrics.win_rate == 0.6
        assert metrics.avg_holding_time > 0


class TestRollingSharpe:
    """Tests for rolling_sharpe function."""

    def test_basic(self):
        """Test rolling Sharpe calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 200)

        rolling = rolling_sharpe(returns, window=63)

        assert len(rolling) == len(returns)
        # First 62 values should be NaN
        assert np.isnan(rolling[:62]).all()


class TestSetupLogger:
    """Tests for logger setup."""

    def test_setup(self):
        """Test logger setup."""
        from loguru import logger as loguru_logger

        # Use a simple directory that won't be auto-deleted
        tmpdir = tempfile.mkdtemp()
        try:
            setup_logger(
                log_dir=tmpdir,
                log_level="DEBUG",
                console=False,
                file=True
            )

            logger = get_logger("test")
            logger.info("Test message")
        finally:
            # Remove all handlers to release file locks
            loguru_logger.remove()
            # Manual cleanup - ignore errors on Windows
            import shutil
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors on Windows

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test_module")

        assert logger is not None


class TestTradeLogger:
    """Tests for TradeLogger context manager."""

    def test_successful_operation(self):
        """Test successful operation logging."""
        with TradeLogger("test_operation") as tl:
            # Simulate some work
            _ = 1 + 1

        # Should complete without error

    def test_failed_operation(self):
        """Test failed operation logging."""
        with pytest.raises(ValueError):
            with TradeLogger("failing_operation"):
                raise ValueError("Test error")


class TestGPUUtils:
    """Tests for GPU utilities."""

    def test_get_device(self):
        """Test device detection."""
        device = get_device(prefer_gpu=True)

        assert device is not None
        if torch.cuda.is_available():
            assert device.type == 'cuda'
        else:
            assert device.type == 'cpu'

    def test_get_device_cpu_only(self):
        """Test forcing CPU."""
        device = get_device(prefer_gpu=False)

        assert device.type == 'cpu'

    def test_memory_info(self):
        """Test GPU memory info."""
        total, allocated, cached = get_gpu_memory_info()

        # Should return values (may be 0 if no GPU)
        assert total >= 0
        assert allocated >= 0
        assert cached >= 0

    def test_clear_memory(self):
        """Test memory clearing."""
        # Should not raise error
        clear_gpu_memory()

    def test_memory_monitor(self):
        """Test GPU memory monitor context manager."""
        with GPUMemoryMonitor(label="test") as monitor:
            # Some operation
            x = torch.randn(100, 100)

        # Should complete without error

    def test_optimize_model(self):
        """Test model optimization for inference."""
        model = torch.nn.Linear(10, 5)

        optimized = optimize_model_for_inference(model)

        assert not any(p.requires_grad for p in optimized.parameters())


class TestLogTrade:
    """Tests for log_trade function."""

    def test_buy_trade(self):
        """Test logging buy trade."""
        # Should not raise error
        log_trade(
            action="BUY",
            symbol="NQ",
            quantity=1,
            price=15000.0
        )

    def test_close_trade_with_pnl(self):
        """Test logging close trade with P&L."""
        log_trade(
            action="CLOSE",
            symbol="NQ",
            quantity=1,
            price=15100.0,
            pnl=2000.0
        )

    def test_with_extra_kwargs(self):
        """Test logging with extra parameters."""
        log_trade(
            action="BUY",
            symbol="NQ",
            quantity=1,
            price=15000.0,
            stop=14900.0,
            target=15200.0,
            confidence=0.85
        )
