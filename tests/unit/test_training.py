"""
Unit tests for training module.

Tests:
- FitnessEvaluator
- GeneticOptimizer
- Trainer
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.fitness import (
    FitnessEvaluator,
    TradeResult,
    PerformanceMetrics,
    simulate_trades_from_signals
)
from training.genetic_optimizer import (
    GeneticOptimizer,
    HyperparameterSpace,
    Individual
)
from training.trainer import Trainer, TrainingConfig


class TestFitnessEvaluator:
    """Tests for FitnessEvaluator."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = FitnessEvaluator()

        assert evaluator.risk_free_rate == 0.0
        assert evaluator.annualization_factor == 252.0

    def test_calculate_returns(self, sample_equity_curve):
        """Test returns calculation."""
        evaluator = FitnessEvaluator()
        returns = evaluator.calculate_returns(sample_equity_curve)

        assert len(returns) == len(sample_equity_curve) - 1
        assert not np.any(np.isnan(returns))

    def test_sharpe_ratio_positive(self, sample_equity_curve):
        """Test Sharpe ratio with positive returns."""
        evaluator = FitnessEvaluator()
        returns = evaluator.calculate_returns(sample_equity_curve)
        sharpe = evaluator.calculate_sharpe_ratio(returns)

        # With positive drift, Sharpe should be positive
        assert sharpe > 0

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio with negative returns."""
        evaluator = FitnessEvaluator()

        # Generate negative drift equity curve
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.01, 252)
        equity = 100000 * np.cumprod(1 + returns)

        sharpe = evaluator.calculate_sharpe_ratio(evaluator.calculate_returns(equity))
        assert sharpe < 0

    def test_sortino_ratio(self, sample_equity_curve):
        """Test Sortino ratio calculation."""
        evaluator = FitnessEvaluator()
        returns = evaluator.calculate_returns(sample_equity_curve)
        sortino = evaluator.calculate_sortino_ratio(returns)

        # Should be positive for upward trending curve
        assert sortino > 0

    def test_max_drawdown(self, sample_equity_curve):
        """Test max drawdown calculation."""
        evaluator = FitnessEvaluator()
        max_dd = evaluator.calculate_max_drawdown(sample_equity_curve)

        assert 0 <= max_dd <= 1
        assert max_dd > 0  # Some drawdown should exist

    def test_calmar_ratio(self, sample_equity_curve):
        """Test Calmar ratio calculation."""
        evaluator = FitnessEvaluator()
        returns = evaluator.calculate_returns(sample_equity_curve)
        calmar = evaluator.calculate_calmar_ratio(returns, sample_equity_curve)

        # Should be positive for profitable curve
        assert calmar > 0

    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        evaluator = FitnessEvaluator()
        trades = [TradeResult(
            entry_time=pd.Timestamp('2023-01-01'),
            exit_time=pd.Timestamp('2023-01-01 00:30:00'),
            entry_price=15000,
            exit_price=15000 + t['pnl']/20,
            position_size=1.0,
            pnl=t['pnl'],
            pnl_pct=t['pnl']/100000,
            holding_time_minutes=t['holding_time'],
            mae=0.01,
            mfe=0.02
        ) for t in sample_trades]

        pf = evaluator.calculate_profit_factor(trades)

        assert pf > 0
        # Total wins: 500+800+300+600+150+450 = 2800
        # Total losses: 200+100+400+250 = 950
        # PF should be ~2.95
        assert 2.0 < pf < 4.0

    def test_win_rate(self, sample_trades):
        """Test win rate calculation."""
        evaluator = FitnessEvaluator()
        trades = [TradeResult(
            entry_time=pd.Timestamp('2023-01-01'),
            exit_time=pd.Timestamp('2023-01-01'),
            entry_price=15000,
            exit_price=15000,
            position_size=1.0,
            pnl=t['pnl'],
            pnl_pct=0,
            holding_time_minutes=t['holding_time'],
            mae=0,
            mfe=0
        ) for t in sample_trades]

        wr = evaluator.calculate_win_rate(trades)

        # 6 wins out of 10 = 60%
        assert wr == 0.6

    def test_evaluate_trades(self, sample_trades):
        """Test full trade evaluation."""
        evaluator = FitnessEvaluator()
        trades = [TradeResult(
            entry_time=pd.Timestamp('2023-01-01'),
            exit_time=pd.Timestamp('2023-01-01 00:30:00'),
            entry_price=15000,
            exit_price=15000 + t['pnl']/20,
            position_size=1.0,
            pnl=t['pnl'],
            pnl_pct=t['pnl']/100000,
            holding_time_minutes=t['holding_time'],
            mae=0.01,
            mfe=0.02
        ) for t in sample_trades]

        metrics = evaluator.evaluate_trades(trades)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 10
        assert metrics.win_rate == 0.6
        assert metrics.profit_factor > 0

    def test_calculate_fitness(self):
        """Test fitness score calculation."""
        evaluator = FitnessEvaluator()

        metrics = PerformanceMetrics(
            total_return=0.20,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=1.8,
            avg_win=500,
            avg_loss=-300,
            total_trades=100,
            avg_holding_time=30,
            expectancy=100
        )

        fitness = evaluator.calculate_fitness(metrics)

        assert 0 < fitness < 2  # Reasonable fitness range


class TestSimulateTrades:
    """Tests for trade simulation."""

    def test_simulate_basic(self):
        """Test basic trade simulation."""
        n_bars = 1000
        signals = np.zeros(n_bars)
        signals[100:200] = 1.0   # Long signal
        signals[300:400] = -1.0  # Short signal

        np.random.seed(42)
        prices = 15000 + np.cumsum(np.random.randn(n_bars) * 5)
        timestamps = pd.date_range('2023-01-01', periods=n_bars, freq='1min')

        trades, equity = simulate_trades_from_signals(
            signals, prices, timestamps
        )

        assert len(trades) >= 2  # At least 2 trades
        # Equity curve: 1 initial value + (n_bars - 1) loop iterations = n_bars
        assert len(equity) == n_bars

    def test_equity_curve_length(self):
        """Test equity curve has correct length."""
        n_bars = 100
        signals = np.zeros(n_bars)
        prices = np.ones(n_bars) * 15000
        timestamps = pd.date_range('2023-01-01', periods=n_bars, freq='1min')

        _, equity = simulate_trades_from_signals(signals, prices, timestamps)

        # Equity curve: 1 initial value + (n_bars - 1) loop iterations = n_bars
        assert len(equity) == n_bars


class TestHyperparameterSpace:
    """Tests for HyperparameterSpace."""

    def test_default_values(self):
        """Test default search space values."""
        space = HyperparameterSpace()

        assert len(space.d_model) > 0
        assert len(space.n_heads) > 0
        assert len(space.learning_rate) > 0

    def test_custom_values(self):
        """Test custom search space."""
        space = HyperparameterSpace(
            d_model=[64, 128],
            n_heads=[2, 4]
        )

        assert space.d_model == [64, 128]
        assert space.n_heads == [2, 4]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        space = HyperparameterSpace()
        d = space.to_dict()

        assert isinstance(d, dict)
        assert 'd_model' in d
        assert 'learning_rate' in d


class TestIndividual:
    """Tests for Individual class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ind = Individual(
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff_multiplier=4,
            dropout=0.1,
            learning_rate=0.0001,
            batch_size=64,
            weight_decay=0.0001,
            profit_hidden=128,
            risk_hidden=64,
            max_risk_per_trade=0.02,
            atr_multiplier=2.0
        )

        d = ind.to_dict()

        assert d['d_model'] == 256
        assert d['d_ff'] == 256 * 4  # d_model * multiplier

    def test_is_valid(self):
        """Test validity check."""
        # Valid: 256 / 8 = 32
        ind_valid = Individual(
            d_model=256, n_heads=8, n_layers=4, d_ff_multiplier=4,
            dropout=0.1, learning_rate=0.0001, batch_size=64,
            weight_decay=0.0001, profit_hidden=128, risk_hidden=64,
            max_risk_per_trade=0.02, atr_multiplier=2.0
        )
        assert ind_valid.is_valid()

        # Invalid: 256 / 7 is not integer
        ind_invalid = Individual(
            d_model=256, n_heads=7, n_layers=4, d_ff_multiplier=4,
            dropout=0.1, learning_rate=0.0001, batch_size=64,
            weight_decay=0.0001, profit_hidden=128, risk_hidden=64,
            max_risk_per_trade=0.02, atr_multiplier=2.0
        )
        assert not ind_invalid.is_valid()


class TestGeneticOptimizer:
    """Tests for GeneticOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        space = HyperparameterSpace()
        optimizer = GeneticOptimizer(
            search_space=space,
            population_size=10,
            n_generations=5
        )

        assert optimizer.population_size == 10
        assert optimizer.n_generations == 5

    def test_individual_to_dict(self):
        """Test individual conversion."""
        space = HyperparameterSpace()
        optimizer = GeneticOptimizer(space, population_size=10)

        # Create individual list matching expected format
        ind = [256, 8, 4, 4, 0.1, 0.0001, 64, 0.0001, 128, 64, 0.02, 2.0]
        d = optimizer._individual_to_dict(ind)

        assert d['d_model'] == 256
        assert d['n_heads'] == 8

    def test_repair(self):
        """Test individual repair."""
        space = HyperparameterSpace()
        optimizer = GeneticOptimizer(space, population_size=10)

        # Invalid individual
        ind = [256, 7, 4, 4, 0.1, 0.0001, 64, 0.0001, 128, 64, 0.02, 2.0]
        repaired = optimizer._repair(ind)

        # Should be valid after repair
        assert repaired[0] % repaired[1] == 0

    @pytest.mark.slow
    def test_optimize(self):
        """Test optimization run (slow)."""
        space = HyperparameterSpace(
            d_model=[64],
            n_heads=[4],
            n_layers=[2],
            learning_rate=[0.001]
        )

        optimizer = GeneticOptimizer(
            search_space=space,
            population_size=5,
            n_generations=3
        )

        # Simple fitness function
        def dummy_fitness(hp):
            return hp['d_model'] / 100 + np.random.random() * 0.1

        best_hp, history = optimizer.optimize(
            dummy_fitness,
            early_stopping_generations=10
        )

        assert 'd_model' in best_hp
        assert len(history['best_fitness']) == 3


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = TrainingConfig()

        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.batch_size == 64
        assert config.device == "cuda"


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def mock_loader(self, small_ohlcv_data):
        """Create mock data loader."""
        from data_pipeline.loader import TradingDataset
        from torch.utils.data import DataLoader

        dataset = TradingDataset(small_ohlcv_data, sequence_length=60)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    @pytest.fixture
    def simple_model(self, config_dict, mock_loader):
        """Create simple model for testing."""
        from models.meta_agent import MetaAgent

        # Get actual input dimensions from the dataset
        batch = next(iter(mock_loader))
        input_dim = batch['features'].shape[-1]

        cfg = config_dict['model']
        return MetaAgent(
            input_dim=input_dim,
            embedding_dim=cfg['transformer']['d_model'],
            n_heads=4,
            transformer_layers=cfg['transformer']['n_layers'],
            transformer_heads=cfg['transformer']['n_heads'],
            transformer_ff=cfg['transformer']['d_ff']
        )

    def test_initialization(self, simple_model, mock_loader):
        """Test trainer initialization."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_compute_loss(self, simple_model, mock_loader):
        """Test loss computation."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch = next(iter(mock_loader))
        features = batch['features']
        targets = batch['target']
        batch_size = features.size(0)

        outputs = simple_model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        loss, components = trainer._compute_loss(
            outputs, targets, torch.zeros(batch_size, 3)
        )

        assert loss.item() >= 0
        assert 'total' in components

    def test_compute_loss_inaction_penalty(self, simple_model, mock_loader):
        """Test that inaction penalty is computed and penalizes zero positions."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch = next(iter(mock_loader))
        features = batch['features']
        targets = batch['target']
        batch_size = features.size(0)

        outputs = simple_model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        loss, components = trainer._compute_loss(
            outputs, targets, torch.zeros(batch_size, 3)
        )

        # Check that inaction penalty component exists
        assert 'inaction' in components
        assert components['inaction'] >= 0

    def test_compute_loss_hold_penalty(self, simple_model, mock_loader):
        """Test that hold penalty is computed."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch = next(iter(mock_loader))
        features = batch['features']
        targets = batch['target']
        batch_size = features.size(0)

        outputs = simple_model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        loss, components = trainer._compute_loss(
            outputs, targets, torch.zeros(batch_size, 3)
        )

        # Check that hold penalty component exists
        assert 'hold_penalty' in components
        assert 0 <= components['hold_penalty'] <= 1  # It's a probability

    def test_compute_loss_low_confidence_penalty(self, simple_model, mock_loader):
        """Test that low confidence penalty is computed."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch = next(iter(mock_loader))
        features = batch['features']
        targets = batch['target']
        batch_size = features.size(0)

        outputs = simple_model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        loss, components = trainer._compute_loss(
            outputs, targets, torch.zeros(batch_size, 3)
        )

        # Check that low confidence penalty component exists
        assert 'low_conf' in components
        assert components['low_conf'] >= 0

    def test_compute_loss_missed_opportunity_penalty(self, simple_model, mock_loader):
        """Test that missed opportunity penalty is computed."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch = next(iter(mock_loader))
        features = batch['features']
        targets = batch['target']
        batch_size = features.size(0)

        outputs = simple_model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        loss, components = trainer._compute_loss(
            outputs, targets, torch.zeros(batch_size, 3)
        )

        # Check that missed opportunity penalty component exists
        assert 'missed_opp' in components
        assert components['missed_opp'] >= 0

    def test_inaction_penalty_high_for_zero_position(self, simple_model, mock_loader):
        """Test that inaction penalty is high when position size is zero."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch_size = 4

        # Create mock outputs with zero position sizes
        outputs_zero_position = {
            'action_probs': torch.softmax(torch.randn(batch_size, 4), dim=-1),
            'position_size': torch.zeros(batch_size),  # Zero positions
            'confidence': torch.ones(batch_size) * 0.7,
            'profit_output': {
                'value': torch.zeros(batch_size),
                'entropy': torch.ones(batch_size)
            }
        }

        # Create mock outputs with non-zero position sizes
        outputs_active_position = {
            'action_probs': torch.softmax(torch.randn(batch_size, 4), dim=-1),
            'position_size': torch.ones(batch_size) * 0.8,  # Active positions
            'confidence': torch.ones(batch_size) * 0.7,
            'profit_output': {
                'value': torch.zeros(batch_size),
                'entropy': torch.ones(batch_size)
            }
        }

        targets = torch.randn(batch_size) * 0.01

        _, components_zero = trainer._compute_loss(
            outputs_zero_position, targets, torch.zeros(batch_size, 3)
        )
        _, components_active = trainer._compute_loss(
            outputs_active_position, targets, torch.zeros(batch_size, 3)
        )

        # Inaction penalty should be higher for zero positions
        assert components_zero['inaction'] > components_active['inaction']

    def test_missed_opportunity_penalty_high_when_not_trading_on_movement(self, simple_model, mock_loader):
        """Test that missed opportunity penalty is high when not trading during price movement."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch_size = 4

        # Large price movement targets
        large_targets = torch.ones(batch_size) * 0.05  # 5% movement

        # Small price movement targets
        small_targets = torch.ones(batch_size) * 0.001  # 0.1% movement

        # Zero position outputs
        outputs = {
            'action_probs': torch.softmax(torch.randn(batch_size, 4), dim=-1),
            'position_size': torch.zeros(batch_size),  # Not trading
            'confidence': torch.ones(batch_size) * 0.7,
            'profit_output': {
                'value': torch.zeros(batch_size),
                'entropy': torch.ones(batch_size)
            }
        }

        _, components_large_move = trainer._compute_loss(
            outputs, large_targets, torch.zeros(batch_size, 3)
        )
        _, components_small_move = trainer._compute_loss(
            outputs, small_targets, torch.zeros(batch_size, 3)
        )

        # Missing a large move should have higher penalty than missing a small move
        assert components_large_move['missed_opp'] > components_small_move['missed_opp']

    def test_all_loss_components_present(self, simple_model, mock_loader):
        """Test that all expected loss components are present in output."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        batch = next(iter(mock_loader))
        features = batch['features']
        targets = batch['target']
        batch_size = features.size(0)

        outputs = simple_model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        loss, components = trainer._compute_loss(
            outputs, targets, torch.zeros(batch_size, 3)
        )

        expected_components = [
            'direction', 'return', 'value', 'sharpe', 'entropy',
            'inaction', 'hold_penalty', 'low_conf', 'missed_opp', 'total'
        ]

        for component in expected_components:
            assert component in components, f"Missing loss component: {component}"

    @pytest.mark.slow
    def test_train_epoch(self, simple_model, mock_loader):
        """Test single training epoch."""
        trainer = Trainer(
            model=simple_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            device='cpu'
        )

        metrics = trainer.train_epoch()

        assert 'loss' in metrics
        assert metrics['loss'] >= 0

    def test_save_load_checkpoint(self, simple_model, mock_loader):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(checkpoint_dir=tmpdir, device='cpu')
            trainer = Trainer(
                model=simple_model,
                train_loader=mock_loader,
                val_loader=mock_loader,
                config=config
            )

            # Save
            trainer.save_checkpoint('test_checkpoint.pt')

            # Load
            trainer2 = Trainer(
                model=simple_model,
                train_loader=mock_loader,
                val_loader=mock_loader,
                config=config
            )
            trainer2.load_checkpoint(str(Path(tmpdir) / 'test_checkpoint.pt'))

            assert trainer2.current_epoch == trainer.current_epoch
