"""
Integration tests for the full trading pipeline.

Tests end-to-end workflows combining multiple modules.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_pipeline.loader import CSVDataLoader, TradingDataset, create_data_loaders
from data_pipeline.preprocessor import OHLCVPreprocessor
from data_pipeline.time_encoder import CyclicalTimeEncoder
from models.meta_agent import MetaAgent
from training.trainer import Trainer, TrainingConfig
from training.fitness import FitnessEvaluator, simulate_trades_from_signals
from trading.paper_trader import PaperTrader
from trading.order_manager import OrderSide


class TestDataToModelPipeline:
    """Tests for data loading to model inference pipeline."""

    def test_full_data_pipeline(self, sample_ohlcv_data, config_dict):
        """Test complete data processing pipeline."""
        # Create dataset
        dataset = TradingDataset(
            sample_ohlcv_data,
            sequence_length=config_dict['data']['sequence_length']
        )

        # Get a batch
        batch = dataset[0]

        # Verify shapes
        seq_len = config_dict['data']['sequence_length']
        assert batch['features'].shape[0] == seq_len
        assert batch['features'].shape[1] > 0

        # Create model with matching input dim
        model = MetaAgent(
            input_dim=batch['features'].shape[1],
            embedding_dim=config_dict['model']['transformer']['d_model'],
            n_heads=4,
            transformer_layers=config_dict['model']['transformer']['n_layers'],
            transformer_heads=config_dict['model']['transformer']['n_heads'],
            transformer_ff=config_dict['model']['transformer']['d_ff']
        )

        # Forward pass
        features = batch['features'].unsqueeze(0)
        batch_size = 1

        outputs = model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        assert 'action' in outputs
        assert 'position_size' in outputs

    def test_data_loader_to_training(self, temp_csv_folder, config_dict):
        """Test data loaders with training."""
        train_loader, val_loader, test_loader, feature_info = create_data_loaders(
            str(temp_csv_folder),
            sequence_length=30,
            batch_size=4,
            train_split=0.7,
            val_split=0.15
        )

        # Create model
        model = MetaAgent(
            input_dim=feature_info['total_features'],
            embedding_dim=config_dict['model']['transformer']['d_model'],
            n_heads=4,
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff=128
        )

        # Get a batch and verify training step
        batch = next(iter(train_loader))
        features = batch['features']
        batch_size = features.shape[0]

        model.train()
        outputs = model(
            features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50,
            torch.ones(batch_size, 1) * 15000
        )

        # Backward pass - use multiple outputs for full gradient flow
        loss = (
            outputs['position_size'].mean() +
            outputs['confidence'].mean() +
            outputs['action_probs'].sum()
        )
        loss.backward()

        # Verify at least some gradients exist (not all outputs connect to all params)
        has_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert has_grads > total_params * 0.5  # At least 50% of params should have gradients


class TestModelTradingPipeline:
    """Tests for model inference to trading execution pipeline."""

    def test_model_to_paper_trading(self, sample_ohlcv_data, config_dict):
        """Test model inference driving paper trading."""
        # Setup
        dataset = TradingDataset(
            sample_ohlcv_data,
            sequence_length=60
        )

        model = MetaAgent(
            input_dim=dataset.num_features,
            embedding_dim=config_dict['model']['transformer']['d_model'],
            n_heads=4,
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff=128
        )
        model.eval()

        paper_trader = PaperTrader(
            initial_capital=100000,
            max_position=2
        )

        # Simulate trading for some bars
        n_trades = 0
        for i in range(min(100, len(dataset))):
            batch = dataset[i]
            features = batch['features'].unsqueeze(0)
            price = batch['close_price'].item()

            paper_trader.update_market(price)

            with torch.no_grad():
                outputs = model(
                    features,
                    torch.zeros(1, 3),
                    torch.zeros(1, 4),
                    torch.zeros(1, 5),
                    torch.ones(1, 1) * 50,
                    torch.tensor([[price]])
                )

            action = outputs['action'].item()
            position_size = outputs['position_size'].item()

            # Execute based on action
            current_pos = paper_trader.get_position()

            if action == 1 and abs(position_size) > 0.3 and not current_pos:
                paper_trader.place_bracket_order(
                    OrderSide.BUY, 1,
                    stop_loss=price - 50,
                    take_profit=price + 100
                )
                n_trades += 1
            elif action == 2 and abs(position_size) > 0.3 and not current_pos:
                paper_trader.place_bracket_order(
                    OrderSide.SELL, 1,
                    stop_loss=price + 50,
                    take_profit=price - 100
                )
                n_trades += 1
            elif action == 3 and current_pos:
                paper_trader.close_position()
                n_trades += 1

        # Verify trading occurred
        assert paper_trader.get_equity() > 0

    def test_signal_to_trades_simulation(self, sample_ohlcv_data):
        """Test converting model signals to simulated trades."""
        # Generate random signals
        n_bars = len(sample_ohlcv_data)
        np.random.seed(42)
        signals = np.random.choice([-1, 0, 1], size=n_bars, p=[0.1, 0.8, 0.1])

        prices = sample_ohlcv_data['close'].values
        timestamps = sample_ohlcv_data.index

        trades, equity = simulate_trades_from_signals(
            signals, prices, timestamps
        )

        # Equity curve: 1 initial value + (n_bars - 1) loop iterations = n_bars
        assert len(equity) == n_bars
        assert equity[-1] > 0  # Should still have some capital


class TestTrainingPipeline:
    """Tests for the full training pipeline."""

    @pytest.mark.slow
    def test_short_training_run(self, temp_csv_folder, config_dict):
        """Test a short training run."""
        train_loader, val_loader, _, feature_info = create_data_loaders(
            str(temp_csv_folder),
            sequence_length=30,
            batch_size=4
        )

        model = MetaAgent(
            input_dim=feature_info['total_features'],
            embedding_dim=32,
            n_heads=4,
            transformer_layers=1,
            transformer_heads=2,
            transformer_ff=64
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = TrainingConfig(
                epochs=2,
                learning_rate=0.001,
                batch_size=4,
                checkpoint_dir=tmpdir,
                log_dir=tmpdir,
                device='cpu',
                mixed_precision=False,
                eval_every=1,
                save_every=1
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config
            )

            trainer.train(epochs=2)

            # Verify training occurred
            assert len(trainer.train_losses) == 2
            assert Path(tmpdir, 'final_model.pt').exists()


class TestBacktestPipeline:
    """Tests for backtesting pipeline."""

    def test_backtest_simulation(self, sample_ohlcv_data, config_dict):
        """Test full backtest simulation."""
        from backtest import Backtester

        # Create and initialize model
        preprocessor = OHLCVPreprocessor()
        time_encoder = CyclicalTimeEncoder()

        input_dim = preprocessor.num_features + time_encoder.num_features

        model = MetaAgent(
            input_dim=input_dim,
            embedding_dim=config_dict['model']['transformer']['d_model'],
            n_heads=4,
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff=128
        )

        backtester = Backtester(
            model=model,
            data=sample_ohlcv_data,
            initial_capital=100000,
            sequence_length=60,
            device='cpu'
        )

        # Run partial backtest
        results = backtester.run(
            start_idx=100,
            end_idx=300,
            progress_bar=False
        )

        assert 'metrics' in results
        assert 'equity_curve' in results
        assert len(results['equity_curve']) > 0


class TestRealDataIntegration:
    """Integration tests using real NQ data."""

    @pytest.fixture
    def real_data(self, data_dir):
        """Load sample of real data."""
        nq_file = data_dir / "NQ.csv"
        if not nq_file.exists():
            pytest.skip("NQ.csv not found")

        loader = CSVDataLoader(
            nq_file.parent,
            start_date='2020-01-02',
            end_date='2020-01-10'  # One week of data
        )
        return loader.load_all_files()

    @pytest.mark.slow
    def test_real_data_processing(self, real_data):
        """Test processing real market data."""
        preprocessor = OHLCVPreprocessor()
        time_encoder = CyclicalTimeEncoder()

        # Process
        price_features = preprocessor.fit_transform(real_data)
        time_features = time_encoder.encode(real_data.index)

        assert len(price_features) == len(real_data)
        assert len(time_features) == len(real_data)

        # Check for reasonable values
        assert price_features.iloc[100:].isna().sum().sum() == 0

    @pytest.mark.slow
    def test_real_data_model_inference(self, real_data, config_dict):
        """Test model inference on real data."""
        dataset = TradingDataset(
            real_data,
            sequence_length=60
        )

        model = MetaAgent(
            input_dim=dataset.num_features,
            embedding_dim=config_dict['model']['transformer']['d_model'],
            n_heads=4,
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff=128
        )
        model.eval()

        # Run inference on multiple samples
        for i in range(0, min(100, len(dataset)), 10):
            batch = dataset[i]
            features = batch['features'].unsqueeze(0)
            price = batch['close_price'].item()

            with torch.no_grad():
                outputs = model(
                    features,
                    torch.zeros(1, 3),
                    torch.zeros(1, 4),
                    torch.zeros(1, 5),
                    torch.ones(1, 1) * 50,
                    torch.tensor([[price]])
                )

            # Verify outputs are reasonable
            assert 0 <= outputs['action'].item() <= 3
            assert -1 <= outputs['position_size'].item() <= 1
            assert outputs['confidence'].item() >= 0


class TestMetricsIntegration:
    """Tests for metrics calculation integration."""

    def test_full_metrics_pipeline(self, sample_ohlcv_data):
        """Test complete metrics calculation pipeline."""
        evaluator = FitnessEvaluator()

        # Generate signals and simulate
        n_bars = len(sample_ohlcv_data)
        np.random.seed(42)
        signals = np.random.choice([-1, 0, 1], size=n_bars, p=[0.15, 0.70, 0.15])

        trades, equity = simulate_trades_from_signals(
            signals,
            sample_ohlcv_data['close'].values,
            sample_ohlcv_data.index
        )

        # Calculate metrics
        if trades:
            metrics = evaluator.evaluate_trades(trades)
            fitness = evaluator.calculate_fitness(metrics)

            assert fitness is not None
            assert isinstance(metrics.sharpe_ratio, float)
            assert isinstance(metrics.win_rate, float)
