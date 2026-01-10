"""
Pytest configuration and shared fixtures for NQ Trading System tests.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Generator
import tempfile
import asyncio

import pytest
import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create event loop for ib_insync compatibility
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root) -> Path:
    """Return data directory path."""
    return project_root / "data" / "csv"


@pytest.fixture(scope="session")
def config_dict() -> dict:
    """Return default configuration dictionary."""
    return {
        'data': {
            'csv_folder': 'data/csv',
            'sequence_length': 60,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        },
        'model': {
            'transformer': {
                'd_model': 64,  # Smaller for tests
                'n_heads': 4,
                'n_layers': 2,
                'd_ff': 128,
                'dropout': 0.1
            },
            'profit_agent': {'hidden_dim': 32},
            'risk_agent': {'hidden_dim': 32},
            'duration_agent': {'hidden_dim': 32}
        },
        'training': {
            'batch_size': 16,
            'epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'gradient_clip': 1.0,
            'device': 'cpu'
        },
        'trading': {
            'max_position': 2,
            'risk_per_trade': 0.02,
            'atr_multiplier': 2.0
        },
        'backtest': {
            'initial_capital': 100000.0,
            'commission': 2.25,
            'slippage': 0.25
        }
    }


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 1000

    # Generate realistic price series
    base_price = 15000.0
    returns = np.random.normal(0.0001, 0.002, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    dates = pd.date_range(
        start='2023-01-01 09:30:00',
        periods=n_bars,
        freq='1min'
    )

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': prices * (1 + np.random.uniform(0, 0.003, n_bars)),
        'low': prices * (1 - np.random.uniform(0, 0.003, n_bars)),
        'close': prices,
        'volume': np.random.randint(100, 10000, n_bars)
    }, index=dates)

    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    data.index.name = 'datetime'
    return data


@pytest.fixture(scope="session")
def small_ohlcv_data() -> pd.DataFrame:
    """Generate smaller OHLCV data for quick tests."""
    np.random.seed(42)
    n_bars = 200

    base_price = 15000.0
    returns = np.random.normal(0.0001, 0.002, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    dates = pd.date_range(
        start='2023-01-01 09:30:00',
        periods=n_bars,
        freq='1min'
    )

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': prices * (1 + np.random.uniform(0, 0.003, n_bars)),
        'low': prices * (1 - np.random.uniform(0, 0.003, n_bars)),
        'close': prices,
        'volume': np.random.randint(100, 10000, n_bars)
    }, index=dates)

    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    data.index.name = 'datetime'
    return data


@pytest.fixture
def temp_csv_file(sample_ohlcv_data) -> Generator[Path, None, None]:
    """Create temporary CSV file with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_data.csv"
        sample_ohlcv_data.to_csv(filepath)
        yield filepath


@pytest.fixture
def temp_csv_file_headerless(sample_ohlcv_data) -> Generator[Path, None, None]:
    """Create temporary CSV file without headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_data_no_header.csv"
        sample_ohlcv_data.to_csv(filepath, header=False)
        yield filepath


@pytest.fixture
def temp_csv_folder(sample_ohlcv_data) -> Generator[Path, None, None]:
    """Create temporary folder with CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)

        # Split data into multiple files
        n = len(sample_ohlcv_data)
        chunk_size = n // 3

        for i in range(3):
            start = i * chunk_size
            end = start + chunk_size if i < 2 else n
            chunk = sample_ohlcv_data.iloc[start:end]
            filepath = folder / f"data_part_{i}.csv"
            chunk.to_csv(filepath)

        yield folder


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def device() -> torch.device:
    """Return available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_features(config_dict) -> torch.Tensor:
    """Generate sample feature tensor for model testing."""
    batch_size = 4
    seq_length = config_dict['data']['sequence_length']
    n_features = 44  # Approximate number of features

    return torch.randn(batch_size, seq_length, n_features)


@pytest.fixture
def transformer_agent(config_dict):
    """Create TransformerAgent for testing."""
    from models.transformer_agent import TransformerAgent

    cfg = config_dict['model']['transformer']
    return TransformerAgent(
        input_dim=44,
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        dropout=cfg['dropout']
    )


@pytest.fixture
def profit_agent(config_dict):
    """Create ProfitMaximizer for testing."""
    from models.profit_agent import ProfitMaximizer

    return ProfitMaximizer(
        embedding_dim=config_dict['model']['transformer']['d_model'],
        hidden_dim=config_dict['model']['profit_agent']['hidden_dim']
    )


@pytest.fixture
def risk_agent(config_dict):
    """Create RiskController for testing."""
    from models.risk_agent import RiskController

    return RiskController(
        embedding_dim=config_dict['model']['transformer']['d_model'],
        hidden_dim=config_dict['model']['risk_agent']['hidden_dim']
    )


@pytest.fixture
def duration_agent(config_dict):
    """Create TradeDurationAgent for testing."""
    from models.duration_agent import TradeDurationAgent

    return TradeDurationAgent(
        embedding_dim=config_dict['model']['transformer']['d_model'],
        hidden_dim=config_dict['model']['duration_agent']['hidden_dim']
    )


@pytest.fixture
def meta_agent(config_dict):
    """Create MetaAgent for testing."""
    from models.meta_agent import MetaAgent

    cfg = config_dict['model']
    return MetaAgent(
        input_dim=44,
        embedding_dim=cfg['transformer']['d_model'],
        n_heads=4,
        transformer_layers=cfg['transformer']['n_layers'],
        transformer_heads=cfg['transformer']['n_heads'],
        transformer_ff=cfg['transformer']['d_ff'],
        dropout=cfg['transformer']['dropout'],
        profit_hidden=cfg['profit_agent']['hidden_dim'],
        risk_hidden=cfg['risk_agent']['hidden_dim']
    )


# ============================================================================
# Trading Fixtures
# ============================================================================

@pytest.fixture
def paper_trader(config_dict):
    """Create PaperTrader for testing."""
    from trading.paper_trader import PaperTrader

    return PaperTrader(
        initial_capital=config_dict['backtest']['initial_capital'],
        commission_per_side=config_dict['backtest']['commission'],
        slippage_points=config_dict['backtest']['slippage'],
        max_position=config_dict['trading']['max_position']
    )


@pytest.fixture
def position_manager():
    """Create PositionManager for testing."""
    from trading.position_manager import PositionManager

    return PositionManager(
        symbol="NQ",
        point_value=20.0,
        commission_per_side=2.25
    )


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def sample_equity_curve() -> np.ndarray:
    """Generate sample equity curve for metric testing."""
    np.random.seed(42)
    n_points = 252  # One year of daily data

    # Generate returns with positive drift
    returns = np.random.normal(0.0005, 0.01, n_points)
    equity = 100000 * np.cumprod(1 + returns)

    return equity


@pytest.fixture
def sample_trades() -> list:
    """Generate sample trade list for metric testing."""
    return [
        {'pnl': 500.0, 'holding_time': 30},
        {'pnl': -200.0, 'holding_time': 15},
        {'pnl': 800.0, 'holding_time': 45},
        {'pnl': -100.0, 'holding_time': 10},
        {'pnl': 300.0, 'holding_time': 25},
        {'pnl': -400.0, 'holding_time': 35},
        {'pnl': 600.0, 'holding_time': 20},
        {'pnl': 150.0, 'holding_time': 40},
        {'pnl': -250.0, 'holding_time': 30},
        {'pnl': 450.0, 'holding_time': 55},
    ]


# ============================================================================
# Helper Functions
# ============================================================================

def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_no_nan(tensor: torch.Tensor):
    """Assert tensor contains no NaN values."""
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values"


def assert_no_inf(tensor: torch.Tensor):
    """Assert tensor contains no infinite values."""
    assert not torch.isinf(tensor).any(), "Tensor contains infinite values"


# Make helper functions available to tests
pytest.assert_tensor_shape = assert_tensor_shape
pytest.assert_no_nan = assert_no_nan
pytest.assert_no_inf = assert_no_inf
