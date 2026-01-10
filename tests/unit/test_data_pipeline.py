"""
Unit tests for data_pipeline module.

Tests:
- CSVDataLoader
- OHLCVPreprocessor
- CyclicalTimeEncoder
- TradingDataset
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
from data_pipeline.preprocessor import OHLCVPreprocessor, DataNormalizer
from data_pipeline.time_encoder import CyclicalTimeEncoder, compute_time_distance


class TestCyclicalTimeEncoder:
    """Tests for CyclicalTimeEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = CyclicalTimeEncoder()
        assert encoder.session_start == "18:00"
        assert encoder.session_end == "17:00"

    def test_encode_basic(self, sample_ohlcv_data):
        """Test basic encoding."""
        encoder = CyclicalTimeEncoder()
        result = encoder.encode(sample_ohlcv_data.index)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'day_sin' in result.columns
        assert 'day_cos' in result.columns

    def test_cyclical_properties(self):
        """Test cyclical encoding properties (hour 23 close to hour 0)."""
        encoder = CyclicalTimeEncoder()

        # Create timestamps for hour 23 and hour 0
        timestamps = pd.DatetimeIndex([
            '2023-01-01 23:00:00',
            '2023-01-02 00:00:00',
            '2023-01-02 12:00:00'
        ])

        result = encoder.encode(timestamps)

        # Hour 23 and Hour 0 should be closer together than Hour 23 and Hour 12
        h23_h0_dist = np.sqrt(
            (result.loc[timestamps[0], 'hour_sin'] - result.loc[timestamps[1], 'hour_sin'])**2 +
            (result.loc[timestamps[0], 'hour_cos'] - result.loc[timestamps[1], 'hour_cos'])**2
        )

        h23_h12_dist = np.sqrt(
            (result.loc[timestamps[0], 'hour_sin'] - result.loc[timestamps[2], 'hour_sin'])**2 +
            (result.loc[timestamps[0], 'hour_cos'] - result.loc[timestamps[2], 'hour_cos'])**2
        )

        assert h23_h0_dist < h23_h12_dist, "Hour 23 should be closer to Hour 0 than to Hour 12"

    def test_feature_count(self):
        """Test number of features generated."""
        encoder = CyclicalTimeEncoder(include_session_features=True)
        assert encoder.num_features == 14  # 10 cyclical + 4 session features

        encoder_no_session = CyclicalTimeEncoder(include_session_features=False)
        assert encoder_no_session.num_features == 10

    def test_sunday_monday_distance(self):
        """Test that Sunday midnight and Monday 10am have meaningful distance."""
        encoder = CyclicalTimeEncoder()

        sunday_midnight = pd.Timestamp('2023-01-01 00:00:00')  # Sunday
        monday_10am = pd.Timestamp('2023-01-02 10:00:00')      # Monday

        distance = compute_time_distance(sunday_midnight, monday_10am, encoder)
        assert distance > 0, "Time distance should be positive"
        assert distance < 5, "Distance should be reasonable"

    def test_market_open_detection(self):
        """Test market open detection."""
        encoder = CyclicalTimeEncoder()

        # Saturday should be closed
        saturday = pd.Timestamp('2023-01-07 12:00:00')  # Saturday
        assert encoder._is_market_open(saturday) == False

        # Sunday before 6pm should be closed
        sunday_noon = pd.Timestamp('2023-01-08 12:00:00')  # Sunday noon
        assert encoder._is_market_open(sunday_noon) == False

        # Monday midday should be open
        monday_noon = pd.Timestamp('2023-01-09 12:00:00')  # Monday
        assert encoder._is_market_open(monday_noon) == True


class TestOHLCVPreprocessor:
    """Tests for OHLCVPreprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = OHLCVPreprocessor()
        assert preprocessor.lookback_period == 252
        assert preprocessor.use_log_returns == True
        assert preprocessor.fitted == False

    def test_fit(self, sample_ohlcv_data):
        """Test fitting preprocessor."""
        preprocessor = OHLCVPreprocessor()
        preprocessor.fit(sample_ohlcv_data)

        assert preprocessor.fitted == True
        assert len(preprocessor.scalers) > 0

    def test_transform(self, sample_ohlcv_data):
        """Test data transformation."""
        preprocessor = OHLCVPreprocessor()
        result = preprocessor.fit_transform(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert 'returns' in result.columns
        assert 'atr' in result.columns
        assert 'rsi' in result.columns

    def test_output_range(self, sample_ohlcv_data):
        """Test that output is clipped to reasonable range."""
        preprocessor = OHLCVPreprocessor()
        result = preprocessor.fit_transform(sample_ohlcv_data)

        # Should be clipped to [-5, 5]
        assert result.max().max() <= 5.0
        assert result.min().min() >= -5.0

    def test_no_nan_in_output(self, sample_ohlcv_data):
        """Test that NaN values are handled."""
        preprocessor = OHLCVPreprocessor()
        result = preprocessor.fit_transform(sample_ohlcv_data)

        # Fill NaN should result in no NaN (after initial warmup period)
        assert result.iloc[100:].isna().sum().sum() == 0

    def test_technical_indicators(self, sample_ohlcv_data):
        """Test technical indicator calculation."""
        preprocessor = OHLCVPreprocessor(add_technical_features=True)
        result = preprocessor.fit_transform(sample_ohlcv_data)

        expected_features = ['rsi', 'macd', 'bb_width', 'atr']
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"


class TestDataNormalizer:
    """Tests for online DataNormalizer."""

    def test_update(self):
        """Test online normalization update."""
        normalizer = DataNormalizer(alpha=0.1)

        # First update initializes
        x1 = np.array([1.0, 2.0, 3.0])
        result1 = normalizer.update(x1)
        assert np.allclose(result1, np.zeros(3))

        # Second update should normalize
        x2 = np.array([1.1, 2.1, 3.1])
        result2 = normalizer.update(x2)
        assert result2 is not None
        assert len(result2) == 3

    def test_streaming(self):
        """Test streaming normalization."""
        normalizer = DataNormalizer(alpha=0.05)

        # Simulate streaming data
        np.random.seed(42)
        for _ in range(100):
            x = np.random.randn(5) + 10
            result = normalizer.update(x)

        # After many updates, should produce reasonable normalized values
        final = normalizer.update(np.array([10, 10, 10, 10, 10]))
        assert np.all(np.abs(final) < 3), "Normalized values should be reasonable"


class TestCSVDataLoader:
    """Tests for CSVDataLoader."""

    def test_load_with_header(self, temp_csv_file):
        """Test loading CSV with headers."""
        loader = CSVDataLoader(temp_csv_file.parent)
        df = loader.load_single_file(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_headerless(self, temp_csv_file_headerless):
        """Test loading CSV without headers."""
        loader = CSVDataLoader(temp_csv_file_headerless.parent)
        df = loader.load_single_file(temp_csv_file_headerless)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_load_all_files(self, temp_csv_folder):
        """Test loading multiple CSV files."""
        loader = CSVDataLoader(temp_csv_folder)
        df = loader.load_all_files()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_date_filtering(self, temp_csv_folder):
        """Test date range filtering."""
        loader = CSVDataLoader(
            temp_csv_folder,
            start_date='2023-01-01 10:00:00',
            end_date='2023-01-01 12:00:00'
        )
        df = loader.load_all_files()

        assert df.index.min() >= pd.Timestamp('2023-01-01 10:00:00')
        assert df.index.max() <= pd.Timestamp('2023-01-01 12:00:00')

    def test_headerless_detection(self, temp_csv_file, temp_csv_file_headerless):
        """Test automatic header detection."""
        loader = CSVDataLoader(temp_csv_file.parent)

        # File with header
        assert loader._detect_headerless(temp_csv_file) == False

        # File without header
        loader2 = CSVDataLoader(temp_csv_file_headerless.parent)
        assert loader2._detect_headerless(temp_csv_file_headerless) == True


class TestTradingDataset:
    """Tests for TradingDataset."""

    def test_initialization(self, sample_ohlcv_data):
        """Test dataset initialization."""
        dataset = TradingDataset(sample_ohlcv_data, sequence_length=60)

        assert dataset.sequence_length == 60
        assert dataset.num_features > 0
        assert len(dataset) > 0

    def test_getitem(self, sample_ohlcv_data):
        """Test getting items from dataset."""
        dataset = TradingDataset(sample_ohlcv_data, sequence_length=60)

        item = dataset[0]
        assert 'features' in item
        assert 'target' in item
        assert 'close_price' in item

        assert item['features'].shape == (60, dataset.num_features)
        assert isinstance(item['target'], torch.Tensor)

    def test_feature_info(self, sample_ohlcv_data):
        """Test feature info method."""
        dataset = TradingDataset(sample_ohlcv_data, sequence_length=60)
        info = dataset.get_feature_info()

        assert 'num_price_features' in info
        assert 'num_time_features' in info
        assert 'total_features' in info
        assert info['sequence_length'] == 60

    def test_no_target(self, sample_ohlcv_data):
        """Test dataset without targets."""
        dataset = TradingDataset(
            sample_ohlcv_data,
            sequence_length=60,
            include_target=False
        )

        item = dataset[0]
        assert 'target' not in item


class TestCreateDataLoaders:
    """Tests for create_data_loaders function."""

    def test_create_loaders(self, temp_csv_folder):
        """Test creating train/val/test loaders."""
        train_loader, val_loader, test_loader, feature_info = create_data_loaders(
            str(temp_csv_folder),
            sequence_length=30,
            batch_size=8,
            train_split=0.7,
            val_split=0.15
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert feature_info['sequence_length'] == 30

    def test_batch_shape(self, temp_csv_folder):
        """Test batch shapes from data loaders."""
        train_loader, _, _, feature_info = create_data_loaders(
            str(temp_csv_folder),
            sequence_length=30,
            batch_size=4
        )

        batch = next(iter(train_loader))
        assert batch['features'].shape[0] <= 4  # Batch size
        assert batch['features'].shape[1] == 30  # Sequence length


class TestRealData:
    """Tests using the actual NQ.csv data file."""

    @pytest.fixture
    def real_data_path(self, data_dir):
        """Get path to real NQ data."""
        nq_file = data_dir / "NQ.csv"
        if not nq_file.exists():
            pytest.skip("NQ.csv not found")
        return nq_file

    def test_load_real_data(self, real_data_path):
        """Test loading actual NQ data."""
        loader = CSVDataLoader(real_data_path.parent)
        df = loader.load_single_file(real_data_path)

        assert len(df) > 1000000, "Should have millions of rows"
        assert df.index.min().year == 2008
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_real_data_sample(self, real_data_path):
        """Test loading a sample of real data."""
        loader = CSVDataLoader(
            real_data_path.parent,
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        df = loader.load_all_files()

        assert len(df) > 0
        assert df['close'].mean() > 5000  # NQ should be above 5000 in 2020
