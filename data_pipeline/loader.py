"""
Data Loader for CSV files with OHLCV data.
"""

import os
import glob
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from .preprocessor import OHLCVPreprocessor
from .time_encoder import CyclicalTimeEncoder


class CSVDataLoader:
    """
    Load and merge multiple CSV files containing OHLCV data.

    Expected CSV format:
    - datetime/date/timestamp column
    - open, high, low, close, volume columns
    """

    DATETIME_COLUMNS = ['datetime', 'date', 'timestamp', 'time', 'Date', 'DateTime', 'Timestamp']
    OHLCV_COLUMNS = {
        'open': ['open', 'Open', 'OPEN', 'o'],
        'high': ['high', 'High', 'HIGH', 'h'],
        'low': ['low', 'Low', 'LOW', 'l'],
        'close': ['close', 'Close', 'CLOSE', 'c'],
        'volume': ['volume', 'Volume', 'VOLUME', 'v', 'vol', 'Vol']
    }

    def __init__(
        self,
        csv_folder: str,
        date_format: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Args:
            csv_folder: Path to folder containing CSV files
            date_format: Optional datetime format string
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        """
        self.csv_folder = Path(csv_folder)
        self.date_format = date_format
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the datetime column in a DataFrame."""
        for col in self.DATETIME_COLUMNS:
            if col in df.columns:
                return col

        # Try to find any column with 'date' or 'time' in the name
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return col

        return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase OHLCV format."""
        result = df.copy()
        column_mapping = {}

        for standard_name, variations in self.OHLCV_COLUMNS.items():
            for var in variations:
                if var in result.columns:
                    column_mapping[var] = standard_name
                    break

        result = result.rename(columns=column_mapping)
        return result

    def _parse_datetime(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Parse and set datetime index."""
        result = df.copy()

        if self.date_format:
            result[datetime_col] = pd.to_datetime(result[datetime_col], format=self.date_format)
        else:
            result[datetime_col] = pd.to_datetime(result[datetime_col])

        result = result.set_index(datetime_col)
        result.index.name = 'datetime'

        return result

    def _detect_headerless(self, filepath: Union[str, Path]) -> bool:
        """Detect if CSV file has no header row."""
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        # Check if first value looks like a datetime (headerless)
        first_val = first_line.split(',')[0]
        try:
            pd.to_datetime(first_val)
            return True  # First value is datetime, so no header
        except (ValueError, TypeError):
            return False  # First value is not datetime, likely a header

    def load_single_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load a single CSV file."""
        # Detect if file has headers
        headerless = self._detect_headerless(filepath)

        if headerless:
            # No header - assume format: datetime, open, high, low, close, volume
            df = pd.read_csv(
                filepath,
                header=None,
                names=['datetime', 'open', 'high', 'low', 'close', 'volume']
            )
            datetime_col = 'datetime'
        else:
            df = pd.read_csv(filepath)
            # Find and parse datetime
            datetime_col = self._find_datetime_column(df)
            if datetime_col is None:
                raise ValueError(f"Could not find datetime column in {filepath}")
            # Standardize columns
            df = self._standardize_columns(df)

        # Parse datetime and set as index
        df = self._parse_datetime(df, datetime_col)

        # Ensure OHLCV columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {filepath}")

        # Select only OHLCV columns
        df = df[required]

        return df

    def load_all_files(self) -> pd.DataFrame:
        """Load and merge all CSV files from folder."""
        csv_files = list(self.csv_folder.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_folder}")

        print(f"Found {len(csv_files)} CSV files")

        dfs = []
        for filepath in sorted(csv_files):
            try:
                df = self.load_single_file(filepath)
                dfs.append(df)
                print(f"  Loaded: {filepath.name} ({len(df)} rows)")
            except Exception as e:
                print(f"  Error loading {filepath.name}: {e}")

        if not dfs:
            raise ValueError("No valid data files loaded")

        # Concatenate and sort
        result = pd.concat(dfs)
        result = result.sort_index()

        # Remove duplicates
        result = result[~result.index.duplicated(keep='first')]

        # Apply date filters
        if self.start_date:
            result = result[result.index >= self.start_date]
        if self.end_date:
            result = result[result.index <= self.end_date]

        print(f"Total: {len(result)} rows from {result.index.min()} to {result.index.max()}")

        return result


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading data with sequence windows.
    """

    def __init__(
        self,
        ohlcv_data: pd.DataFrame,
        sequence_length: int = 60,
        preprocessor: Optional[OHLCVPreprocessor] = None,
        time_encoder: Optional[CyclicalTimeEncoder] = None,
        target_horizon: int = 1,
        include_target: bool = True
    ):
        """
        Args:
            ohlcv_data: DataFrame with OHLCV data (datetime index)
            sequence_length: Number of bars in each sequence
            preprocessor: OHLCVPreprocessor instance
            time_encoder: CyclicalTimeEncoder instance
            target_horizon: Number of bars ahead for target
            include_target: Whether to compute target returns
        """
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.include_target = include_target

        # Initialize preprocessor and time encoder
        self.preprocessor = preprocessor or OHLCVPreprocessor()
        self.time_encoder = time_encoder or CyclicalTimeEncoder()

        # Process data
        self.timestamps = ohlcv_data.index
        self.raw_data = ohlcv_data.copy()

        # Fit and transform features
        self.price_features = self.preprocessor.fit_transform(ohlcv_data)
        self.time_features = self.time_encoder.encode(self.timestamps)

        # Combine features
        self.features = pd.concat([self.price_features, self.time_features], axis=1)
        self.features = self.features.fillna(0)  # Handle NaN
        # Ensure all columns are numeric
        self.features = self.features.astype(np.float32)

        # Calculate targets (future returns)
        if include_target:
            self.targets = ohlcv_data['close'].pct_change(target_horizon).shift(-target_horizon)
        else:
            self.targets = None

        # Store close prices for position sizing calculations
        self.close_prices = ohlcv_data['close'].values

        # Calculate valid indices (where we have enough history and future)
        self.valid_start = sequence_length
        self.valid_end = len(ohlcv_data) - target_horizon if include_target else len(ohlcv_data)

        # Feature dimensions
        self.num_price_features = self.price_features.shape[1]
        self.num_time_features = self.time_features.shape[1]
        self.num_features = self.features.shape[1]

    def __len__(self) -> int:
        return max(0, self.valid_end - self.valid_start)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence window.

        Returns:
            Dict containing:
            - features: (seq_len, num_features) tensor
            - target: scalar target return (if include_target)
            - close_price: current close price
            - timestamp_idx: index in original data
        """
        actual_idx = self.valid_start + idx

        # Get sequence window
        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx

        features = self.features.iloc[start_idx:end_idx].values
        features_tensor = torch.tensor(features, dtype=torch.float32)

        result = {
            'features': features_tensor,
            'close_price': torch.tensor(self.close_prices[actual_idx - 1], dtype=torch.float32),
            'timestamp_idx': torch.tensor(actual_idx - 1, dtype=torch.long)
        }

        if self.include_target and self.targets is not None:
            target = self.targets.iloc[actual_idx - 1]
            result['target'] = torch.tensor(target if not pd.isna(target) else 0.0, dtype=torch.float32)

        return result

    def get_feature_info(self) -> Dict[str, int]:
        """Return information about features."""
        return {
            'num_price_features': self.num_price_features,
            'num_time_features': self.num_time_features,
            'total_features': self.num_features,
            'sequence_length': self.sequence_length
        }


def create_data_loaders(
    csv_folder: str,
    sequence_length: int = 60,
    batch_size: int = 64,
    train_split: float = 0.7,
    val_split: float = 0.15,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_workers: int = 0
) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, Dict]:
    """
    Create train, validation, and test data loaders.

    Args:
        csv_folder: Path to CSV files
        sequence_length: Sequence length for model
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation
        start_date: Optional start date filter
        end_date: Optional end date filter
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader, test_loader, feature_info)
    """
    # Load data
    loader = CSVDataLoader(csv_folder, start_date=start_date, end_date=end_date)
    data = loader.load_all_files()

    # Calculate split indices
    n = len(data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    # Split data (time-ordered, no shuffle)
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create shared preprocessor and time encoder (fit on train only)
    preprocessor = OHLCVPreprocessor()
    preprocessor.fit(train_data)
    time_encoder = CyclicalTimeEncoder()

    # Create datasets
    train_dataset = TradingDataset(
        train_data,
        sequence_length=sequence_length,
        preprocessor=preprocessor,
        time_encoder=time_encoder
    )

    val_dataset = TradingDataset(
        val_data,
        sequence_length=sequence_length,
        preprocessor=preprocessor,
        time_encoder=time_encoder
    )

    test_dataset = TradingDataset(
        test_data,
        sequence_length=sequence_length,
        preprocessor=preprocessor,
        time_encoder=time_encoder
    )

    # Create data loaders
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    feature_info = train_dataset.get_feature_info()

    return train_loader, val_loader, test_loader, feature_info
