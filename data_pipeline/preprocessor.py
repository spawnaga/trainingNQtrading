"""
OHLCV Data Preprocessor with standardization and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
import ta


class OHLCVPreprocessor:
    """
    Preprocessor for OHLCV data with standardization and technical features.

    Features:
    - Z-score normalization with rolling window
    - Log returns for price stability
    - Volume log-transformation
    - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
    - Price ratios and spreads
    """

    def __init__(
        self,
        lookback_period: int = 252,  # For rolling statistics
        use_log_returns: bool = True,
        use_robust_scaler: bool = False,
        add_technical_features: bool = True,
        atr_period: int = 14,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        self.lookback_period = lookback_period
        self.use_log_returns = use_log_returns
        self.use_robust_scaler = use_robust_scaler
        self.add_technical_features = add_technical_features

        # Technical indicator parameters
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std

        # Scalers for each feature type
        self.scalers: Dict[str, StandardScaler] = {}
        self.fitted = False

        # Feature statistics for online normalization
        self.rolling_means: Dict[str, pd.Series] = {}
        self.rolling_stds: Dict[str, pd.Series] = {}

    def _calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log or simple returns."""
        if self.use_log_returns:
            return np.log(prices / prices.shift(1))
        return prices.pct_change()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        result = df.copy()

        # ATR - Average True Range (for volatility and stops)
        result['atr'] = ta.volatility.average_true_range(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.atr_period
        )

        # ATR percentage (normalized by price)
        result['atr_pct'] = result['atr'] / result['close']

        # RSI - Relative Strength Index
        result['rsi'] = ta.momentum.rsi(
            close=df['close'],
            window=self.rsi_period
        )
        # Normalize RSI to [-1, 1] range
        result['rsi_norm'] = (result['rsi'] - 50) / 50

        # MACD
        macd = ta.trend.MACD(
            close=df['close'],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_hist'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        result['bb_upper'] = bb.bollinger_hband()
        result['bb_lower'] = bb.bollinger_lband()
        result['bb_mid'] = bb.bollinger_mavg()
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_mid']
        result['bb_position'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])

        # Moving averages
        result['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        result['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        result['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        result['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

        # MA crossover signals
        result['ma_cross_20_50'] = (result['sma_20'] - result['sma_50']) / result['close']
        result['ema_cross'] = (result['ema_12'] - result['ema_26']) / result['close']

        # Volume indicators
        result['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        result['volume_ratio'] = df['volume'] / result['volume_sma']

        # Price momentum
        result['momentum_10'] = ta.momentum.roc(df['close'], window=10)
        result['momentum_20'] = ta.momentum.roc(df['close'], window=20)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        result['stoch_k'] = stoch.stoch() / 100  # Normalize to [0, 1]
        result['stoch_d'] = stoch.stoch_signal() / 100

        return result

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features."""
        result = df.copy()

        # Returns
        result['returns'] = self._calculate_returns(df['close'])
        result['returns_high'] = self._calculate_returns(df['high'])
        result['returns_low'] = self._calculate_returns(df['low'])

        # Intrabar features
        result['hl_range'] = (df['high'] - df['low']) / df['close']
        result['oc_range'] = (df['close'] - df['open']) / df['close']
        result['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        result['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        # Body position within range
        result['body_position'] = np.where(
            df['high'] != df['low'],
            (df['close'] - df['low']) / (df['high'] - df['low']),
            0.5
        )

        # Gap
        result['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Volume features
        result['volume_log'] = np.log1p(df['volume'])

        return result

    def _rolling_normalize(
        self,
        series: pd.Series,
        window: int
    ) -> pd.Series:
        """Apply rolling z-score normalization."""
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero

        return (series - rolling_mean) / rolling_std

    def fit(self, df: pd.DataFrame) -> 'OHLCVPreprocessor':
        """
        Fit preprocessor on training data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        # Calculate all features
        processed = self._calculate_price_features(df)

        if self.add_technical_features:
            processed = self._add_technical_indicators(processed)

        # Fit scalers for each column
        ScalerClass = RobustScaler if self.use_robust_scaler else StandardScaler

        for col in processed.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                clean_data = processed[col].dropna().values.reshape(-1, 1)
                if len(clean_data) > 0:
                    self.scalers[col] = ScalerClass()
                    self.scalers[col].fit(clean_data)

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        use_rolling_norm: bool = True
    ) -> pd.DataFrame:
        """
        Transform OHLCV data to normalized features.

        Args:
            df: DataFrame with OHLCV columns
            use_rolling_norm: Use rolling window normalization (default True)

        Returns:
            DataFrame with normalized features
        """
        # Calculate features
        result = self._calculate_price_features(df)

        if self.add_technical_features:
            result = self._add_technical_indicators(result)

        # Normalize features
        normalized = pd.DataFrame(index=result.index)

        for col in result.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue  # Skip raw OHLCV

            if use_rolling_norm:
                normalized[col] = self._rolling_normalize(
                    result[col],
                    self.lookback_period
                )
            elif col in self.scalers:
                values = result[col].values.reshape(-1, 1)
                # Handle NaN values
                mask = ~np.isnan(values.flatten())
                normalized_values = np.full_like(values.flatten(), np.nan)
                if mask.any():
                    normalized_values[mask] = self.scalers[col].transform(
                        values[mask].reshape(-1, 1)
                    ).flatten()
                normalized[col] = normalized_values
            else:
                normalized[col] = result[col]

        # Clip extreme values
        normalized = normalized.clip(-5, 5)

        return normalized

    def fit_transform(
        self,
        df: pd.DataFrame,
        use_rolling_norm: bool = True
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df, use_rolling_norm)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names after preprocessing."""
        base_features = [
            'returns', 'returns_high', 'returns_low',
            'hl_range', 'oc_range', 'upper_shadow', 'lower_shadow',
            'body_position', 'gap', 'volume_log'
        ]

        if self.add_technical_features:
            base_features.extend([
                'atr', 'atr_pct',
                'rsi', 'rsi_norm',
                'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_position',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'ma_cross_20_50', 'ema_cross',
                'volume_sma', 'volume_ratio',
                'momentum_10', 'momentum_20',
                'stoch_k', 'stoch_d'
            ])

        return base_features

    @property
    def num_features(self) -> int:
        """Return number of features after preprocessing."""
        return len(self.get_feature_names())


class DataNormalizer:
    """
    Simple normalizer for real-time inference.
    Uses exponential moving statistics for online normalization.
    """

    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha: Smoothing factor for exponential moving average
        """
        self.alpha = alpha
        self.ema_mean: Optional[np.ndarray] = None
        self.ema_var: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> np.ndarray:
        """
        Update statistics and return normalized values.

        Args:
            x: Input array of shape (n_features,)

        Returns:
            Normalized array
        """
        if self.ema_mean is None:
            self.ema_mean = x.copy()
            self.ema_var = np.ones_like(x)
            return np.zeros_like(x)

        # Update statistics
        delta = x - self.ema_mean
        self.ema_mean = self.ema_mean + self.alpha * delta
        self.ema_var = (1 - self.alpha) * (self.ema_var + self.alpha * delta ** 2)

        # Normalize
        std = np.sqrt(self.ema_var)
        std = np.where(std < 1e-8, 1.0, std)

        return (x - self.ema_mean) / std
