"""
Live Trading Script for NQ Futures
===================================
Uses ib_insync to connect to Interactive Brokers and execute trades
based on the trained SimpleMetaAgent model.

Requirements:
    pip install ib_insync numpy torch pandas ta

Usage:
    1. Start IB Gateway or TWS with API enabled (port 7497 for paper, 7496 for live)
    2. Run: python live_trading.py --paper  (for paper trading)
    3. Run: python live_trading.py --live   (for live trading - BE CAREFUL!)

Author: Alex Oraibi
"""

import asyncio
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import yaml
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque
from collections import deque
import threading

# IB imports
from ib_insync import IB, Future, MarketOrder, LimitOrder, util

sys.path.insert(0, str(Path(__file__).parent))

from models.simple_meta_agent import SimpleMetaAgent
from data_pipeline.preprocessor import OHLCVPreprocessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Live trading configuration."""
    # Connection
    host: str = '127.0.0.1'
    port_paper: int = 7497
    port_live: int = 7496
    client_id: int = 1

    # Contract
    symbol: str = 'NQ'
    exchange: str = 'CME'
    currency: str = 'USD'

    # Model
    model_path: str = 'checkpoints/best_model_fixed.pt'
    sequence_length: int = 60
    embedding_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff: int = 512
    dropout: float = 0.1

    # Trading parameters
    max_position: int = 1  # Maximum contracts
    min_signal_threshold: float = 0.1  # Minimum signal to trade
    confidence_threshold: float = 0.5  # Minimum confidence
    update_interval: int = 60  # Seconds between updates

    # Risk management
    max_daily_loss: float = 2000.0  # Maximum daily loss in USD
    stop_loss_ticks: int = 20  # Stop loss in ticks (0.25 per tick for NQ)
    take_profit_ticks: int = 40  # Take profit in ticks

    # Features
    feature_cols: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'log_returns', 'high_low_range', 'close_open_range',
        'volume_ma_ratio', 'price_ma_5', 'price_ma_10', 'price_ma_20',
        'volatility_5', 'volatility_10', 'volatility_20',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr_14', 'obv', 'vwap'
    ])


class DataBuffer:
    """Rolling buffer for storing OHLCV data."""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.data: Deque[Dict] = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, bar: Dict):
        """Add a new bar to the buffer."""
        with self.lock:
            self.data.append(bar)

    def get_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame."""
        with self.lock:
            if len(self.data) == 0:
                return pd.DataFrame()
            return pd.DataFrame(list(self.data))

    def __len__(self):
        return len(self.data)


class FeatureGenerator:
    """Generate features from OHLCV data."""

    def __init__(self):
        self.preprocessor = OHLCVPreprocessor()

    def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute features from raw OHLCV data."""
        if len(df) < 60:
            return None

        # Ensure proper column names
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['open']

        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ma_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)

        # Moving averages
        for window in [5, 10, 20]:
            df[f'price_ma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_ma_{window}'] = df['close'] / df[f'price_ma_{window}'] - 1

        # Volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = (df['rsi_14'] - 50) / 50  # Normalize to [-1, 1]

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        # Normalize
        macd_std = df['macd'].rolling(20).std()
        df['macd'] = df['macd'] / (macd_std + 1e-8)
        df['macd_signal'] = df['macd_signal'] / (macd_std + 1e-8)
        df['macd_hist'] = df['macd_hist'] / (macd_std + 1e-8)

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        # Normalize price position within bands
        df['bb_upper'] = (df['close'] - df['bb_upper']) / (bb_std + 1e-8)
        df['bb_middle'] = (df['close'] - df['bb_middle']) / (bb_std + 1e-8)
        df['bb_lower'] = (df['close'] - df['bb_lower']) / (bb_std + 1e-8)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean() / df['close']

        # OBV (normalized)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv'] = (df['obv'] - df['obv'].rolling(20).mean()) / (df['obv'].rolling(20).std() + 1e-8)

        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap'] = df['close'] / df['vwap'] - 1

        # Fill NaN
        df = df.fillna(0)

        # Select feature columns
        feature_cols = [
            'returns', 'log_returns', 'high_low_range', 'close_open_range',
            'volume_ma_ratio', 'price_ma_5', 'price_ma_10', 'price_ma_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr_14', 'obv', 'vwap'
        ]

        # Pad to 48 features if needed
        available_cols = [c for c in feature_cols if c in df.columns]
        features = df[available_cols].values

        # Pad with zeros if needed
        if features.shape[1] < 48:
            padding = np.zeros((features.shape[0], 48 - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)

        return features


class LiveTrader:
    """Live trading system using Interactive Brokers."""

    def __init__(self, config: TradingConfig, paper: bool = True):
        self.config = config
        self.paper = paper
        self.ib = IB()

        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data
        self.data_buffer = DataBuffer(max_size=500)
        self.feature_generator = FeatureGenerator()

        # State
        self.current_position = 0
        self.daily_pnl = 0.0
        self.last_signal = 0.0
        self.last_confidence = 0.0
        self.is_running = False
        self.contract = None

        # Order tracking
        self.pending_orders = []
        self.filled_orders = []

    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.config.model_path}")

        # Determine input dimension (we'll use 48 to match training)
        input_dim = 48

        self.model = SimpleMetaAgent(
            input_dim=input_dim,
            embedding_dim=self.config.embedding_dim,
            transformer_layers=self.config.transformer_layers,
            transformer_heads=self.config.transformer_heads,
            transformer_ff=self.config.transformer_ff,
            dropout=self.config.dropout
        ).to(self.device)

        checkpoint = torch.load(
            self.config.model_path,
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded (epoch {checkpoint['epoch']}, Sharpe {checkpoint['sharpe']:.2f})")

    def connect(self):
        """Connect to Interactive Brokers."""
        port = self.config.port_paper if self.paper else self.config.port_live
        mode = "PAPER" if self.paper else "LIVE"

        logger.info(f"Connecting to IB Gateway ({mode}) at {self.config.host}:{port}")

        self.ib.connect(
            self.config.host,
            port,
            clientId=self.config.client_id
        )

        if self.ib.isConnected():
            logger.info("Connected to Interactive Brokers")
        else:
            raise ConnectionError("Failed to connect to IB")

        # Create NQ futures contract
        self.contract = Future(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            currency=self.config.currency
        )

        # Qualify the contract
        contracts = self.ib.qualifyContracts(self.contract)
        if contracts:
            self.contract = contracts[0]
            logger.info(f"Contract qualified: {self.contract}")
        else:
            raise ValueError(f"Could not qualify contract for {self.config.symbol}")

    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from Interactive Brokers")

    def get_current_position(self) -> int:
        """Get current position in the contract."""
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == self.config.symbol:
                return int(pos.position)
        return 0

    def subscribe_to_data(self):
        """Subscribe to real-time market data."""
        logger.info("Subscribing to market data...")

        # Request real-time bars (1-minute)
        self.ib.reqRealTimeBars(
            self.contract,
            barSize=5,  # 5-second bars (smallest available)
            whatToShow='TRADES',
            useRTH=False  # Include extended hours
        )

        # Set up callback for real-time bars
        self.ib.pendingTickersEvent += self.on_pending_tickers

    def on_pending_tickers(self, tickers):
        """Callback for real-time data updates."""
        for ticker in tickers:
            if ticker.contract.symbol == self.config.symbol:
                bar = {
                    'time': datetime.now(),
                    'open': ticker.open,
                    'high': ticker.high,
                    'low': ticker.low,
                    'close': ticker.close if ticker.close else ticker.last,
                    'volume': ticker.volume if ticker.volume else 0
                }
                self.data_buffer.add(bar)

    def get_historical_data(self, duration: str = '1 D', bar_size: str = '1 min'):
        """Get historical data to initialize the buffer."""
        logger.info(f"Fetching historical data ({duration}, {bar_size})...")

        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )

        for bar in bars:
            self.data_buffer.add({
                'time': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })

        logger.info(f"Loaded {len(bars)} historical bars")

    def predict(self) -> tuple:
        """Generate trading signal from model."""
        df = self.data_buffer.get_dataframe()

        if len(df) < self.config.sequence_length:
            logger.warning(f"Insufficient data: {len(df)} bars (need {self.config.sequence_length})")
            return 0.0, 0.0

        # Compute features
        features = self.feature_generator.compute_features(df)
        if features is None:
            return 0.0, 0.0

        # Get last sequence_length bars
        features = features[-self.config.sequence_length:]

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(features_tensor)
            signal = outputs['position_size'].item()
            confidence = outputs['confidence'].item()

        return signal, confidence

    def execute_trade(self, target_position: int):
        """Execute trade to reach target position."""
        current_pos = self.get_current_position()
        delta = target_position - current_pos

        if delta == 0:
            return

        action = 'BUY' if delta > 0 else 'SELL'
        quantity = abs(delta)

        logger.info(f"Executing {action} {quantity} contracts (current: {current_pos}, target: {target_position})")

        # Create market order
        order = MarketOrder(action, quantity)

        # Submit order
        trade = self.ib.placeOrder(self.contract, order)

        # Wait for fill
        while not trade.isDone():
            self.ib.sleep(0.1)

        if trade.orderStatus.status == 'Filled':
            fill_price = trade.orderStatus.avgFillPrice
            logger.info(f"Order filled at {fill_price}")
            self.current_position = target_position
        else:
            logger.error(f"Order failed: {trade.orderStatus.status}")

    def trading_loop(self):
        """Main trading loop."""
        logger.info("Starting trading loop...")
        self.is_running = True

        while self.is_running:
            try:
                # Check daily loss limit
                if self.daily_pnl < -self.config.max_daily_loss:
                    logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                    if self.current_position != 0:
                        self.execute_trade(0)  # Flatten
                    self.ib.sleep(60)
                    continue

                # Generate signal
                signal, confidence = self.predict()
                self.last_signal = signal
                self.last_confidence = confidence

                logger.info(f"Signal: {signal:.4f}, Confidence: {confidence:.4f}")

                # Determine target position
                if abs(signal) < self.config.min_signal_threshold:
                    target_position = 0
                elif confidence < self.config.confidence_threshold:
                    target_position = self.current_position  # Hold
                else:
                    # Scale position by signal strength
                    target_position = int(np.sign(signal) * min(
                        abs(signal) * self.config.max_position,
                        self.config.max_position
                    ))

                # Execute if needed
                if target_position != self.current_position:
                    self.execute_trade(target_position)

                # Sleep until next update
                self.ib.sleep(self.config.update_interval)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.is_running = False
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.ib.sleep(10)

        # Flatten position on exit
        if self.current_position != 0:
            logger.info("Flattening position on exit...")
            self.execute_trade(0)

    def run(self):
        """Run the live trading system."""
        try:
            self.load_model()
            self.connect()
            self.get_historical_data()
            self.subscribe_to_data()

            # Wait for data to populate
            logger.info("Waiting for data buffer to fill...")
            while len(self.data_buffer) < self.config.sequence_length:
                self.ib.sleep(1)

            logger.info(f"Data buffer ready with {len(self.data_buffer)} bars")

            # Start trading
            self.trading_loop()

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(description='NQ Futures Live Trading')
    parser.add_argument('--paper', action='store_true', help='Use paper trading (port 7497)')
    parser.add_argument('--live', action='store_true', help='Use live trading (port 7496)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IB Gateway host')
    parser.add_argument('--client-id', type=int, default=1, help='Client ID')
    parser.add_argument('--model', type=str, default='checkpoints/best_model_fixed.pt', help='Model path')
    parser.add_argument('--max-position', type=int, default=1, help='Maximum position size')
    parser.add_argument('--update-interval', type=int, default=60, help='Update interval in seconds')

    args = parser.parse_args()

    if not args.paper and not args.live:
        print("ERROR: Must specify --paper or --live")
        print("  --paper : Connect to paper trading (port 7497)")
        print("  --live  : Connect to live trading (port 7496) - USE WITH CAUTION!")
        sys.exit(1)

    if args.live:
        print("=" * 60)
        print("WARNING: LIVE TRADING MODE")
        print("This will execute REAL trades with REAL money!")
        print("=" * 60)
        confirm = input("Type 'I UNDERSTAND' to continue: ")
        if confirm != 'I UNDERSTAND':
            print("Aborted.")
            sys.exit(1)

    config = TradingConfig(
        host=args.host,
        client_id=args.client_id,
        model_path=args.model,
        max_position=args.max_position,
        update_interval=args.update_interval
    )

    trader = LiveTrader(config, paper=args.paper)
    trader.run()


if __name__ == '__main__':
    main()
