"""
Main entry point for the NQ Multi-Agent Trading System.

This script supports:
- Live trading with IB Gateway
- Paper trading simulation
- Training mode
- Backtesting mode

Usage:
    python main.py --mode live --config config/config.yaml
    python main.py --mode paper --config config/config.yaml
    python main.py --mode train --config config/config.yaml
    python main.py --mode backtest --model checkpoints/best_model.pt
"""

import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import yaml
import torch
import pandas as pd
import numpy as np

from data_pipeline import OHLCVPreprocessor, CyclicalTimeEncoder
from models import MetaAgent
from trading import IBConnector, OrderManager, PositionManager, PaperTrader
from trading.ib_connector import ConnectionConfig, DataStreamer, MarketData
from trading.order_manager import OrderSide
from utils import setup_logger, get_logger, get_device, log_trade, log_performance


class TradingSystem:
    """
    Main trading system that orchestrates all components.
    """

    def __init__(
        self,
        config: dict,
        model_path: str,
        paper_trading: bool = True
    ):
        """
        Args:
            config: Configuration dictionary
            model_path: Path to trained model
            paper_trading: Use paper trading mode
        """
        self.config = config
        self.paper_trading = paper_trading
        self.running = False

        self.logger = get_logger("system")

        # Setup device
        self.device = get_device(prefer_gpu=False)  # CPU for inference

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Initialize preprocessors
        self.preprocessor = OHLCVPreprocessor()
        self.time_encoder = CyclicalTimeEncoder()

        # Data buffer for model input
        self.data_buffer: list = []
        self.sequence_length = config['data']['sequence_length']

        # Trading components
        if paper_trading:
            self.trader = PaperTrader(
                initial_capital=config['backtest']['initial_capital'],
                commission_per_side=config['backtest']['commission'],
                slippage_points=config['backtest']['slippage'],
                max_position=config['trading']['max_position']
            )
            self.connector = None
        else:
            self.connector = IBConnector(
                ConnectionConfig(
                    host=config['trading']['ib_host'],
                    port=config['trading']['ib_port'],
                    client_id=config['trading']['client_id']
                ),
                paper_trading=paper_trading
            )
            self.trader = None

        # State tracking
        self.last_signal_time = None
        self.daily_trades = 0
        self.daily_pnl = 0.0

    def _load_model(self, model_path: str) -> MetaAgent:
        """Load trained model."""
        self.logger.info(f"Loading model from {model_path}")

        # Estimate input dimension
        input_dim = self.preprocessor.num_features + self.time_encoder.num_features

        model = MetaAgent(
            input_dim=input_dim,
            embedding_dim=self.config['model']['transformer']['d_model'],
            n_heads=4,
            transformer_layers=self.config['model']['transformer']['n_layers'],
            transformer_heads=self.config['model']['transformer']['n_heads'],
            transformer_ff=self.config['model']['transformer']['d_ff'],
            dropout=0.0,  # No dropout for inference
            profit_hidden=self.config['model']['profit_agent']['hidden_dim'],
            risk_hidden=self.config['model']['risk_agent']['hidden_dim']
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        self.logger.info("Model loaded successfully")
        return model

    def _prepare_features(self) -> torch.Tensor:
        """Prepare features from data buffer."""
        if len(self.data_buffer) < self.sequence_length:
            return None

        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer[-self.sequence_length:])
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Process features
        price_features = self.preprocessor.transform(df, use_rolling_norm=True)
        time_features = self.time_encoder.encode(df.index)

        features = pd.concat([price_features, time_features], axis=1).fillna(0)

        tensor = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def _get_trading_decision(
        self,
        features: torch.Tensor,
        current_price: float,
        atr: float
    ) -> dict:
        """Get trading decision from model."""
        # Get position state
        if self.paper_trading:
            pos_state = self.trader.position_manager.get_position_state()
            risk_state_dict = self.trader.position_manager.get_risk_state()
        else:
            pos_state = {'side': 0, 'unrealized_pnl': 0, 'holding_time': 0, 'pnl_pct': 0}
            risk_state_dict = {'current_drawdown': 0, 'daily_pnl': 0, 'position_exposure': 0}

        position_state = torch.tensor([
            pos_state['side'] if 'side' in pos_state else 0,
            pos_state.get('pnl_pct', 0),
            min(pos_state.get('holding_time', 0) / 240, 1.0)
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        risk_state = torch.tensor([
            risk_state_dict['current_drawdown'],
            risk_state_dict['daily_pnl'],
            0.5,
            risk_state_dict['position_exposure']
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        trade_state = torch.zeros(1, 5, device=self.device)

        atr_tensor = torch.tensor([[atr]], dtype=torch.float32).to(self.device)
        price_tensor = torch.tensor([[current_price]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                features,
                position_state,
                risk_state,
                trade_state,
                atr_tensor,
                price_tensor,
                deterministic=True
            )

        return {
            'action': outputs['action'].item(),
            'position_size': outputs['position_size'].item(),
            'stop_price': outputs['stop_price'].item(),
            'target_price': outputs['target_price'].item(),
            'confidence': outputs['confidence'].item()
        }

    def on_bar(self, data: MarketData):
        """Handle new bar data."""
        # Add to buffer
        self.data_buffer.append({
            'timestamp': data.timestamp,
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        })

        # Trim buffer
        if len(self.data_buffer) > self.sequence_length * 2:
            self.data_buffer = self.data_buffer[-self.sequence_length * 2:]

        # Fit preprocessor on initial data
        if len(self.data_buffer) == self.sequence_length:
            df = pd.DataFrame(self.data_buffer)
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
            self.preprocessor.fit(df)

        # Check if we have enough data
        if len(self.data_buffer) < self.sequence_length:
            return

        # Prepare features
        features = self._prepare_features()
        if features is None:
            return

        # Calculate ATR
        df = pd.DataFrame(self.data_buffer[-14:])
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = tr.mean()

        # Get decision
        decision = self._get_trading_decision(features, data.close, atr)

        self.logger.debug(
            f"Signal: action={decision['action']}, "
            f"size={decision['position_size']:.2f}, "
            f"conf={decision['confidence']:.2f}"
        )

        # Execute if confidence threshold met
        if decision['confidence'] > 0.6:
            self._execute_decision(decision, data.close, atr)

    def _execute_decision(self, decision: dict, price: float, atr: float):
        """Execute trading decision."""
        action = decision['action']
        position_size = decision['position_size']
        stop_price = decision['stop_price']
        target_price = decision['target_price']

        max_position = self.config['trading']['max_position']

        if self.paper_trading:
            current_position = self.trader.get_position()

            if action == 3 and current_position:
                self.trader.close_position()
                log_trade("CLOSE", "NQ", current_position.quantity, price)

            elif action == 1:  # Buy
                if current_position and current_position.is_short:
                    self.trader.close_position()

                if not current_position or current_position.is_flat:
                    qty = max(1, min(int(abs(position_size) * max_position), max_position))
                    self.trader.place_bracket_order(
                        OrderSide.BUY, qty, stop_price, target_price
                    )
                    log_trade("BUY", "NQ", qty, price, stop=stop_price, target=target_price)

            elif action == 2:  # Sell
                if current_position and current_position.is_long:
                    self.trader.close_position()

                if not current_position or current_position.is_flat:
                    qty = max(1, min(int(abs(position_size) * max_position), max_position))
                    self.trader.place_bracket_order(
                        OrderSide.SELL, qty, stop_price, target_price
                    )
                    log_trade("SELL", "NQ", qty, price, stop=stop_price, target=target_price)

        else:
            # Live trading with IB
            # TODO: Implement live trading logic
            pass

    def start(self):
        """Start the trading system."""
        self.logger.info("Starting trading system...")
        self.running = True

        if self.paper_trading:
            self.logger.info("Running in paper trading mode")
            # Paper trading doesn't need IB connection
            # Data would come from a simulation or external feed
            self.logger.info("Paper trading started. Waiting for data...")
        else:
            # Connect to IB Gateway
            self.logger.info("Connecting to IB Gateway...")
            connected = self.connector.connect_sync()

            if not connected:
                self.logger.error("Failed to connect to IB Gateway")
                return False

            # Get NQ contract
            contract = self.connector.get_nq_contract()

            # Subscribe to real-time data
            self.logger.info(f"Subscribing to {contract.symbol}...")
            self.connector.subscribe_realtime_bars(contract, self.on_bar)

            # Run event loop
            self.logger.info("Trading system running. Press Ctrl+C to stop.")

            try:
                while self.running:
                    self.connector.sleep(1)
            except KeyboardInterrupt:
                self.stop()

        return True

    def stop(self):
        """Stop the trading system."""
        self.logger.info("Stopping trading system...")
        self.running = False

        # Close any open positions
        if self.paper_trading and self.trader.get_position():
            self.trader.close_position()
        elif self.connector:
            # Close live positions if needed
            pass

        # Disconnect
        if self.connector:
            self.connector.disconnect()

        # Print final statistics
        if self.paper_trading:
            stats = self.trader.get_statistics()
            log_performance(stats)

        self.logger.info("Trading system stopped")


def main():
    parser = argparse.ArgumentParser(description="NQ Multi-Agent Trading System")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper", "train", "backtest"],
        default="paper",
        help="Operating mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/csv",
        help="Path to data folder"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_level=args.log_level)
    logger = get_logger("main")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Handle different modes
    if args.mode == "train":
        logger.info("Starting training mode...")
        from train import main as train_main
        sys.argv = ['train.py', '--config', args.config]
        train_main()

    elif args.mode == "backtest":
        logger.info("Starting backtest mode...")
        from backtest import main as backtest_main
        sys.argv = ['backtest.py', '--model', args.model, '--data', args.data, '--config', args.config]
        backtest_main()

    elif args.mode in ["live", "paper"]:
        # Check model exists
        if not Path(args.model).exists():
            logger.error(f"Model not found: {args.model}")
            logger.info("Train a model first with: python main.py --mode train")
            sys.exit(1)

        # Create trading system
        paper_mode = args.mode == "paper"
        system = TradingSystem(config, args.model, paper_trading=paper_mode)

        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal")
            system.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start system
        system.start()


if __name__ == "__main__":
    main()
