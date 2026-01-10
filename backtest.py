"""
Backtesting engine for the NQ Multi-Agent Trading System.

Usage:
    python backtest.py --model checkpoints/best_model.pt --data data/csv
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_pipeline import CSVDataLoader, OHLCVPreprocessor, CyclicalTimeEncoder, TradingDataset
from models import MetaAgent
from trading import PaperTrader
from trading.position_manager import PositionSide
from trading.order_manager import OrderSide
from utils import setup_logger, get_logger, get_device, TradingMetrics
from utils.metrics import calculate_comprehensive_metrics
from training.fitness import FitnessEvaluator


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 2.25,
        slippage: float = 0.25,
        point_value: float = 20.0,
        max_position: int = 2,
        sequence_length: int = 60,
        device: str = "cuda"
    ):
        """
        Args:
            model: Trained MetaAgent model
            data: OHLCV DataFrame
            initial_capital: Starting capital
            commission: Commission per contract per side
            slippage: Slippage in points
            point_value: Dollar value per point
            max_position: Maximum contracts
            sequence_length: Model input sequence length
            device: Compute device
        """
        self.model = model
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.point_value = point_value
        self.max_position = max_position
        self.sequence_length = sequence_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # Initialize preprocessors
        self.preprocessor = OHLCVPreprocessor()
        self.time_encoder = CyclicalTimeEncoder()

        # Paper trader for simulation
        self.paper_trader = PaperTrader(
            initial_capital=initial_capital,
            commission_per_side=commission,
            slippage_points=slippage,
            point_value=point_value,
            max_position=max_position
        )

        # Results tracking
        self.equity_curve: List[float] = [initial_capital]
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []

        self.logger = get_logger("backtest")

    def _prepare_features(self, idx: int) -> Optional[torch.Tensor]:
        """Prepare input features for the model."""
        if idx < self.sequence_length:
            return None

        # Get sequence window
        window_data = self.data.iloc[idx - self.sequence_length:idx]

        # Process features
        price_features = self.preprocessor.transform(window_data, use_rolling_norm=True)
        time_features = self.time_encoder.encode(window_data.index)

        # Combine and ensure numeric dtype
        features = pd.concat([price_features, time_features], axis=1).fillna(0)
        features = features.astype(np.float32)

        # Convert to tensor
        tensor = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def _get_model_decision(
        self,
        features: torch.Tensor,
        current_price: float,
        atr: float
    ) -> Dict:
        """Get trading decision from model."""
        batch_size = 1

        # Create state tensors
        pos_state = self.paper_trader.position_manager.get_position_state()
        position_state = torch.tensor([
            pos_state['side'],
            pos_state['unrealized_pnl'] / self.initial_capital if self.initial_capital > 0 else 0,
            min(pos_state['holding_time'] / 240, 1.0)  # Normalize to 4 hours
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        risk_state_dict = self.paper_trader.position_manager.get_risk_state()
        risk_state = torch.tensor([
            risk_state_dict['current_drawdown'],
            risk_state_dict['daily_pnl'],
            0.5,  # Placeholder volatility
            risk_state_dict['position_exposure']
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        trade_state = torch.zeros(1, 5, device=self.device)
        if pos_state['has_position']:
            trade_state[0, 0] = min(pos_state['holding_time'] / 240, 1.0)
            trade_state[0, 1] = pos_state['pnl_pct']

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

    def run(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        progress_bar: bool = True
    ) -> Dict:
        """
        Run backtest simulation.

        Args:
            start_idx: Starting index (default: sequence_length)
            end_idx: Ending index (default: len(data))
            progress_bar: Show progress bar

        Returns:
            Backtest results dictionary
        """
        self.logger.info("Starting backtest...")

        # Fit preprocessor on initial data
        self.preprocessor.fit(self.data.iloc[:self.sequence_length * 2])

        start_idx = start_idx or self.sequence_length
        end_idx = end_idx or len(self.data)

        # Calculate ATR for the entire series
        from ta.volatility import average_true_range
        atr_series = average_true_range(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            window=14
        ).fillna(0)

        iterator = range(start_idx, end_idx)
        if progress_bar:
            iterator = tqdm(iterator, desc="Backtesting")

        for idx in iterator:
            row = self.data.iloc[idx]
            current_price = row['close']
            atr = atr_series.iloc[idx]

            # Update market price
            self.paper_trader.update_market(
                current_price,
                row['low'],
                row['high']
            )

            # Get features
            features = self._prepare_features(idx)
            if features is None:
                self.equity_curve.append(self.paper_trader.get_equity())
                continue

            # Get model decision
            decision = self._get_model_decision(features, current_price, atr)

            # Record signal
            self.signals.append({
                'timestamp': self.data.index[idx],
                'price': current_price,
                **decision
            })

            # Execute trading logic
            self._execute_decision(decision, current_price, atr)

            # Record equity
            self.equity_curve.append(self.paper_trader.get_equity())

        # Close any remaining position
        if self.paper_trader.get_position():
            self.paper_trader.close_position()
            self.equity_curve.append(self.paper_trader.get_equity())

        # Calculate results
        results = self._calculate_results()

        return results

    def _execute_decision(self, decision: Dict, current_price: float, atr: float):
        """Execute trading decision."""
        action = decision['action']
        position_size = decision['position_size']
        stop_price = decision['stop_price']
        target_price = decision['target_price']
        confidence = decision['confidence']

        current_position = self.paper_trader.get_position()

        # Action mapping: 0=hold, 1=buy, 2=sell, 3=close
        if action == 3 and current_position:
            # Close position
            self.paper_trader.close_position()

        elif action == 1 and confidence > 0.5:
            # Buy signal
            if current_position and current_position.is_short:
                self.paper_trader.close_position()

            if not current_position or current_position.is_flat:
                quantity = max(1, min(abs(int(position_size * self.max_position)), self.max_position))
                self.paper_trader.place_bracket_order(
                    OrderSide.BUY,
                    quantity,
                    stop_price,
                    target_price
                )

        elif action == 2 and confidence > 0.5:
            # Sell signal
            if current_position and current_position.is_long:
                self.paper_trader.close_position()

            if not current_position or current_position.is_flat:
                quantity = max(1, min(abs(int(position_size * self.max_position)), self.max_position))
                self.paper_trader.place_bracket_order(
                    OrderSide.SELL,
                    quantity,
                    stop_price,
                    target_price
                )

    def _calculate_results(self) -> Dict:
        """Calculate backtest results and metrics."""
        equity_curve = np.array(self.equity_curve)
        trades = self.paper_trader.position_manager.closed_positions

        # Convert trades to dict format
        trade_dicts = [{
            'pnl': t.pnl,
            'holding_time': t.holding_time_minutes,
            'side': t.side.name,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price
        } for t in trades]

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            equity_curve,
            trade_dicts,
            annualization_factor=252 * 390  # Minutes in trading year
        )

        results = {
            'metrics': metrics,
            'equity_curve': equity_curve.tolist(),
            'trades': trade_dicts,
            'signals': self.signals,
            'final_equity': equity_curve[-1],
            'total_return': (equity_curve[-1] / equity_curve[0]) - 1
        }

        return results


def load_model(model_path: str, config: dict, input_dim: int) -> MetaAgent:
    """Load trained model from checkpoint."""
    model = MetaAgent(
        input_dim=input_dim,
        embedding_dim=config['model']['transformer']['d_model'],
        n_heads=4,
        transformer_layers=config['model']['transformer']['n_layers'],
        transformer_heads=config['model']['transformer']['n_heads'],
        transformer_ff=config['model']['transformer']['d_ff'],
        dropout=config['model']['transformer']['dropout'],
        profit_hidden=config['model']['profit_agent']['hidden_dim'],
        risk_hidden=config['model']['risk_agent']['hidden_dim']
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def main():
    parser = argparse.ArgumentParser(description="Backtest NQ Trading System")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/csv",
        help="Path to data folder"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger("main")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found, using defaults")
        config = {
            'model': {
                'transformer': {
                    'd_model': 256, 'n_heads': 8, 'n_layers': 4,
                    'd_ff': 1024, 'dropout': 0.1
                },
                'profit_agent': {'hidden_dim': 128},
                'risk_agent': {'hidden_dim': 64}
            },
            'data': {'sequence_length': 60}
        }

    # Load data
    logger.info(f"Loading data from {args.data}...")
    loader = CSVDataLoader(
        args.data,
        start_date=args.start_date,
        end_date=args.end_date
    )
    data = loader.load_all_files()

    # Estimate input dimension
    preprocessor = OHLCVPreprocessor()
    time_encoder = CyclicalTimeEncoder()
    input_dim = preprocessor.num_features + time_encoder.num_features

    # Load model
    logger.info(f"Loading model from {args.model}...")
    model = load_model(args.model, config, input_dim)

    # Create backtester
    backtester = Backtester(
        model=model,
        data=data,
        initial_capital=args.initial_capital,
        sequence_length=config['data']['sequence_length'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run backtest
    results = backtester.run()

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 50)
    logger.info(str(results['metrics']))

    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump({
            'final_equity': results['final_equity'],
            'total_return': results['total_return'],
            'metrics': results['metrics'].to_dict(),
            'num_trades': len(results['trades'])
        }, f, indent=2)

    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
