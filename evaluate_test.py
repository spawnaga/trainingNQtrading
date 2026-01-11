"""
Test Set Evaluation Script
==========================
Evaluates the trained model on the held-out test set.
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.loader import create_data_loaders
from models.simple_meta_agent import SimpleMetaAgent


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    embedding_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff: int = 512
    dropout: float = 0.1
    sequence_length: int = 60
    batch_size: int = 256


def evaluate_model(model, loader, device):
    """Comprehensive evaluation on test set."""
    model.eval()

    all_positions = []
    all_targets = []
    all_prices = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            close_price = batch['close_price'].to(device)

            outputs = model(features)

            all_positions.append(outputs['position_size'].cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_prices.append(close_price.cpu().numpy())

    positions = np.concatenate(all_positions)
    targets = np.concatenate(all_targets)
    prices = np.concatenate(all_prices)

    # Compute comprehensive metrics
    metrics = compute_trading_metrics(positions, targets, prices)

    return metrics, positions, targets, prices


def compute_trading_metrics(positions: np.ndarray, targets: np.ndarray, prices: np.ndarray) -> Dict:
    """Compute comprehensive trading metrics."""

    # Direction accuracy
    pred_signs = np.sign(positions)
    target_signs = np.sign(targets)
    direction_acc = (pred_signs == target_signs).mean()

    # P&L calculations
    pnl_per_trade = positions * targets * 100  # Scaled for readability
    total_pnl = pnl_per_trade.sum()
    avg_pnl = pnl_per_trade.mean()

    # Sharpe ratio (annualized)
    pnl_std = pnl_per_trade.std()
    sharpe = (avg_pnl / (pnl_std + 1e-8)) * np.sqrt(252 * 390)  # 390 minutes per day

    # Win rate
    wins = (pnl_per_trade > 0).sum()
    losses = (pnl_per_trade < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5

    # Average win/loss
    avg_win = pnl_per_trade[pnl_per_trade > 0].mean() if wins > 0 else 0
    avg_loss = pnl_per_trade[pnl_per_trade < 0].mean() if losses > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Trading activity
    position_changes = np.abs(np.diff(np.sign(positions)))
    trades = (position_changes > 0).sum()
    avg_position = np.abs(positions).mean()
    position_std = positions.std()

    # Maximum drawdown
    cumulative_pnl = np.cumsum(pnl_per_trade)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    # Calmar ratio (annualized return / max drawdown)
    annual_return = total_pnl * (252 * 390 / len(pnl_per_trade))
    calmar = annual_return / (max_drawdown + 1e-8)

    # Sortino ratio (uses downside deviation)
    negative_pnl = pnl_per_trade[pnl_per_trade < 0]
    downside_std = negative_pnl.std() if len(negative_pnl) > 0 else 1e-8
    sortino = (avg_pnl / (downside_std + 1e-8)) * np.sqrt(252 * 390)

    return {
        'direction_accuracy': direction_acc,
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': avg_pnl,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': trades,
        'total_signals': len(positions),
        'avg_position_size': avg_position,
        'position_std': position_std,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'cumulative_pnl': cumulative_pnl[-1],
        'wins': wins,
        'losses': losses
    }


def main():
    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    config = EvalConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load configuration
    with open('config/config_remote.yaml') as f:
        yaml_config = yaml.safe_load(f)

    # Load data - full dataset
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader, feature_info = create_data_loaders(
        csv_folder=yaml_config['data']['csv_folder'],
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        train_split=0.7,
        val_split=0.15,
        start_date=None,
        end_date=None,
        num_workers=4
    )
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Test samples: {len(test_loader) * config.batch_size}")

    # Get input dimension
    sample = next(iter(test_loader))
    input_dim = sample['features'].shape[-1]
    print(f"   Input dim: {input_dim}")

    # Create model
    print("\n2. Loading model...")
    model = SimpleMetaAgent(
        input_dim=input_dim,
        embedding_dim=config.embedding_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_ff=config.transformer_ff,
        dropout=config.dropout
    ).to(device)

    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model_fixed.pt'
    if not Path(checkpoint_path).exists():
        checkpoint_path = 'best_model_fixed.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Checkpoint Sharpe: {checkpoint['sharpe']:.2f}")

    # Evaluate on test set
    print("\n3. Evaluating on test set...")
    metrics, positions, targets, prices = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print("\n--- Performance Metrics ---")
    print(f"Direction Accuracy:  {metrics['direction_accuracy']:.2%}")
    print(f"Win Rate:            {metrics['win_rate']:.2%}")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")

    print("\n--- Risk-Adjusted Returns ---")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:.2f}")

    print("\n--- P&L Statistics ---")
    print(f"Total PnL:           {metrics['total_pnl']:.2f}")
    print(f"Avg PnL per Trade:   {metrics['avg_pnl_per_trade']:.4f}")
    print(f"Average Win:         {metrics['avg_win']:.4f}")
    print(f"Average Loss:        {metrics['avg_loss']:.4f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']:.2f}")

    print("\n--- Trading Activity ---")
    print(f"Total Signals:       {metrics['total_signals']:,}")
    print(f"Trade Count:         {metrics['total_trades']:,}")
    print(f"Wins:                {metrics['wins']:,}")
    print(f"Losses:              {metrics['losses']:,}")
    print(f"Avg Position Size:   {metrics['avg_position_size']:.4f}")
    print(f"Position Std:        {metrics['position_std']:.4f}")

    print("\n" + "=" * 70)

    # Save detailed results
    results = {
        'metrics': metrics,
        'config': {
            'embedding_dim': config.embedding_dim,
            'transformer_layers': config.transformer_layers,
            'transformer_heads': config.transformer_heads,
            'batch_size': config.batch_size,
            'sequence_length': config.sequence_length
        }
    }

    import json
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nResults saved to test_results.json")

    return metrics


if __name__ == '__main__':
    main()
