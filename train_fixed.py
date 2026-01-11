"""
Fixed Training Script
=====================
Based on diagnostics:
1. Model architecture is fine (gradients flow)
2. Direction accuracy ~50% is normal (market efficiency)
3. Original model needs proper initialization and monitoring

This script fixes the issues and runs training properly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.loader import create_data_loaders
from models.simple_meta_agent import SimpleMetaAgent


@dataclass
class TrainingConfig:
    """Training configuration with fixed hyperparameters."""
    # Model
    embedding_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff: int = 512
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4  # Lower learning rate
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 100
    eval_every: int = 5

    # Data
    sequence_length: int = 60
    target_horizon: int = 5


def init_weights_properly(model):
    """Initialize weights with proper gains."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Use Xavier for most layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # Special initialization for output layers
    if hasattr(model, 'action_head'):
        nn.init.xavier_uniform_(model.action_head.weight, gain=0.1)
    if hasattr(model, 'position_head'):
        for m in model.position_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)


def compute_loss(outputs, targets, device):
    """Simple MSE loss - same as working LSTM test."""
    position_size = outputs['position_size']

    # Just predict the sign of the target (direction)
    # This is what worked in test_horizons_v2.py
    target_signs = torch.sign(targets)
    loss = F.mse_loss(position_size, target_signs)

    return loss, {
        'mse': loss.item(),
        'total': loss.item()
    }


def evaluate(model, loader, device):
    """Evaluate model and compute trading metrics."""
    model.eval()

    all_positions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            close_price = batch['close_price'].to(device)

            # Create dummy states
            batch_size = features.size(0)
            position_state = torch.zeros(batch_size, 3, device=device)
            risk_state = torch.zeros(batch_size, 4, device=device)
            trade_state = torch.zeros(batch_size, 5, device=device)
            atr = close_price.unsqueeze(-1) * 0.01
            price = close_price.unsqueeze(-1)

            outputs = model(features, position_state, risk_state, trade_state, atr, price, deterministic=True)

            all_positions.append(outputs['position_size'].cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    positions = np.concatenate(all_positions)
    targets = np.concatenate(all_targets)

    # Compute metrics
    # Direction accuracy
    pred_signs = np.sign(positions)
    target_signs = np.sign(targets)
    direction_acc = (pred_signs == target_signs).mean()

    # Simulated P&L (simplified)
    pnl = positions * targets * 100  # Scale for readability
    total_pnl = pnl.sum()
    sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252 * 390)  # Annualized

    # Trading stats
    trades = np.sum(np.abs(np.diff(np.sign(positions))) > 0)
    avg_position = np.abs(positions).mean()

    # Win rate (for trades in direction of target)
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5

    return {
        'direction_acc': direction_acc,
        'sharpe': sharpe,
        'total_pnl': total_pnl,
        'trades': trades,
        'avg_position': avg_position,
        'win_rate': win_rate,
        'position_std': positions.std()
    }


def train():
    print("=" * 70)
    print("FIXED TRAINING SCRIPT")
    print("=" * 70)

    config = TrainingConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load configuration
    with open('config/config_remote.yaml') as f:
        yaml_config = yaml.safe_load(f)

    # Load data - start with small subset
    print("\n1. Loading data...")
    train_loader, val_loader, _, feature_info = create_data_loaders(
        csv_folder=yaml_config['data']['csv_folder'],
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        train_split=0.7,
        val_split=0.15,
        start_date=None,  # Use all available data
        end_date=None,
        num_workers=4
    )
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Get input dimension
    sample = next(iter(train_loader))
    input_dim = sample['features'].shape[-1]
    print(f"   Input dim: {input_dim}")

    # Create model with smaller size
    print("\n2. Creating model...")
    model = SimpleMetaAgent(
        input_dim=input_dim,
        embedding_dim=config.embedding_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_ff=config.transformer_ff,
        dropout=config.dropout
    ).to(device)

    # Apply proper initialization
    init_weights_properly(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Mixed precision
    scaler = GradScaler()

    print(f"\n3. Training for {config.epochs} epochs...")
    print("-" * 70)
    print(f"{'Epoch':>5} | {'Loss':>8} | {'DirAcc':>8} | {'Sharpe':>8} | {'WinRate':>8} | {'PosStd':>8}")
    print("-" * 70)

    best_sharpe = -float('inf')

    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_losses = []
        grad_norms = []

        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            close_price = batch['close_price'].to(device)

            # Create dummy states
            batch_size = features.size(0)
            position_state = torch.zeros(batch_size, 3, device=device)
            risk_state = torch.zeros(batch_size, 4, device=device)
            trade_state = torch.zeros(batch_size, 5, device=device)
            atr = close_price.unsqueeze(-1) * 0.01
            price = close_price.unsqueeze(-1)

            optimizer.zero_grad()

            with autocast():
                outputs = model(features, position_state, risk_state, trade_state, atr, price)
                loss, loss_components = compute_loss(outputs, targets, device)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        scheduler.step()

        avg_loss = np.mean(epoch_losses)
        avg_grad = np.mean(grad_norms)

        # Evaluate every N epochs
        if (epoch + 1) % config.eval_every == 0 or epoch == 0:
            metrics = evaluate(model, val_loader, device)

            improved = metrics['sharpe'] > best_sharpe
            if improved:
                best_sharpe = metrics['sharpe']
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sharpe': best_sharpe
                }, 'best_model_fixed.pt')

            print(f"{epoch:5d} | {avg_loss:8.4f} | {metrics['direction_acc']:8.1%} | "
                  f"{metrics['sharpe']:8.2f} | {metrics['win_rate']:8.1%} | "
                  f"{metrics['position_std']:8.4f} {'*' if improved else ''}")

            # Warning checks
            if metrics['position_std'] < 0.01:
                print(f"       ⚠️ Predictions collapsed (std={metrics['position_std']:.4f})")
            if avg_grad < 0.001:
                print(f"       ⚠️ Vanishing gradients (norm={avg_grad:.6f})")
        else:
            print(f"{epoch:5d} | {avg_loss:8.4f} | {'--':>8} | {'--':>8} | {'--':>8} | {'--':>8}")

    print("-" * 70)
    print(f"\nTraining complete!")
    print(f"Best Sharpe Ratio: {best_sharpe:.2f}")

    # Final evaluation
    print("\nFinal evaluation:")
    final_metrics = evaluate(model, val_loader, device)
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    train()
