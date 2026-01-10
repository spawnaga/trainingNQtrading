"""
Main training pipeline for the multi-agent trading system.

Supports:
- Mixed precision training
- Gradient accumulation
- Checkpointing
- TensorBoard logging
- Multi-GPU training
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .fitness import FitnessEvaluator, PerformanceMetrics, simulate_trades_from_signals


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    batch_size: int = 64
    accumulation_steps: int = 1
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 10
    eval_every: int = 5
    device: str = "cuda"


class Trainer:
    """
    Trainer for the multi-agent trading system.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda"
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            learning_rate: Learning rate (used if config not provided)
            weight_decay: Weight decay (used if config not provided)
            device: Device to use
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config or TrainingConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device
        )

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

        # Mixed precision
        self.scaler = GradScaler() if self.config.mixed_precision else None

        # Fitness evaluator
        self.fitness_evaluator = FitnessEvaluator()

        # Logging
        self.writer = None
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_fitness = float('-inf')
        self.train_losses = []
        self.val_metrics = []

    def _setup_logging(self):
        """Setup TensorBoard logging."""
        log_dir = Path(self.config.log_dir) / f"run_{int(time.time())}"
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        position_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss from all agents.

        Args:
            outputs: Model outputs
            targets: Target returns
            position_state: Position state

        Returns:
            (total_loss, loss_components)
        """
        loss_components = {}

        # 1. Direction prediction loss (cross-entropy)
        # Target: 0=down, 1=neutral, 2=up
        target_direction = torch.where(
            targets > 0.001,
            torch.tensor(2, device=self.device),
            torch.where(
                targets < -0.001,
                torch.tensor(0, device=self.device),
                torch.tensor(1, device=self.device)
            )
        )
        direction_loss = nn.CrossEntropyLoss()(
            outputs['action_probs'][:, :3],  # Exclude 'close' action
            target_direction
        )
        loss_components['direction'] = direction_loss.item()

        # 2. Return prediction loss (MSE for profit agent)
        position_size = outputs['position_size']
        predicted_return = position_size * targets
        return_loss = -predicted_return.mean()  # Negative because we maximize
        loss_components['return'] = return_loss.item()

        # 3. Value estimation loss (for critic)
        if 'profit_output' in outputs:
            value_pred = outputs['profit_output']['value']
            # Value target is the actual return achieved
            value_loss = nn.MSELoss()(value_pred, targets)
            loss_components['value'] = value_loss.item()
        else:
            value_loss = torch.tensor(0.0, device=self.device)

        # 4. Risk-adjusted return (Sharpe-like objective)
        if len(predicted_return) > 1:
            sharpe_proxy = predicted_return.mean() / (predicted_return.std() + 1e-6)
            sharpe_loss = -sharpe_proxy
            loss_components['sharpe'] = sharpe_loss.item()
        else:
            sharpe_loss = torch.tensor(0.0, device=self.device)

        # 5. Entropy regularization (encourage exploration)
        if 'profit_output' in outputs:
            entropy = outputs['profit_output']['entropy'].mean()
            entropy_loss = -0.01 * entropy
            loss_components['entropy'] = entropy_loss.item()
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        # 6. INACTION PENALTY - Penalize the agent for not trading
        # Penalize position sizes close to zero
        abs_position = torch.abs(position_size)
        inaction_penalty = torch.exp(-5.0 * abs_position).mean()  # High when position ~0
        loss_components['inaction'] = inaction_penalty.item()

        # 7. ACTION DIVERSITY - Encourage taking buy/sell actions vs hold
        action_probs = outputs['action_probs']
        # Penalize high probability on "hold" (action index 1 = neutral)
        hold_prob = action_probs[:, 1] if action_probs.size(1) > 1 else torch.zeros_like(position_size)
        hold_penalty = hold_prob.mean()
        loss_components['hold_penalty'] = hold_penalty.item()

        # 8. CONFIDENCE PENALTY - Penalize low confidence (encourages decisive actions)
        confidence = outputs.get('confidence', torch.ones_like(position_size) * 0.5)
        low_confidence_penalty = torch.relu(0.5 - confidence).mean()  # Penalize confidence < 0.5
        loss_components['low_conf'] = low_confidence_penalty.item()

        # 9. TRADE WHEN OPPORTUNITY EXISTS - Penalize not trading when there's movement
        abs_target = torch.abs(targets)
        missed_opportunity = (abs_target * (1.0 - abs_position)).mean()  # High target + low position = bad
        loss_components['missed_opp'] = missed_opportunity.item()

        # Combined loss with inaction penalties
        total_loss = (
            0.20 * direction_loss +
            0.20 * return_loss +
            0.10 * value_loss +
            0.10 * sharpe_loss +
            0.05 * entropy_loss +
            0.15 * inaction_penalty +      # NEW: Penalize not trading
            0.10 * hold_penalty +           # NEW: Penalize hold action
            0.05 * low_confidence_penalty + # NEW: Penalize low confidence
            0.05 * missed_opportunity       # NEW: Penalize missing opportunities
        )

        loss_components['total'] = total_loss.item()

        return total_loss, loss_components

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {k: [] for k in ['direction', 'return', 'value', 'sharpe', 'entropy',
                                          'inaction', 'hold_penalty', 'low_conf', 'missed_opp', 'total']}

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)
            close_price = batch['close_price'].to(self.device)

            batch_size = features.size(0)

            # Create dummy states (in real training, these would come from simulation)
            position_state = torch.zeros(batch_size, 3, device=self.device)
            risk_state = torch.zeros(batch_size, 4, device=self.device)
            trade_state = torch.zeros(batch_size, 5, device=self.device)
            atr = close_price * 0.01  # Approximate ATR as 1% of price

            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(
                        features,
                        position_state,
                        risk_state,
                        trade_state,
                        atr.unsqueeze(-1),
                        close_price.unsqueeze(-1)
                    )
                    loss, loss_components = self._compute_loss(outputs, targets, position_state)
                    loss = loss / self.config.accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(
                    features,
                    position_state,
                    risk_state,
                    trade_state,
                    atr.unsqueeze(-1),
                    close_price.unsqueeze(-1)
                )
                loss, loss_components = self._compute_loss(outputs, targets, position_state)
                loss = loss / self.config.accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            epoch_losses.append(loss.item() * self.config.accumulation_steps)
            for k, v in loss_components.items():
                epoch_metrics[k].append(v)

        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        avg_metrics['loss'] = np.mean(epoch_losses)

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> PerformanceMetrics:
        """
        Evaluate model on data loader.

        Returns:
            PerformanceMetrics object
        """
        self.model.eval()

        all_signals = []
        all_prices = []
        all_targets = []

        for batch in loader:
            features = batch['features'].to(self.device)
            close_price = batch['close_price'].to(self.device)
            targets = batch['target'].to(self.device)

            batch_size = features.size(0)

            # Create dummy states
            position_state = torch.zeros(batch_size, 3, device=self.device)
            risk_state = torch.zeros(batch_size, 4, device=self.device)
            trade_state = torch.zeros(batch_size, 5, device=self.device)
            atr = close_price * 0.01

            outputs = self.model(
                features,
                position_state,
                risk_state,
                trade_state,
                atr.unsqueeze(-1),
                close_price.unsqueeze(-1),
                deterministic=True
            )

            all_signals.append(outputs['position_size'].cpu().numpy())
            all_prices.append(close_price.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Concatenate
        signals = np.concatenate(all_signals)
        prices = np.concatenate(all_prices)
        targets = np.concatenate(all_targets)

        # Simulate trades
        import pandas as pd
        timestamps = pd.date_range(start='2020-01-01', periods=len(signals), freq='1min')

        trades, equity_curve = simulate_trades_from_signals(
            signals, prices, timestamps
        )

        # Calculate metrics
        metrics = self.fitness_evaluator.evaluate_trades(trades)

        return metrics

    def train(self, epochs: Optional[int] = None):
        """
        Main training loop.

        Args:
            epochs: Number of epochs (uses config if not specified)
        """
        epochs = epochs or self.config.epochs

        if self.writer is None:
            self._setup_logging()

        print(f"Training for {epochs} epochs on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - start_time

            # Update scheduler
            self.scheduler.step()

            # Log training metrics
            self.train_losses.append(train_metrics['loss'])

            if self.writer:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, epoch)
                self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], epoch)

            print(f"Epoch {epoch:3d} | Loss: {train_metrics['loss']:.4f} | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f} | Time: {train_time:.1f}s")

            # Evaluate
            if (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.evaluate(self.val_loader)
                fitness = self.fitness_evaluator.calculate_fitness(val_metrics)

                self.val_metrics.append({
                    'epoch': epoch,
                    'fitness': fitness,
                    'metrics': val_metrics
                })

                if self.writer:
                    self.writer.add_scalar('val/fitness', fitness, epoch)
                    self.writer.add_scalar('val/sharpe', val_metrics.sharpe_ratio, epoch)
                    self.writer.add_scalar('val/win_rate', val_metrics.win_rate, epoch)
                    self.writer.add_scalar('val/max_drawdown', val_metrics.max_drawdown, epoch)

                print(f"  Val | Fitness: {fitness:.4f} | Sharpe: {val_metrics.sharpe_ratio:.2f} | "
                      f"Win Rate: {val_metrics.win_rate:.1%} | Max DD: {val_metrics.max_drawdown:.1%}")

                # Save best model
                if fitness > self.best_val_fitness:
                    self.best_val_fitness = fitness
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best model saved!")

            # Regular checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Final save
        self.save_checkpoint('final_model.pt')

        if self.writer:
            self.writer.close()

        print(f"\nTraining complete! Best validation fitness: {self.best_val_fitness:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_fitness': self.best_val_fitness,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_fitness = checkpoint['best_val_fitness']
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for the profit agent.
    """

    def __init__(
        self,
        model: nn.Module,
        env,  # Trading environment
        device: str = "cuda",
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64
    ):
        self.model = model
        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def collect_rollouts(self, n_steps: int) -> Dict[str, torch.Tensor]:
        """Collect experience from environment."""
        # Implementation depends on trading environment
        pass

    def update(self, rollouts: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform PPO update."""
        # Implementation of PPO update
        pass
