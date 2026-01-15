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
        Compute combined loss designed to produce VARYING trading signals.

        Key insight from research: Standard regression loss causes mode collapse
        (constant outputs). We need to:
        1. Emphasize CLASSIFICATION (direction) over regression
        2. Penalize LOW VARIANCE in outputs (constant signals = bad)
        3. Reward CORRECT direction weighted by magnitude
        4. Encourage signal DIVERSITY across the batch

        Args:
            outputs: Model outputs
            targets: Target returns
            position_state: Position state

        Returns:
            (total_loss, loss_components)
        """
        loss_components = {}
        position_size = outputs['position_size']
        batch_size = position_size.size(0)

        # =========================================================
        # 1. DIRECTION CLASSIFICATION (Primary objective)
        # =========================================================
        # Use wider thresholds for clearer signals
        target_direction = torch.where(
            targets > 0.0005,
            torch.tensor(2, device=self.device),  # UP
            torch.where(
                targets < -0.0005,
                torch.tensor(0, device=self.device),  # DOWN
                torch.tensor(1, device=self.device)   # NEUTRAL
            )
        )

        # Weight classes to handle imbalance (neutral is often most common)
        class_weights = torch.tensor([1.5, 0.5, 1.5], device=self.device)
        direction_loss = nn.CrossEntropyLoss(weight=class_weights)(
            outputs['action_probs'][:, :3],
            target_direction
        )
        loss_components['direction'] = direction_loss.item()

        # =========================================================
        # 2. SIGNAL VARIATION PENALTY (Critical for avoiding mode collapse)
        # =========================================================
        # Penalize if all outputs in the batch are too similar
        position_std = position_size.std()
        # We want std > 0.3 for good variation; penalize if std is too low
        target_std = 0.3
        variation_penalty = torch.relu(target_std - position_std) ** 2
        loss_components['variation'] = variation_penalty.item()

        # Also penalize if outputs are saturated (all near +1 or -1)
        saturation = (torch.abs(position_size) > 0.95).float().mean()
        saturation_penalty = saturation
        loss_components['saturation'] = saturation_penalty.item()

        # =========================================================
        # 3. DIRECTIONAL ACCURACY (Correct sign matters more than magnitude)
        # =========================================================
        # Reward correct direction, penalize wrong direction
        # This is different from return_loss which just maximizes avg return
        predicted_sign = torch.sign(position_size)
        target_sign = torch.sign(targets)

        # Correct direction = same sign (positive reward)
        # Wrong direction = opposite sign (penalty)
        direction_match = predicted_sign * target_sign  # +1 if correct, -1 if wrong, 0 if either is 0

        # Weight by magnitude of actual move (bigger moves matter more)
        weighted_accuracy = direction_match * torch.abs(targets) * 100
        accuracy_loss = -weighted_accuracy.mean()  # Negative because we maximize
        loss_components['accuracy'] = accuracy_loss.item()

        # =========================================================
        # 4. POSITION CHANGE ENCOURAGEMENT
        # =========================================================
        # The model should produce DIFFERENT outputs for DIFFERENT inputs
        # Shuffle and compare: outputs should differ when inputs differ
        if batch_size > 1:
            # Compare adjacent samples - they should have different outputs
            # if market conditions are different
            position_diff = torch.abs(position_size[1:] - position_size[:-1])
            target_diff = torch.abs(targets[1:] - targets[:-1])

            # If targets are different, positions should be different
            # Penalize when targets differ but positions don't
            should_differ = (target_diff > 0.001).float()
            change_penalty = (should_differ * torch.relu(0.1 - position_diff)).mean()
            loss_components['change'] = change_penalty.item()
        else:
            change_penalty = torch.tensor(0.0, device=self.device)
            loss_components['change'] = 0.0

        # =========================================================
        # 5. VALUE ESTIMATION (Critic loss)
        # =========================================================
        if 'profit_output' in outputs:
            value_pred = outputs['profit_output']['value']
            value_loss = nn.MSELoss()(value_pred, targets * 100)  # Scale targets
            loss_components['value'] = value_loss.item()
        else:
            value_loss = torch.tensor(0.0, device=self.device)
            loss_components['value'] = 0.0

        # =========================================================
        # 6. ENTROPY REGULARIZATION (Exploration)
        # =========================================================
        if 'profit_output' in outputs and 'entropy' in outputs['profit_output']:
            entropy = outputs['profit_output']['entropy'].mean()
            entropy_loss = -0.01 * entropy  # Encourage higher entropy
            loss_components['entropy'] = entropy_loss.item()
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)
            loss_components['entropy'] = 0.0

        # =========================================================
        # 7. ACTION DISTRIBUTION (Encourage balanced buy/sell/hold)
        # =========================================================
        action_probs = outputs['action_probs']
        # Target: roughly equal distribution, not all hold
        # Penalize if one action dominates
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        action_diversity_loss = -0.1 * action_entropy  # Higher entropy = more diverse
        loss_components['action_div'] = action_diversity_loss.item()

        # =========================================================
        # COMBINED LOSS - Emphasize classification and variation
        # =========================================================
        total_loss = (
            0.30 * direction_loss +       # Primary: classify direction correctly
            0.20 * accuracy_loss +        # Correct direction weighted by magnitude
            0.20 * variation_penalty +    # CRITICAL: prevent constant outputs
            0.10 * saturation_penalty +   # Prevent saturated outputs
            0.10 * change_penalty +       # Encourage position changes
            0.05 * value_loss +           # Value estimation
            0.03 * entropy_loss +         # Exploration
            0.02 * action_diversity_loss  # Action distribution
        )

        loss_components['total'] = total_loss.item()

        return total_loss, loss_components

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {k: [] for k in ['direction', 'variation', 'saturation', 'accuracy',
                                          'change', 'value', 'entropy', 'action_div', 'total']}

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
        import sys
        print(f"  [DEBUG] Starting evaluation with {len(loader)} batches...", flush=True)
        sys.stdout.flush()

        all_signals = []
        all_prices = []
        all_targets = []

        for batch_idx, batch in enumerate(loader):
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

            pos_size = outputs['position_size'].cpu().numpy()
            all_signals.append(pos_size)
            all_prices.append(close_price.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Debug: Print first batch signal stats
            if batch_idx == 0:
                print(f"  [DEBUG] First batch signals: min={pos_size.min():.4f}, max={pos_size.max():.4f}, "
                      f"mean={pos_size.mean():.4f}, >0.005: {(np.abs(pos_size) > 0.005).sum()}/{len(pos_size)}", flush=True)

        # Concatenate
        signals = np.concatenate(all_signals)
        prices = np.concatenate(all_prices)
        targets = np.concatenate(all_targets)

        # Debug: Print signal statistics
        print(f"  Signal stats: min={signals.min():.4f}, max={signals.max():.4f}, "
              f"mean={signals.mean():.4f}, std={signals.std():.4f}, "
              f"non-zero={np.sum(np.abs(signals) > 0.005)}/{len(signals)}", flush=True)

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
