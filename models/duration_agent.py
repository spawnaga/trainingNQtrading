"""
Trade Duration Agent for determining optimal holding periods.

Classifies trades into scalp, intraday, or swing categories
and provides exit timing recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from enum import IntEnum


class TradeDuration(IntEnum):
    """Trade duration categories."""
    SCALP = 0       # < 5 minutes
    INTRADAY = 1    # 5 minutes to 4 hours
    SWING = 2       # > 4 hours


class TradeDurationAgent(nn.Module):
    """
    Agent for determining optimal trade duration and exit timing.

    Outputs:
    - Duration class: scalp, intraday, swing
    - Expected holding time in minutes
    - Exit urgency score
    - Partial exit recommendations
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        num_duration_classes: int = 3
    ):
        """
        Args:
            embedding_dim: Market embedding dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            dropout: Dropout rate
            num_duration_classes: Number of duration categories
        """
        super().__init__()

        self.num_classes = num_duration_classes

        # Trade state: time_in_trade, unrealized_pnl, pnl_velocity, distance_to_stop, distance_to_target
        trade_state_dim = 5

        # Feature extractor
        layers = []
        in_dim = embedding_dim + trade_state_dim

        for i in range(n_layers):
            out_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Duration classifier
        self.duration_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_duration_classes)
        )

        # Expected holding time regressor (in minutes, log scale)
        self.holding_time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Positive output
        )

        # Exit urgency score (0 = hold, 1 = exit immediately)
        self.exit_urgency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Partial exit fraction (0 to 1)
        self.partial_exit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Exit condition probabilities (momentum_fade, target_reached, time_exit, reversal_signal)
        self.exit_condition_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4)
        )

    def forward(
        self,
        market_embedding: torch.Tensor,
        trade_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for duration prediction.

        Args:
            market_embedding: Market state from Transformer (batch, embedding_dim)
            trade_state: Current trade state (batch, 5) or None if no position
                - time_in_trade: Minutes in current trade (normalized)
                - unrealized_pnl: Current unrealized P&L (normalized)
                - pnl_velocity: Rate of P&L change
                - distance_to_stop: Distance to stop-loss (normalized)
                - distance_to_target: Distance to take-profit (normalized)

        Returns:
            Dict containing duration predictions
        """
        batch_size = market_embedding.size(0)

        # Handle no position case
        if trade_state is None:
            trade_state = torch.zeros(batch_size, 5, device=market_embedding.device)

        # Combine inputs
        x = torch.cat([market_embedding, trade_state], dim=-1)

        # Extract features
        features = self.feature_extractor(x)

        # Duration classification
        duration_logits = self.duration_classifier(features)
        duration_probs = F.softmax(duration_logits, dim=-1)
        duration_class = torch.argmax(duration_probs, dim=-1)

        # Expected holding time (exponential to handle wide range)
        log_holding_time = self.holding_time_head(features)
        holding_time_minutes = torch.exp(log_holding_time)  # Convert from log scale

        # Exit urgency
        exit_urgency = self.exit_urgency_head(features)

        # Partial exit
        partial_exit_fraction = self.partial_exit_head(features)

        # Exit conditions
        exit_condition_logits = self.exit_condition_head(features)
        exit_condition_probs = F.softmax(exit_condition_logits, dim=-1)

        # Adjust recommendations based on trade state
        has_position = (trade_state[:, 0:1] > 0).float()

        return {
            'duration_class': duration_class,
            'duration_probs': duration_probs,
            'holding_time_minutes': holding_time_minutes.squeeze(-1),
            'exit_urgency': exit_urgency.squeeze(-1) * has_position.squeeze(-1),
            'partial_exit_fraction': partial_exit_fraction.squeeze(-1),
            'exit_condition_probs': exit_condition_probs,
            'has_position': has_position.squeeze(-1)
        }

    def get_recommended_action(
        self,
        outputs: Dict[str, torch.Tensor],
        exit_urgency_threshold: float = 0.7,
        partial_exit_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Get concrete trading recommendations from model outputs.

        Args:
            outputs: Output dict from forward()
            exit_urgency_threshold: Threshold for full exit
            partial_exit_threshold: Threshold for partial exit

        Returns:
            Dict with trading recommendations
        """
        exit_urgency = outputs['exit_urgency']
        partial_fraction = outputs['partial_exit_fraction']
        has_position = outputs['has_position']

        # Determine action
        should_exit_full = (exit_urgency > exit_urgency_threshold) & (has_position > 0.5)
        should_exit_partial = (
            (exit_urgency > partial_exit_threshold) &
            (exit_urgency <= exit_urgency_threshold) &
            (has_position > 0.5)
        )
        should_hold = ~should_exit_full & ~should_exit_partial & (has_position > 0.5)

        return {
            'action': torch.where(
                should_exit_full,
                torch.tensor(2),  # Full exit
                torch.where(
                    should_exit_partial,
                    torch.tensor(1),  # Partial exit
                    torch.tensor(0)   # Hold
                )
            ),
            'exit_fraction': torch.where(
                should_exit_full,
                torch.ones_like(partial_fraction),
                torch.where(
                    should_exit_partial,
                    partial_fraction,
                    torch.zeros_like(partial_fraction)
                )
            ),
            'should_exit_full': should_exit_full,
            'should_exit_partial': should_exit_partial,
            'should_hold': should_hold
        }


class TradeTimer:
    """
    Utility class for tracking trade timing.
    """

    def __init__(self):
        self.entry_time = None
        self.entry_price = None
        self.highest_price = None
        self.lowest_price = None
        self.last_price = None
        self.last_pnl = 0.0

    def enter_trade(self, price: float, timestamp):
        """Record trade entry."""
        self.entry_time = timestamp
        self.entry_price = price
        self.highest_price = price
        self.lowest_price = price
        self.last_price = price
        self.last_pnl = 0.0

    def update(self, price: float, timestamp) -> Dict[str, float]:
        """Update trade state with new price."""
        if self.entry_time is None:
            return self._empty_state()

        # Update price extremes
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)

        # Calculate metrics
        time_in_trade = (timestamp - self.entry_time).total_seconds() / 60.0  # Minutes

        is_long = True  # Placeholder - should be passed or stored
        if is_long:
            unrealized_pnl = (price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = (self.entry_price - price) / self.entry_price

        pnl_velocity = unrealized_pnl - self.last_pnl
        self.last_pnl = unrealized_pnl
        self.last_price = price

        return {
            'time_in_trade': time_in_trade,
            'unrealized_pnl': unrealized_pnl,
            'pnl_velocity': pnl_velocity,
            'mae': (self.entry_price - self.lowest_price) / self.entry_price if is_long else
                   (self.highest_price - self.entry_price) / self.entry_price,
            'mfe': (self.highest_price - self.entry_price) / self.entry_price if is_long else
                   (self.entry_price - self.lowest_price) / self.entry_price
        }

    def exit_trade(self):
        """Clear trade state on exit."""
        self.entry_time = None
        self.entry_price = None
        self.highest_price = None
        self.lowest_price = None
        self.last_price = None
        self.last_pnl = 0.0

    def _empty_state(self) -> Dict[str, float]:
        return {
            'time_in_trade': 0.0,
            'unrealized_pnl': 0.0,
            'pnl_velocity': 0.0,
            'mae': 0.0,
            'mfe': 0.0
        }

    def get_state_tensor(self, stop_distance: float, target_distance: float) -> torch.Tensor:
        """Get normalized state tensor for model input."""
        state = self.update(self.last_price, self.entry_time) if self.entry_time else self._empty_state()

        # Normalize time (log scale for wide range)
        time_norm = min(state['time_in_trade'] / 240.0, 1.0)  # Normalize to 4 hours max

        # Distance to stop/target (as fraction of remaining distance)
        if stop_distance > 0:
            dist_to_stop = max(0, 1 - state['mae'] / stop_distance)
        else:
            dist_to_stop = 1.0

        if target_distance > 0:
            dist_to_target = max(0, 1 - state['mfe'] / target_distance)
        else:
            dist_to_target = 1.0

        return torch.tensor([
            time_norm,
            state['unrealized_pnl'],
            state['pnl_velocity'],
            dist_to_stop,
            dist_to_target
        ], dtype=torch.float32)
