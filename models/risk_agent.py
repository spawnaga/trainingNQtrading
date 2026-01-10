"""
Risk Controller Agent for position and portfolio risk management.

Combines ATR-based dynamic stops with fixed percentage limits.
Outputs risk multipliers and stop-loss distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class RiskController(nn.Module):
    """
    Risk Control Agent that manages position sizing and stop-losses.

    Outputs:
    - Risk multiplier (0 to 1): Scales down position size based on risk
    - Stop-loss distance: ATR-based dynamic stop level
    - Take-profit distance: Reward-to-risk based target
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_risk_per_trade: float = 0.02,  # 2% max risk per trade
        max_daily_drawdown: float = 0.05,  # 5% max daily drawdown
        default_atr_multiplier: float = 2.0,
        min_reward_risk_ratio: float = 1.5
    ):
        """
        Args:
            embedding_dim: Market embedding dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            dropout: Dropout rate
            max_risk_per_trade: Maximum risk per trade as fraction of account
            max_daily_drawdown: Maximum daily drawdown allowed
            default_atr_multiplier: Default ATR multiplier for stops
            min_reward_risk_ratio: Minimum reward-to-risk ratio
        """
        super().__init__()

        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.default_atr_multiplier = default_atr_multiplier
        self.min_reward_risk_ratio = min_reward_risk_ratio

        # Risk state includes: current_drawdown, daily_pnl, volatility, position_exposure
        risk_state_dim = 4

        # Feature extractor
        layers = []
        in_dim = embedding_dim + risk_state_dim

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

        # Output heads
        # Risk multiplier: 0 to 1 (sigmoid output)
        self.risk_multiplier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # ATR multiplier: 1 to 4 (for stop distance)
        self.atr_multiplier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive output
        )

        # Reward-risk ratio: 1 to 5
        self.rr_ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        # Risk regime classifier: low, medium, high
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(
        self,
        market_embedding: torch.Tensor,
        risk_state: torch.Tensor,
        atr: torch.Tensor,
        current_price: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get risk parameters.

        Args:
            market_embedding: Market state from Transformer (batch, embedding_dim)
            risk_state: Current risk metrics (batch, 4)
                - current_drawdown: Current drawdown from peak
                - daily_pnl: Today's P&L as fraction of account
                - volatility: Current market volatility (normalized)
                - position_exposure: Current position exposure
            atr: Average True Range value (batch, 1)
            current_price: Current price (batch, 1)

        Returns:
            Dict containing risk parameters
        """
        batch_size = market_embedding.size(0)

        # Combine inputs
        x = torch.cat([market_embedding, risk_state], dim=-1)

        # Extract features
        features = self.feature_extractor(x)

        # Get raw outputs
        raw_risk_mult = self.risk_multiplier_head(features)
        raw_atr_mult = self.atr_multiplier_head(features) + 1.0  # Minimum 1.0
        raw_rr_ratio = self.rr_ratio_head(features) + self.min_reward_risk_ratio

        # Risk regime
        regime_logits = self.regime_classifier(features)
        regime_probs = F.softmax(regime_logits, dim=-1)
        regime = torch.argmax(regime_probs, dim=-1)

        # Apply drawdown-based override
        current_drawdown = risk_state[:, 0:1]
        daily_pnl = risk_state[:, 1:2]

        # Reduce risk as drawdown increases
        drawdown_factor = torch.clamp(1.0 - (current_drawdown / self.max_daily_drawdown), 0.0, 1.0)

        # If daily loss limit hit, set risk to 0
        daily_loss_exceeded = (daily_pnl < -self.max_daily_drawdown).float()
        risk_multiplier = raw_risk_mult * drawdown_factor * (1.0 - daily_loss_exceeded)

        # Calculate stop-loss and take-profit distances
        atr_multiplier = torch.clamp(raw_atr_mult, 1.0, 4.0)
        stop_distance = atr * atr_multiplier  # In price units

        rr_ratio = torch.clamp(raw_rr_ratio, self.min_reward_risk_ratio, 5.0)
        target_distance = stop_distance * rr_ratio

        # Calculate position size based on risk
        # Position size = (Account * Risk%) / Stop distance
        # Returns as a fraction of maximum position
        risk_per_contract = stop_distance / current_price
        max_position_fraction = self.max_risk_per_trade / (risk_per_contract + 1e-8)
        position_size_limit = torch.clamp(max_position_fraction, 0.0, 1.0)

        # Final position limit considering all factors
        final_position_limit = position_size_limit * risk_multiplier

        return {
            'risk_multiplier': risk_multiplier.squeeze(-1),
            'atr_multiplier': atr_multiplier.squeeze(-1),
            'stop_distance': stop_distance.squeeze(-1),
            'target_distance': target_distance.squeeze(-1),
            'rr_ratio': rr_ratio.squeeze(-1),
            'position_limit': final_position_limit.squeeze(-1),
            'regime': regime,
            'regime_probs': regime_probs,
            'drawdown_factor': drawdown_factor.squeeze(-1)
        }

    def calculate_stop_price(
        self,
        entry_price: torch.Tensor,
        stop_distance: torch.Tensor,
        is_long: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate stop-loss price.

        Args:
            entry_price: Entry price
            stop_distance: Stop distance in price units
            is_long: Boolean tensor indicating long positions

        Returns:
            Stop price
        """
        is_long_float = is_long.float()
        stop_price = entry_price - stop_distance * (2 * is_long_float - 1)
        return stop_price

    def calculate_target_price(
        self,
        entry_price: torch.Tensor,
        target_distance: torch.Tensor,
        is_long: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate take-profit price.

        Args:
            entry_price: Entry price
            target_distance: Target distance in price units
            is_long: Boolean tensor indicating long positions

        Returns:
            Target price
        """
        is_long_float = is_long.float()
        target_price = entry_price + target_distance * (2 * is_long_float - 1)
        return target_price


class PositionSizer:
    """
    Utility class for position sizing calculations.
    """

    def __init__(
        self,
        account_size: float,
        max_risk_per_trade: float = 0.02,
        max_position: int = 2,
        point_value: float = 20.0  # NQ point value
    ):
        """
        Args:
            account_size: Total account value
            max_risk_per_trade: Maximum risk per trade (fraction)
            max_position: Maximum contracts allowed
            point_value: Dollar value per point
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position = max_position
        self.point_value = point_value

    def calculate_contracts(
        self,
        stop_distance: float,
        risk_multiplier: float = 1.0
    ) -> int:
        """
        Calculate number of contracts based on risk.

        Args:
            stop_distance: Stop distance in points
            risk_multiplier: Risk adjustment factor (0-1)

        Returns:
            Number of contracts
        """
        risk_amount = self.account_size * self.max_risk_per_trade * risk_multiplier
        risk_per_contract = stop_distance * self.point_value

        if risk_per_contract <= 0:
            return 0

        contracts = int(risk_amount / risk_per_contract)
        return min(contracts, self.max_position)

    def update_account_size(self, new_size: float):
        """Update account size after P&L."""
        self.account_size = new_size


class DrawdownMonitor:
    """
    Monitor and track drawdown metrics.
    """

    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.daily_start_equity = initial_equity

        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0

    def update(self, new_equity: float):
        """Update with new equity value."""
        self.current_equity = new_equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

        # Calculate current drawdown
        self.current_drawdown = (self.peak_equity - new_equity) / self.peak_equity

        # Update max drawdown
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

        # Daily P&L
        self.daily_pnl = (new_equity - self.daily_start_equity) / self.daily_start_equity

    def new_day(self):
        """Reset for new trading day."""
        self.daily_start_equity = self.current_equity
        self.daily_pnl = 0.0

    def get_risk_state(self) -> torch.Tensor:
        """Get risk state tensor for model input."""
        return torch.tensor([
            self.current_drawdown,
            self.daily_pnl,
            0.0,  # Placeholder for volatility (set externally)
            0.0   # Placeholder for position exposure
        ], dtype=torch.float32)
