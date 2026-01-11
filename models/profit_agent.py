"""
Profit Maximizer Agent using Policy Gradient methods.

This agent learns optimal position sizing to maximize risk-adjusted returns.
Uses actor-critic architecture for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Optional


class ProfitMaximizer(nn.Module):
    """
    Profit Maximization Agent using Actor-Critic architecture.

    Actor: Outputs position size (-1 to 1)
    Critic: Estimates state value

    The agent learns to maximize Sharpe ratio through policy gradient.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0
    ):
        """
        Args:
            embedding_dim: Dimension of market state embedding from Transformer
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            dropout: Dropout rate
            min_log_std: Minimum log standard deviation
            max_log_std: Maximum log standard deviation
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Shared feature extractor
        layers = []
        in_dim = embedding_dim + 3  # +3 for position state (current_position, unrealized_pnl, time_in_trade)

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

        # Actor head (policy network)
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Linear(hidden_dim, 1)

        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use proper gain for GELU activation (sqrt(2) is good for ReLU-like)
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize actor output layer with smaller weights for stable initial policy
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.1)
        nn.init.orthogonal_(self.actor_log_std.weight, gain=0.1)

    def forward(
        self,
        market_embedding: torch.Tensor,
        position_state: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get action and value.

        Args:
            market_embedding: Market state from Transformer (batch, embedding_dim)
            position_state: Current position info (batch, 3)
                - current_position: -1 to 1
                - unrealized_pnl: normalized P&L
                - time_in_trade: normalized time held
            deterministic: If True, return mean action (for evaluation)

        Returns:
            Dict containing:
            - action: Position size (-1 to 1)
            - log_prob: Log probability of action
            - value: State value estimate
            - mean: Action mean
            - std: Action standard deviation
        """
        # Combine inputs
        x = torch.cat([market_embedding, position_state], dim=-1)

        # Extract features
        features = self.feature_extractor(x)

        # Actor outputs
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()

        # Create distribution
        dist = Normal(mean, std)

        if deterministic:
            action = mean
            log_prob = dist.log_prob(action)
        else:
            action = dist.rsample()  # Reparameterization trick
            log_prob = dist.log_prob(action)

        # Squash to [-1, 1] using tanh
        action_squashed = torch.tanh(action)

        # Adjust log_prob for tanh squashing
        log_prob = log_prob - torch.log(1 - action_squashed.pow(2) + 1e-6)

        # Critic output
        value = self.critic(features)

        return {
            'action': action_squashed.squeeze(-1),
            'log_prob': log_prob.squeeze(-1),
            'value': value.squeeze(-1),
            'mean': mean.squeeze(-1),
            'std': std.squeeze(-1),
            'entropy': dist.entropy().squeeze(-1)
        }

    def get_value(
        self,
        market_embedding: torch.Tensor,
        position_state: torch.Tensor
    ) -> torch.Tensor:
        """Get state value only (for training)."""
        x = torch.cat([market_embedding, position_state], dim=-1)
        features = self.feature_extractor(x)
        return self.critic(features).squeeze(-1)

    def evaluate_actions(
        self,
        market_embedding: torch.Tensor,
        position_state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of given actions.
        Used for PPO training.

        Returns:
            (log_probs, entropy, values)
        """
        x = torch.cat([market_embedding, position_state], dim=-1)
        features = self.feature_extractor(x)

        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()

        dist = Normal(mean, std)

        # Inverse tanh to get unsquashed action
        actions_unsquashed = torch.atanh(actions.clamp(-0.999, 0.999)).unsqueeze(-1)

        log_prob = dist.log_prob(actions_unsquashed)
        log_prob = log_prob - torch.log(1 - actions.unsqueeze(-1).pow(2) + 1e-6)

        value = self.critic(features)
        entropy = dist.entropy()

        return log_prob.squeeze(-1), entropy.squeeze(-1), value.squeeze(-1)


class PPOBuffer:
    """
    Buffer for storing trajectories for PPO training.
    """

    def __init__(
        self,
        capacity: int,
        embedding_dim: int,
        gamma: float = 0.99,
        lam: float = 0.95
    ):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam

        self.embeddings = torch.zeros(capacity, embedding_dim)
        self.position_states = torch.zeros(capacity, 3)
        self.actions = torch.zeros(capacity)
        self.rewards = torch.zeros(capacity)
        self.values = torch.zeros(capacity)
        self.log_probs = torch.zeros(capacity)
        self.dones = torch.zeros(capacity)

        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def store(
        self,
        embedding: torch.Tensor,
        position_state: torch.Tensor,
        action: float,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store a single transition."""
        idx = self.ptr % self.capacity

        self.embeddings[idx] = embedding.detach()
        self.position_states[idx] = position_state.detach()
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = float(done)

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float):
        """
        Compute GAE-Lambda advantages and returns.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        # Append last value for bootstrapping
        values_extended = torch.cat([values, torch.tensor([last_value])])

        # GAE-Lambda calculation
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_extended[t + 1] * next_non_terminal - values_extended[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae

        returns = advantages + values

        # Store computed values
        self.advantages = advantages
        self.returns = returns

        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all stored data."""
        actual_size = min(self.ptr, self.capacity)
        return {
            'embeddings': self.embeddings[:actual_size],
            'position_states': self.position_states[:actual_size],
            'actions': self.actions[:actual_size],
            'returns': self.returns[:actual_size] if hasattr(self, 'returns') else None,
            'advantages': self.advantages[:actual_size] if hasattr(self, 'advantages') else None,
            'log_probs': self.log_probs[:actual_size]
        }

    def clear(self):
        """Reset buffer."""
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False


def compute_sharpe_reward(
    returns: torch.Tensor,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0
) -> torch.Tensor:
    """
    Compute Sharpe ratio-based reward.

    Args:
        returns: Tensor of returns
        risk_free_rate: Risk-free rate
        annualization: Annualization factor

    Returns:
        Sharpe-based reward
    """
    if len(returns) < 2:
        return torch.tensor(0.0)

    excess_returns = returns - risk_free_rate / annualization
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()

    if std_return < 1e-8:
        return torch.tensor(0.0)

    sharpe = mean_return / std_return * (annualization ** 0.5)
    return sharpe
