"""
Meta Agent - The orchestrator that combines all sub-agent outputs
to make final trading decisions.

Uses attention-based weighting to dynamically combine agent recommendations
based on market conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .transformer_agent import TransformerAgent
from .profit_agent import ProfitMaximizer
from .risk_agent import RiskController
from .duration_agent import TradeDurationAgent


@dataclass
class TradingDecision:
    """Container for final trading decision."""
    action: int              # 0=hold, 1=buy, 2=sell, 3=close
    position_size: float     # -1 to 1 (negative for short)
    stop_loss: float         # Stop-loss price
    take_profit: float       # Take-profit price
    confidence: float        # Decision confidence
    duration_class: int      # Expected trade duration
    risk_multiplier: float   # Risk adjustment factor
    reasoning: Dict          # Sub-agent outputs for explainability


class MetaAgent(nn.Module):
    """
    Meta Agent that orchestrates all sub-agents.

    Combines outputs from:
    - Transformer Agent (market embedding)
    - Profit Maximizer (position sizing)
    - Risk Controller (risk parameters)
    - Duration Agent (holding period)

    Uses attention mechanism to weight agent contributions based on context.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        # Transformer config
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ff: int = 1024,
        # Sub-agent configs
        profit_hidden: int = 128,
        risk_hidden: int = 64,
        duration_hidden: int = 64
    ):
        """
        Args:
            input_dim: Number of input features
            embedding_dim: Embedding dimension
            n_heads: Attention heads for meta-agent
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Initialize sub-agents
        self.transformer_agent = TransformerAgent(
            input_dim=input_dim,
            d_model=embedding_dim,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            d_ff=transformer_ff,
            dropout=dropout
        )

        self.profit_agent = ProfitMaximizer(
            embedding_dim=embedding_dim,
            hidden_dim=profit_hidden,
            dropout=dropout
        )

        self.risk_agent = RiskController(
            embedding_dim=embedding_dim,
            hidden_dim=risk_hidden,
            dropout=dropout
        )

        self.duration_agent = TradeDurationAgent(
            embedding_dim=embedding_dim,
            hidden_dim=duration_hidden,
            dropout=dropout
        )

        # Agent output dimensions
        # Profit: action(1), value(1), confidence from std(1) = 3
        # Risk: risk_mult(1), stop_dist(1), target_dist(1), regime(3) = 6
        # Duration: class_probs(3), urgency(1), partial(1) = 5
        agent_output_dim = 3 + 6 + 5  # = 14

        # Attention-based agent weighting
        self.agent_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Project agent outputs to embedding space
        self.profit_proj = nn.Linear(3, embedding_dim)
        self.risk_proj = nn.Linear(6, embedding_dim)
        self.duration_proj = nn.Linear(5, embedding_dim)

        # Final decision layers
        self.decision_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),  # 4 = market + 3 agents
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Output heads
        # Action classifier: hold, buy, sell, close
        self.action_head = nn.Linear(embedding_dim // 2, 4)

        # Position size (continuous)
        self.position_head = nn.Sequential(
            nn.Linear(embedding_dim // 2, 1),
            nn.Tanh()
        )

        # Confidence score
        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        features: torch.Tensor,
        position_state: torch.Tensor,
        risk_state: torch.Tensor,
        trade_state: Optional[torch.Tensor],
        atr: torch.Tensor,
        current_price: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all agents.

        Args:
            features: Market features (batch, seq_len, input_dim)
            position_state: Position info for profit agent (batch, 3)
            risk_state: Risk metrics for risk agent (batch, 4)
            trade_state: Trade state for duration agent (batch, 5) or None
            atr: ATR values (batch, 1)
            current_price: Current prices (batch, 1)
            deterministic: Use deterministic policy

        Returns:
            Dict with all agent outputs and final decision
        """
        batch_size = features.size(0)

        # 1. Transformer Agent - Get market embedding
        transformer_out = self.transformer_agent(features)
        market_embedding = transformer_out['embedding']

        # 2. Profit Agent - Get position sizing
        profit_out = self.profit_agent(
            market_embedding,
            position_state,
            deterministic=deterministic
        )

        # 3. Risk Agent - Get risk parameters
        risk_out = self.risk_agent(
            market_embedding,
            risk_state,
            atr,
            current_price
        )

        # 4. Duration Agent - Get holding period
        duration_out = self.duration_agent(
            market_embedding,
            trade_state
        )

        # Prepare agent outputs for attention
        profit_features = torch.stack([
            profit_out['action'],
            profit_out['value'],
            1.0 / (profit_out['std'] + 0.1)  # Confidence from inverse std
        ], dim=-1)

        risk_features = torch.stack([
            risk_out['risk_multiplier'],
            risk_out['stop_distance'] / (current_price.squeeze(-1) + 1e-6),
            risk_out['target_distance'] / (current_price.squeeze(-1) + 1e-6),
            risk_out['regime_probs'][:, 0],
            risk_out['regime_probs'][:, 1],
            risk_out['regime_probs'][:, 2]
        ], dim=-1)

        duration_features = torch.stack([
            duration_out['duration_probs'][:, 0],
            duration_out['duration_probs'][:, 1],
            duration_out['duration_probs'][:, 2],
            duration_out['exit_urgency'],
            duration_out['partial_exit_fraction']
        ], dim=-1)

        # Project to embedding space
        profit_emb = self.profit_proj(profit_features)
        risk_emb = self.risk_proj(risk_features)
        duration_emb = self.duration_proj(duration_features)

        # Stack agent embeddings for attention
        agent_embeddings = torch.stack([
            market_embedding,
            profit_emb,
            risk_emb,
            duration_emb
        ], dim=1)  # (batch, 4, embedding_dim)

        # Self-attention over agent embeddings
        attended, attention_weights = self.agent_attention(
            agent_embeddings,
            agent_embeddings,
            agent_embeddings
        )

        # Combine attended embeddings
        combined = attended.reshape(batch_size, -1)

        # Decision network
        decision_features = self.decision_net(combined)

        # Output heads
        action_logits = self.action_head(decision_features)
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)

        raw_position = self.position_head(decision_features)

        confidence = self.confidence_head(decision_features)

        # Apply risk multiplier to position size
        position_size = raw_position.squeeze(-1) * risk_out['risk_multiplier']

        # Calculate stop and target prices
        is_long = position_size > 0
        stop_price = self.risk_agent.calculate_stop_price(
            current_price.squeeze(-1),
            risk_out['stop_distance'],
            is_long
        )
        target_price = self.risk_agent.calculate_target_price(
            current_price.squeeze(-1),
            risk_out['target_distance'],
            is_long
        )

        return {
            # Final decisions
            'action': action,
            'action_probs': action_probs,
            'position_size': position_size,
            'stop_price': stop_price,
            'target_price': target_price,
            'confidence': confidence.squeeze(-1),

            # Agent outputs for explainability
            'market_embedding': market_embedding,
            'profit_output': profit_out,
            'risk_output': risk_out,
            'duration_output': duration_out,

            # Attention weights for interpretability
            'agent_attention_weights': attention_weights
        }

    def get_trading_decision(
        self,
        features: torch.Tensor,
        position_state: torch.Tensor,
        risk_state: torch.Tensor,
        trade_state: Optional[torch.Tensor],
        atr: torch.Tensor,
        current_price: torch.Tensor,
        current_position: float = 0.0,
        confidence_threshold: float = 0.6
    ) -> TradingDecision:
        """
        Get a concrete trading decision.

        Args:
            features: Market features
            position_state: Position state
            risk_state: Risk state
            trade_state: Trade state
            atr: ATR value
            current_price: Current price
            current_position: Current position (-1 to 1)
            confidence_threshold: Minimum confidence for action

        Returns:
            TradingDecision object
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                features.unsqueeze(0) if features.dim() == 2 else features,
                position_state.unsqueeze(0) if position_state.dim() == 1 else position_state,
                risk_state.unsqueeze(0) if risk_state.dim() == 1 else risk_state,
                trade_state.unsqueeze(0) if trade_state is not None and trade_state.dim() == 1 else trade_state,
                atr.unsqueeze(0) if atr.dim() == 0 else atr,
                current_price.unsqueeze(0) if current_price.dim() == 0 else current_price,
                deterministic=True
            )

            action = outputs['action'].item()
            position_size = outputs['position_size'].item()
            confidence = outputs['confidence'].item()
            stop_price = outputs['stop_price'].item()
            target_price = outputs['target_price'].item()

            # Override action if confidence is too low
            if confidence < confidence_threshold:
                action = 0  # Hold

            # Adjust action based on current position
            if current_position != 0:
                # If we have a position, check if we should exit
                exit_urgency = outputs['duration_output']['exit_urgency'].item()
                if exit_urgency > 0.7:
                    action = 3  # Close position

            return TradingDecision(
                action=action,
                position_size=position_size,
                stop_loss=stop_price,
                take_profit=target_price,
                confidence=confidence,
                duration_class=outputs['duration_output']['duration_class'].item(),
                risk_multiplier=outputs['risk_output']['risk_multiplier'].item(),
                reasoning={
                    'action_probs': outputs['action_probs'].squeeze().tolist(),
                    'profit_action': outputs['profit_output']['action'].item(),
                    'risk_regime': outputs['risk_output']['regime'].item(),
                    'exit_urgency': outputs['duration_output']['exit_urgency'].item()
                }
            )


class EnsembleMetaAgent(nn.Module):
    """
    Ensemble of multiple MetaAgents for more robust predictions.
    """

    def __init__(
        self,
        n_models: int,
        input_dim: int,
        **kwargs
    ):
        super().__init__()

        self.n_models = n_models
        self.models = nn.ModuleList([
            MetaAgent(input_dim, **kwargs)
            for _ in range(n_models)
        ])

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward through all models and aggregate."""
        all_outputs = [model(*args, **kwargs) for model in self.models]

        # Aggregate action probabilities
        action_probs = torch.stack([o['action_probs'] for o in all_outputs]).mean(dim=0)
        action = torch.argmax(action_probs, dim=-1)

        # Aggregate position sizes
        position_size = torch.stack([o['position_size'] for o in all_outputs]).mean(dim=0)

        # Aggregate confidence
        confidence = torch.stack([o['confidence'] for o in all_outputs]).mean(dim=0)

        # Use disagreement as uncertainty measure
        position_std = torch.stack([o['position_size'] for o in all_outputs]).std(dim=0)
        uncertainty = position_std

        return {
            'action': action,
            'action_probs': action_probs,
            'position_size': position_size,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'stop_price': all_outputs[0]['stop_price'],  # Use first model's stops
            'target_price': all_outputs[0]['target_price'],
            'individual_outputs': all_outputs
        }
