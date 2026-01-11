"""
Simplified Meta Agent for Training Validation
==============================================
Direct transformer -> position prediction without complex sub-agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SimpleMetaAgent(nn.Module):
    """
    Simplified trading agent that directly predicts position from features.
    Uses transformer encoder + MLP head without complex sub-agent architecture.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, embedding_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, embedding_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Layer norm after transformer
        self.post_norm = nn.LayerNorm(embedding_dim)

        # Position prediction head - SIMPLE and direct
        self.position_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.GELU(),
            nn.Linear(embedding_dim // 4, 1)
        )

        # Action head
        self.action_head = nn.Linear(embedding_dim, 4)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper gains."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Smaller init for output layers
        for module in self.position_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)

    def forward(
        self,
        features: torch.Tensor,
        position_state: torch.Tensor = None,  # Ignored
        risk_state: torch.Tensor = None,      # Ignored
        trade_state: torch.Tensor = None,     # Ignored
        atr: torch.Tensor = None,             # Ignored
        current_price: torch.Tensor = None,   # Ignored
        deterministic: bool = False           # Ignored
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Market features (batch, seq_len, input_dim)
            Other args: Ignored, for API compatibility with MetaAgent

        Returns:
            Dict with position_size and other outputs
        """
        batch_size, seq_len, _ = features.shape

        # Project input
        x = self.input_proj(features)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)
        x = self.post_norm(x)

        # Use last position embedding
        embedding = x[:, -1, :]

        # Predict position directly
        raw_position = self.position_head(embedding)
        position_size = torch.tanh(raw_position).squeeze(-1)

        # Action prediction
        action_logits = self.action_head(embedding)
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)

        # Confidence
        confidence = self.confidence_head(embedding).squeeze(-1)

        return {
            'position_size': position_size,
            'action': action,
            'action_probs': action_probs,
            'confidence': confidence,
            'market_embedding': embedding,
            # Dummy outputs for compatibility
            'profit_output': {'action': position_size, 'value': torch.zeros_like(position_size), 'std': torch.ones_like(position_size)},
            'risk_output': {'risk_multiplier': torch.ones_like(position_size), 'stop_distance': torch.zeros_like(position_size), 'target_distance': torch.zeros_like(position_size), 'regime_probs': torch.zeros(batch_size, 3, device=features.device), 'regime': torch.zeros(batch_size, device=features.device, dtype=torch.long)},
            'duration_output': {'duration_probs': torch.zeros(batch_size, 3, device=features.device), 'duration_class': torch.zeros(batch_size, device=features.device, dtype=torch.long), 'exit_urgency': torch.zeros(batch_size, device=features.device), 'partial_exit_fraction': torch.zeros(batch_size, device=features.device)},
            'agent_attention_weights': None,
            'stop_price': torch.zeros_like(position_size),
            'target_price': torch.zeros_like(position_size)
        }
