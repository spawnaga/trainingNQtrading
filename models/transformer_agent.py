"""
Transformer Agent for temporal pattern recognition in market data.

Uses multi-head self-attention to capture complex temporal dependencies
and generate market state embeddings for downstream agents.
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    Adds position information to the input embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: Tensors of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with self-attention and feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x, attention_weights


class TransformerAgent(nn.Module):
    """
    Transformer-based agent for market pattern recognition.

    Takes a sequence of market features and outputs:
    1. Market state embedding (for other agents)
    2. Attention weights (for interpretability)
    3. Optional direct predictions
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        output_dim: Optional[int] = None
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            output_dim: Optional output dimension for direct predictions
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Pooling for sequence-level embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Optional output head
        self.output_dim = output_dim
        if output_dim is not None:
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Dict containing:
            - embedding: Market state embedding (batch, d_model)
            - sequence: Full sequence output (batch, seq_len, d_model)
            - prediction: Optional direct prediction (batch, output_dim)
            - attention: Optional attention weights list
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add CLS token for pooling
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through encoder layers
        attention_weights_list = []
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, mask)
            if return_attention:
                attention_weights_list.append(attn_weights)

        # Normalize
        x = self.output_norm(x)

        # Extract embedding from CLS token
        embedding = x[:, 0, :]  # (batch, d_model)

        # Full sequence (without CLS token)
        sequence = x[:, 1:, :]  # (batch, seq_len, d_model)

        result = {
            'embedding': embedding,
            'sequence': sequence
        }

        # Optional prediction
        if self.output_dim is not None:
            result['prediction'] = self.output_head(embedding)

        if return_attention:
            result['attention'] = attention_weights_list

        return result

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.d_model


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block for capturing local patterns.
    Used as an optional enhancement to the Transformer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class HybridTransformerAgent(nn.Module):
    """
    Hybrid model combining Temporal Convolutions with Transformer.
    TCN captures local patterns, Transformer captures global dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        tcn_channels: int = 64,
        tcn_layers: int = 3
    ):
        super().__init__()

        # TCN for local pattern extraction
        self.tcn_layers = nn.ModuleList()
        in_ch = input_dim
        for i in range(tcn_layers):
            out_ch = tcn_channels if i < tcn_layers - 1 else d_model
            dilation = 2 ** i
            self.tcn_layers.append(
                TemporalConvBlock(in_ch, out_ch, kernel_size=3, dilation=dilation, dropout=dropout)
            )
            in_ch = out_ch

        # Transformer for global dependencies
        self.transformer = TransformerAgent(
            input_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input of shape (batch, seq_len, input_dim)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)

        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # Through transformer
        return self.transformer(x, mask, return_attention)
