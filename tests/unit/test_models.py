"""
Unit tests for models module.

Tests:
- TransformerAgent
- ProfitMaximizer
- RiskController
- TradeDurationAgent
- MetaAgent
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.transformer_agent import (
    TransformerAgent,
    HybridTransformerAgent,
    PositionalEncoding,
    MultiHeadAttention
)
from models.profit_agent import ProfitMaximizer, PPOBuffer, compute_sharpe_reward
from models.risk_agent import RiskController, PositionSizer, DrawdownMonitor
from models.duration_agent import TradeDurationAgent, TradeDuration, TradeTimer
from models.meta_agent import MetaAgent, TradingDecision, EnsembleMetaAgent


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        """Test output shape matches input."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(4, 50, 64)
        result = pe(x)

        assert result.shape == x.shape

    def test_encoding_values(self):
        """Test that encoding values are bounded."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 50, 64)
        result = pe(x)

        # Sine/cosine values should be in [-1, 1]
        assert result.abs().max() <= 1.0


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    def test_output_shape(self):
        """Test output shape."""
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = torch.randn(4, 30, 64)

        output, attn_weights = mha(x, x, x)

        assert output.shape == x.shape
        assert attn_weights.shape == (4, 4, 30, 30)  # (batch, heads, seq, seq)

    def test_masked_attention(self):
        """Test attention with mask."""
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 4, 10, 10)
        mask[:, :, :, 5:] = 0  # Mask out second half

        output, attn_weights = mha(x, x, x, mask)

        assert output.shape == x.shape


class TestTransformerAgent:
    """Tests for TransformerAgent."""

    def test_initialization(self, config_dict):
        """Test model initialization."""
        cfg = config_dict['model']['transformer']
        model = TransformerAgent(
            input_dim=44,
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers']
        )

        assert model.d_model == cfg['d_model']
        assert model.n_heads == cfg['n_heads']

    def test_forward(self, transformer_agent, sample_features):
        """Test forward pass."""
        outputs = transformer_agent(sample_features)

        assert 'embedding' in outputs
        assert 'sequence' in outputs
        assert outputs['embedding'].shape == (4, 64)  # (batch, d_model)

    def test_forward_with_attention(self, transformer_agent, sample_features):
        """Test forward with attention weights."""
        outputs = transformer_agent(sample_features, return_attention=True)

        assert 'attention' in outputs
        assert len(outputs['attention']) == 2  # n_layers

    def test_no_nan_output(self, transformer_agent, sample_features):
        """Test that outputs don't contain NaN."""
        outputs = transformer_agent(sample_features)

        pytest.assert_no_nan(outputs['embedding'])
        pytest.assert_no_nan(outputs['sequence'])

    def test_embedding_dim(self, transformer_agent):
        """Test embedding dimension getter."""
        assert transformer_agent.get_embedding_dim() == 64


class TestHybridTransformerAgent:
    """Tests for HybridTransformerAgent."""

    def test_forward(self, sample_features):
        """Test hybrid model forward pass."""
        model = HybridTransformerAgent(
            input_dim=44,
            d_model=64,
            n_heads=4,
            n_layers=2,
            tcn_channels=32,
            tcn_layers=2
        )

        outputs = model(sample_features)

        assert 'embedding' in outputs
        assert outputs['embedding'].shape == (4, 64)


class TestProfitMaximizer:
    """Tests for ProfitMaximizer."""

    def test_initialization(self, config_dict):
        """Test model initialization."""
        model = ProfitMaximizer(
            embedding_dim=64,
            hidden_dim=config_dict['model']['profit_agent']['hidden_dim']
        )

        assert model.embedding_dim == 64

    def test_forward(self, profit_agent):
        """Test forward pass."""
        embedding = torch.randn(4, 64)
        position_state = torch.zeros(4, 3)

        outputs = profit_agent(embedding, position_state)

        assert 'action' in outputs
        assert 'value' in outputs
        assert 'log_prob' in outputs
        assert outputs['action'].shape == (4,)

    def test_action_bounds(self, profit_agent):
        """Test that actions are bounded [-1, 1]."""
        embedding = torch.randn(4, 64)
        position_state = torch.zeros(4, 3)

        outputs = profit_agent(embedding, position_state)

        assert outputs['action'].min() >= -1.0
        assert outputs['action'].max() <= 1.0

    def test_deterministic_mode(self, profit_agent):
        """Test deterministic action selection."""
        profit_agent.eval()  # Set to eval mode to disable dropout
        embedding = torch.randn(4, 64)
        position_state = torch.zeros(4, 3)

        # Deterministic should give same action for same input
        with torch.no_grad():
            outputs1 = profit_agent(embedding, position_state, deterministic=True)
            outputs2 = profit_agent(embedding, position_state, deterministic=True)

        assert torch.allclose(outputs1['action'], outputs2['action'])

    def test_evaluate_actions(self, profit_agent):
        """Test action evaluation."""
        embedding = torch.randn(4, 64)
        position_state = torch.zeros(4, 3)
        actions = torch.randn(4).clamp(-0.99, 0.99)

        log_probs, entropy, values = profit_agent.evaluate_actions(
            embedding, position_state, actions
        )

        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert values.shape == (4,)


class TestPPOBuffer:
    """Tests for PPO experience buffer."""

    def test_store_and_get(self):
        """Test storing and retrieving data."""
        buffer = PPOBuffer(capacity=100, embedding_dim=64)

        for i in range(10):
            buffer.store(
                embedding=torch.randn(64),
                position_state=torch.zeros(3),
                action=0.5,
                reward=1.0,
                value=0.9,
                log_prob=-0.5,
                done=False
            )

        assert buffer.ptr == 10

        data = buffer.get()
        assert data['embeddings'].shape == (10, 64)

    def test_clear(self):
        """Test buffer clearing."""
        buffer = PPOBuffer(capacity=100, embedding_dim=64)

        buffer.store(torch.randn(64), torch.zeros(3), 0.5, 1.0, 0.9, -0.5, False)
        buffer.clear()

        assert buffer.ptr == 0


class TestComputeSharpeReward:
    """Tests for Sharpe reward calculation."""

    def test_positive_sharpe(self):
        """Test positive Sharpe calculation."""
        returns = torch.tensor([0.01, 0.02, 0.015, 0.01, 0.02])
        sharpe = compute_sharpe_reward(returns)

        assert sharpe > 0

    def test_negative_returns(self):
        """Test negative returns give negative Sharpe."""
        returns = torch.tensor([-0.01, -0.02, -0.015, -0.01, -0.02])
        sharpe = compute_sharpe_reward(returns)

        assert sharpe < 0


class TestRiskController:
    """Tests for RiskController."""

    def test_initialization(self, config_dict):
        """Test model initialization."""
        model = RiskController(
            embedding_dim=64,
            hidden_dim=config_dict['model']['risk_agent']['hidden_dim']
        )

        assert model.max_risk_per_trade == 0.02
        assert model.max_daily_drawdown == 0.05

    def test_forward(self, risk_agent):
        """Test forward pass."""
        embedding = torch.randn(4, 64)
        risk_state = torch.zeros(4, 4)
        atr = torch.ones(4, 1) * 50.0
        price = torch.ones(4, 1) * 15000.0

        outputs = risk_agent(embedding, risk_state, atr, price)

        assert 'risk_multiplier' in outputs
        assert 'stop_distance' in outputs
        assert 'target_distance' in outputs
        assert 'position_limit' in outputs

    def test_risk_bounds(self, risk_agent):
        """Test risk multiplier bounds."""
        embedding = torch.randn(4, 64)
        risk_state = torch.zeros(4, 4)
        atr = torch.ones(4, 1) * 50.0
        price = torch.ones(4, 1) * 15000.0

        outputs = risk_agent(embedding, risk_state, atr, price)

        assert outputs['risk_multiplier'].min() >= 0.0
        assert outputs['risk_multiplier'].max() <= 1.0

    def test_high_drawdown_reduces_risk(self, risk_agent):
        """Test that high drawdown reduces risk."""
        embedding = torch.randn(4, 64)
        atr = torch.ones(4, 1) * 50.0
        price = torch.ones(4, 1) * 15000.0

        # Low drawdown
        low_dd_state = torch.tensor([[0.01, 0.0, 0.5, 0.0]] * 4)
        low_dd_output = risk_agent(embedding, low_dd_state, atr, price)

        # High drawdown
        high_dd_state = torch.tensor([[0.04, 0.0, 0.5, 0.0]] * 4)
        high_dd_output = risk_agent(embedding, high_dd_state, atr, price)

        assert high_dd_output['risk_multiplier'].mean() < low_dd_output['risk_multiplier'].mean()


class TestPositionSizer:
    """Tests for PositionSizer utility."""

    def test_calculate_contracts(self):
        """Test contract calculation."""
        sizer = PositionSizer(
            account_size=100000,
            max_risk_per_trade=0.02,
            max_position=2,
            point_value=20.0
        )

        # 2% of 100k = $2000 risk
        # 50 point stop * $20 = $1000 per contract
        # Should allow 2 contracts
        contracts = sizer.calculate_contracts(stop_distance=50.0)
        assert contracts == 2

    def test_max_position_limit(self):
        """Test max position is enforced."""
        sizer = PositionSizer(
            account_size=1000000,
            max_risk_per_trade=0.02,
            max_position=2
        )

        contracts = sizer.calculate_contracts(stop_distance=10.0)
        assert contracts <= 2


class TestDrawdownMonitor:
    """Tests for DrawdownMonitor."""

    def test_update(self):
        """Test drawdown calculation."""
        monitor = DrawdownMonitor(initial_equity=100000)

        monitor.update(105000)  # New peak
        assert monitor.current_drawdown == 0.0
        assert monitor.peak_equity == 105000

        monitor.update(100000)  # Drawdown
        assert monitor.current_drawdown == pytest.approx(0.0476, rel=0.01)

    def test_max_drawdown_tracking(self):
        """Test max drawdown is tracked."""
        monitor = DrawdownMonitor(initial_equity=100000)

        monitor.update(110000)
        monitor.update(90000)  # 18.18% drawdown
        monitor.update(100000)

        assert monitor.max_drawdown == pytest.approx(0.1818, rel=0.01)


class TestTradeDurationAgent:
    """Tests for TradeDurationAgent."""

    def test_forward(self, duration_agent):
        """Test forward pass."""
        embedding = torch.randn(4, 64)
        trade_state = torch.zeros(4, 5)

        outputs = duration_agent(embedding, trade_state)

        assert 'duration_class' in outputs
        assert 'holding_time_minutes' in outputs
        assert 'exit_urgency' in outputs

    def test_duration_classes(self, duration_agent):
        """Test duration classification."""
        embedding = torch.randn(4, 64)

        outputs = duration_agent(embedding, None)

        assert outputs['duration_class'].min() >= 0
        assert outputs['duration_class'].max() <= 2  # 3 classes

    def test_exit_urgency_bounds(self, duration_agent):
        """Test exit urgency is bounded [0, 1]."""
        embedding = torch.randn(4, 64)
        trade_state = torch.randn(4, 5)
        trade_state[:, 0] = 1.0  # Has position

        outputs = duration_agent(embedding, trade_state)

        assert outputs['exit_urgency'].min() >= 0.0
        assert outputs['exit_urgency'].max() <= 1.0


class TestMetaAgent:
    """Tests for MetaAgent."""

    def test_initialization(self, config_dict):
        """Test model initialization."""
        cfg = config_dict['model']
        model = MetaAgent(
            input_dim=44,
            embedding_dim=cfg['transformer']['d_model'],
            n_heads=4,
            transformer_layers=cfg['transformer']['n_layers'],
            transformer_heads=cfg['transformer']['n_heads'],
            transformer_ff=cfg['transformer']['d_ff']
        )

        assert model.embedding_dim == cfg['transformer']['d_model']

    def test_forward(self, meta_agent, sample_features):
        """Test full forward pass."""
        batch_size = sample_features.shape[0]

        position_state = torch.zeros(batch_size, 3)
        risk_state = torch.zeros(batch_size, 4)
        trade_state = torch.zeros(batch_size, 5)
        atr = torch.ones(batch_size, 1) * 50.0
        price = torch.ones(batch_size, 1) * 15000.0

        outputs = meta_agent(
            sample_features,
            position_state,
            risk_state,
            trade_state,
            atr,
            price
        )

        assert 'action' in outputs
        assert 'position_size' in outputs
        assert 'stop_price' in outputs
        assert 'target_price' in outputs
        assert 'confidence' in outputs

    def test_action_values(self, meta_agent, sample_features):
        """Test action output values."""
        batch_size = sample_features.shape[0]

        outputs = meta_agent(
            sample_features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50.0,
            torch.ones(batch_size, 1) * 15000.0
        )

        # Action should be 0, 1, 2, or 3
        assert outputs['action'].min() >= 0
        assert outputs['action'].max() <= 3

    def test_get_trading_decision(self, meta_agent):
        """Test getting concrete trading decision."""
        features = torch.randn(1, 60, 44)
        position_state = torch.zeros(3)
        risk_state = torch.zeros(4)
        trade_state = torch.zeros(5)
        atr = torch.tensor(50.0)
        price = torch.tensor(15000.0)

        decision = meta_agent.get_trading_decision(
            features, position_state, risk_state, trade_state, atr, price
        )

        assert isinstance(decision, TradingDecision)
        assert 0 <= decision.action <= 3
        assert -1 <= decision.position_size <= 1
        assert 0 <= decision.confidence <= 1

    def test_gradient_flow(self, meta_agent, sample_features):
        """Test gradients flow through main components."""
        batch_size = sample_features.shape[0]

        outputs = meta_agent(
            sample_features,
            torch.zeros(batch_size, 3),
            torch.zeros(batch_size, 4),
            torch.zeros(batch_size, 5),
            torch.ones(batch_size, 1) * 50.0,
            torch.ones(batch_size, 1) * 15000.0
        )

        # Compute combined loss using main outputs
        loss = (
            outputs['position_size'].sum() +
            outputs['confidence'].sum() +
            outputs['action_probs'].sum() +
            outputs['duration_output']['holding_time_minutes'].sum() +
            outputs['duration_output']['exit_condition_probs'].sum() +
            outputs['risk_output']['risk_multiplier'].sum()
        )
        loss.backward()

        # Check that at least some parameters in each key component have gradients
        key_components = ['transformer', 'profit_agent', 'risk_agent', 'duration_agent', 'decision_net']
        components_with_grads = set()

        for name, param in meta_agent.named_parameters():
            if param.requires_grad and param.grad is not None:
                for component in key_components:
                    if component in name:
                        components_with_grads.add(component)
                        break

        # Each key component should have at least one parameter with gradients
        for component in key_components:
            assert component in components_with_grads, f"No gradients in {component}"


class TestEnsembleMetaAgent:
    """Tests for EnsembleMetaAgent."""

    def test_forward(self, config_dict):
        """Test ensemble forward pass."""
        cfg = config_dict['model']
        ensemble = EnsembleMetaAgent(
            n_models=3,
            input_dim=44,
            embedding_dim=cfg['transformer']['d_model'],
            transformer_layers=cfg['transformer']['n_layers'],
            transformer_heads=cfg['transformer']['n_heads'],
            transformer_ff=cfg['transformer']['d_ff']
        )

        features = torch.randn(4, 60, 44)
        outputs = ensemble(
            features,
            torch.zeros(4, 3),
            torch.zeros(4, 4),
            torch.zeros(4, 5),
            torch.ones(4, 1) * 50.0,
            torch.ones(4, 1) * 15000.0
        )

        assert 'action' in outputs
        assert 'uncertainty' in outputs
        assert len(outputs['individual_outputs']) == 3
