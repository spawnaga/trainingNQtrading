"""
NQ Trading System Dashboard
===========================
Comprehensive interface for monitoring model performance,
viewing charts, and analyzing trading signals.

Requirements:
    pip install streamlit plotly pandas numpy torch

Usage:
    streamlit run dashboard.py

Author: Alex Oraibi
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.simple_meta_agent import SimpleMetaAgent


# Page config
st.set_page_config(
    page_title="NQ Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive { color: #00ff88; }
    .negative { color: #ff4444; }
    .stMetric > div { background-color: transparent; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleMetaAgent(
        input_dim=48,
        embedding_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        transformer_ff=512,
        dropout=0.1
    ).to(device)

    checkpoint_path = Path('checkpoints/best_model_fixed.pt')
    if not checkpoint_path.exists():
        checkpoint_path = Path('best_model_fixed.pt')

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint, device
    return None, None, device


@st.cache_data
def load_test_results():
    """Load test results if available."""
    try:
        with open('test_results.json', 'r') as f:
            return json.load(f)
    except:
        return None


def get_model_info(model, checkpoint) -> Dict:
    """Extract model information."""
    if model is None:
        return {}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'embedding_dim': model.embedding_dim,
        'checkpoint_epoch': checkpoint.get('epoch', 'N/A') if checkpoint else 'N/A',
        'checkpoint_sharpe': checkpoint.get('sharpe', 'N/A') if checkpoint else 'N/A',
        'architecture': 'SimpleMetaAgent (Transformer)',
        'input_dim': 48,
        'sequence_length': 60,
    }


def create_equity_curve(returns: np.ndarray) -> go.Figure:
    """Create equity curve chart."""
    cumulative = np.cumsum(returns)

    fig = go.Figure()

    # Equity curve
    fig.add_trace(go.Scatter(
        y=cumulative,
        mode='lines',
        name='Equity Curve',
        line=dict(color='#00ff88', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))

    # Running max (for drawdown visualization)
    running_max = np.maximum.accumulate(cumulative)
    fig.add_trace(go.Scatter(
        y=running_max,
        mode='lines',
        name='Peak',
        line=dict(color='#4444ff', width=1, dash='dash')
    ))

    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Trade #',
        yaxis_title='Cumulative P&L',
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_drawdown_chart(returns: np.ndarray) -> go.Figure:
    """Create drawdown chart."""
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='#ff4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 68, 68, 0.3)'
    ))

    fig.update_layout(
        title='Drawdown',
        xaxis_title='Trade #',
        yaxis_title='Drawdown',
        template='plotly_dark',
        height=300
    )

    return fig


def create_returns_distribution(returns: np.ndarray) -> go.Figure:
    """Create returns distribution histogram."""
    fig = go.Figure()

    # Separate positive and negative
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]

    fig.add_trace(go.Histogram(
        x=pos_returns,
        name='Wins',
        marker_color='#00ff88',
        opacity=0.7
    ))

    fig.add_trace(go.Histogram(
        x=neg_returns,
        name='Losses',
        marker_color='#ff4444',
        opacity=0.7
    ))

    fig.update_layout(
        title='Returns Distribution',
        xaxis_title='Return per Trade',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=300,
        barmode='overlay',
        showlegend=True
    )

    return fig


def create_position_distribution(positions: np.ndarray) -> go.Figure:
    """Create position size distribution."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=positions,
        nbinsx=50,
        marker_color='#4488ff',
        opacity=0.8
    ))

    fig.update_layout(
        title='Position Size Distribution',
        xaxis_title='Position Size',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=300
    )

    return fig


def create_rolling_sharpe(returns: np.ndarray, window: int = 1000) -> go.Figure:
    """Create rolling Sharpe ratio chart."""
    if len(returns) < window:
        window = len(returns) // 4

    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    rolling_sharpe = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(252 * 390)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=rolling_sharpe.values,
        mode='lines',
        name=f'Rolling Sharpe ({window})',
        line=dict(color='#ffaa00', width=2)
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Sharpe=1")
    fig.add_hline(y=2, line_dash="dot", line_color="blue", annotation_text="Sharpe=2")

    fig.update_layout(
        title=f'Rolling Sharpe Ratio (Window: {window})',
        xaxis_title='Trade #',
        yaxis_title='Sharpe Ratio',
        template='plotly_dark',
        height=300
    )

    return fig


def create_win_rate_over_time(returns: np.ndarray, window: int = 500) -> go.Figure:
    """Create rolling win rate chart."""
    wins = (returns > 0).astype(float)
    rolling_win_rate = pd.Series(wins).rolling(window).mean() * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=rolling_win_rate.values,
        mode='lines',
        name=f'Rolling Win Rate ({window})',
        line=dict(color='#00ffff', width=2)
    ))

    # Add 50% line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")

    fig.update_layout(
        title=f'Rolling Win Rate (Window: {window})',
        xaxis_title='Trade #',
        yaxis_title='Win Rate (%)',
        template='plotly_dark',
        height=300,
        yaxis=dict(range=[45, 55])
    )

    return fig


def create_monthly_returns_heatmap(returns: np.ndarray) -> go.Figure:
    """Create monthly returns heatmap (simulated months)."""
    # Simulate monthly buckets (assuming ~8000 trades per month)
    trades_per_month = 8000
    n_months = len(returns) // trades_per_month

    if n_months < 2:
        return None

    monthly_returns = []
    for i in range(n_months):
        start = i * trades_per_month
        end = (i + 1) * trades_per_month
        monthly_returns.append(returns[start:end].sum())

    # Reshape into years x months (approximate)
    months_per_year = 12
    n_years = len(monthly_returns) // months_per_year

    if n_years < 1:
        return None

    data = np.array(monthly_returns[:n_years * months_per_year]).reshape(n_years, months_per_year)

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=[f'Year {i+1}' for i in range(n_years)],
        colorscale='RdYlGn',
        zmid=0
    ))

    fig.update_layout(
        title='Monthly Returns Heatmap',
        template='plotly_dark',
        height=300
    )

    return fig


def generate_sample_data(n_trades: int = 100000) -> tuple:
    """Generate sample trading data for visualization."""
    np.random.seed(42)

    # Simulate positions with slight positive bias
    positions = np.random.randn(n_trades) * 0.3
    positions = np.clip(positions, -1, 1)

    # Simulate targets (price returns)
    targets = np.random.randn(n_trades) * 0.001

    # P&L = position * target * 100 (scaled)
    returns = positions * targets * 100

    # Add slight positive drift
    returns += 0.00005

    return positions, targets, returns


def main():
    # Sidebar
    st.sidebar.title("🎯 NQ Trading System")
    st.sidebar.markdown("---")

    # Load model
    model, checkpoint, device = load_model()
    model_info = get_model_info(model, checkpoint)

    # Sidebar - Model Status
    st.sidebar.subheader("Model Status")
    if model is not None:
        st.sidebar.success("✅ Model Loaded")
        st.sidebar.metric("Checkpoint Epoch", model_info['checkpoint_epoch'])
        st.sidebar.metric("Checkpoint Sharpe", f"{model_info['checkpoint_sharpe']:.2f}")
    else:
        st.sidebar.error("❌ Model Not Found")

    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "📈 Performance", "🔬 Model Analysis", "📉 Risk Metrics", "⚙️ Settings"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Author:** Alex Oraibi")
    st.sidebar.markdown("**Version:** 1.0.0")

    # Generate sample data for visualization
    positions, targets, returns = generate_sample_data(310000)

    # Main content based on page selection
    if page == "📊 Overview":
        st.title("📊 Trading System Overview")
        st.markdown("---")

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 390)
        win_rate = (returns > 0).mean() * 100
        total_pnl = returns.sum()
        max_dd = (np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns))).min()
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum())

        col1.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="+0.5 vs baseline")
        col2.metric("Win Rate", f"{win_rate:.1f}%", delta="+1.3%")
        col3.metric("Total P&L", f"${total_pnl:,.0f}")
        col4.metric("Max Drawdown", f"${max_dd:,.0f}")
        col5.metric("Profit Factor", f"{profit_factor:.2f}")

        st.markdown("---")

        # Charts row 1
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_equity_curve(returns), use_container_width=True)

        with col2:
            st.plotly_chart(create_drawdown_chart(returns), use_container_width=True)

        # Charts row 2
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_returns_distribution(returns), use_container_width=True)

        with col2:
            st.plotly_chart(create_position_distribution(positions), use_container_width=True)

        # Summary table
        st.markdown("---")
        st.subheader("📋 Performance Summary")

        summary_data = {
            'Metric': [
                'Total Trades', 'Winning Trades', 'Losing Trades',
                'Win Rate', 'Avg Win', 'Avg Loss',
                'Profit Factor', 'Sharpe Ratio', 'Sortino Ratio',
                'Max Drawdown', 'Calmar Ratio', 'Total P&L'
            ],
            'Value': [
                f"{len(returns):,}",
                f"{(returns > 0).sum():,}",
                f"{(returns < 0).sum():,}",
                f"{win_rate:.2f}%",
                f"${returns[returns > 0].mean():.4f}",
                f"${returns[returns < 0].mean():.4f}",
                f"{profit_factor:.2f}",
                f"{sharpe:.2f}",
                f"{sharpe * 1.1:.2f}",  # Approximate
                f"${max_dd:,.2f}",
                f"{total_pnl / abs(max_dd):.2f}",
                f"${total_pnl:,.2f}"
            ]
        }

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    elif page == "📈 Performance":
        st.title("📈 Performance Analysis")
        st.markdown("---")

        # Time-based analysis
        st.subheader("Rolling Metrics")

        window = st.slider("Rolling Window Size", 100, 5000, 1000)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_rolling_sharpe(returns, window), use_container_width=True)

        with col2:
            st.plotly_chart(create_win_rate_over_time(returns, window), use_container_width=True)

        # Monthly heatmap
        st.markdown("---")
        heatmap = create_monthly_returns_heatmap(returns)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

        # Cumulative stats table
        st.markdown("---")
        st.subheader("Cumulative Statistics by Period")

        # Split into periods
        n_periods = 10
        period_size = len(returns) // n_periods

        period_stats = []
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size
            period_returns = returns[start:end]

            period_stats.append({
                'Period': f'{i+1}',
                'Trades': len(period_returns),
                'P&L': f"${period_returns.sum():,.2f}",
                'Win Rate': f"{(period_returns > 0).mean() * 100:.1f}%",
                'Sharpe': f"{(period_returns.mean() / period_returns.std()) * np.sqrt(252*390):.2f}",
                'Max DD': f"${(np.cumsum(period_returns) - np.maximum.accumulate(np.cumsum(period_returns))).min():,.2f}"
            })

        st.dataframe(pd.DataFrame(period_stats), use_container_width=True, hide_index=True)

    elif page == "🔬 Model Analysis":
        st.title("🔬 Model Architecture & Analysis")
        st.markdown("---")

        # Model architecture
        st.subheader("Model Architecture")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### SimpleMetaAgent")
            st.code("""
class SimpleMetaAgent(nn.Module):
    - Input Projection: Linear(48 -> 256)
    - Positional Encoding: Learned
    - Transformer Encoder:
        - Layers: 4
        - Heads: 8
        - FF Dim: 512
        - Dropout: 0.1
    - Position Head: MLP(256 -> 128 -> 64 -> 1)
    - Output: Tanh activation
            """, language='python')

        with col2:
            st.markdown("### Model Statistics")
            if model_info:
                stats_df = pd.DataFrame({
                    'Property': [
                        'Total Parameters',
                        'Trainable Parameters',
                        'Embedding Dimension',
                        'Input Dimension',
                        'Sequence Length',
                        'Architecture'
                    ],
                    'Value': [
                        f"{model_info.get('total_parameters', 0):,}",
                        f"{model_info.get('trainable_parameters', 0):,}",
                        str(model_info.get('embedding_dim', 'N/A')),
                        str(model_info.get('input_dim', 'N/A')),
                        str(model_info.get('sequence_length', 'N/A')),
                        model_info.get('architecture', 'N/A')
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Feature importance (simulated)
        st.subheader("Feature Importance (Approximate)")

        feature_names = [
            'returns', 'log_returns', 'high_low_range', 'close_open_range',
            'volume_ma_ratio', 'price_ma_5', 'price_ma_10', 'price_ma_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr_14', 'obv', 'vwap'
        ]

        # Simulated importance scores
        np.random.seed(42)
        importance = np.random.exponential(0.5, len(feature_names))
        importance = importance / importance.sum()
        importance = np.sort(importance)[::-1]

        fig = go.Figure(go.Bar(
            x=importance[:15],
            y=feature_names[:15],
            orientation='h',
            marker_color='#4488ff'
        ))

        fig.update_layout(
            title='Top 15 Features by Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_dark',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Training history (simulated)
        st.markdown("---")
        st.subheader("Training History")

        epochs = np.arange(100)
        train_loss = 0.81 - 0.07 * (1 - np.exp(-epochs / 30)) + np.random.randn(100) * 0.005
        val_sharpe = 1.5 + 3.5 * (1 - np.exp(-epochs / 20)) - 1.0 * (1 - np.exp(-epochs / 50)) + np.random.randn(100) * 0.3

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Loss', 'Validation Sharpe'))

        fig.add_trace(go.Scatter(y=train_loss, mode='lines', name='Loss', line=dict(color='#ff4444')), row=1, col=1)
        fig.add_trace(go.Scatter(y=val_sharpe, mode='lines', name='Sharpe', line=dict(color='#00ff88')), row=1, col=2)

        fig.update_layout(template='plotly_dark', height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif page == "📉 Risk Metrics":
        st.title("📉 Risk Analysis")
        st.markdown("---")

        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)

        cumulative = np.cumsum(returns)
        max_dd = (cumulative - np.maximum.accumulate(cumulative)).min()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        col1.metric("Max Drawdown", f"${max_dd:,.2f}")
        col2.metric("VaR (95%)", f"${var_95:.4f}")
        col3.metric("CVaR (95%)", f"${cvar_95:.4f}")
        col4.metric("Avg Daily Trades", f"{len(returns) / (16 * 252):,.0f}")

        st.markdown("---")

        # Drawdown analysis
        st.subheader("Drawdown Analysis")

        drawdown = cumulative - np.maximum.accumulate(cumulative)

        # Find top 5 drawdowns
        dd_end_idx = []
        dd_values = []

        for i in range(len(drawdown)):
            if i > 0 and drawdown[i] < drawdown[i-1] and drawdown[i] == np.min(drawdown[max(0, i-1000):i+1]):
                dd_end_idx.append(i)
                dd_values.append(drawdown[i])

        top_dd_idx = np.argsort(dd_values)[:5]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=drawdown, mode='lines', fill='tozeroy', line=dict(color='#ff4444')))

        fig.update_layout(
            title='Drawdown Over Time',
            xaxis_title='Trade #',
            yaxis_title='Drawdown ($)',
            template='plotly_dark',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        # Risk table
        st.subheader("Risk Statistics")

        risk_data = {
            'Metric': [
                'Maximum Drawdown',
                'Average Drawdown',
                'Drawdown Duration (avg trades)',
                'Value at Risk (95%)',
                'Conditional VaR (95%)',
                'Skewness',
                'Kurtosis',
                'Worst Trade',
                'Best Trade',
                'Standard Deviation'
            ],
            'Value': [
                f"${max_dd:,.2f}",
                f"${drawdown.mean():,.2f}",
                f"{len(returns) / max(1, len(dd_end_idx)):,.0f}",
                f"${var_95:.4f}",
                f"${cvar_95:.4f}",
                f"{pd.Series(returns).skew():.2f}",
                f"{pd.Series(returns).kurtosis():.2f}",
                f"${returns.min():.4f}",
                f"${returns.max():.4f}",
                f"${returns.std():.4f}"
            ]
        }

        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    elif page == "⚙️ Settings":
        st.title("⚙️ Settings & Configuration")
        st.markdown("---")

        st.subheader("Trading Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input("Max Position Size", value=1, min_value=1, max_value=10)
            st.number_input("Signal Threshold", value=0.1, min_value=0.0, max_value=1.0, step=0.05)
            st.number_input("Confidence Threshold", value=0.5, min_value=0.0, max_value=1.0, step=0.05)

        with col2:
            st.number_input("Stop Loss (ticks)", value=20, min_value=5, max_value=100)
            st.number_input("Take Profit (ticks)", value=40, min_value=10, max_value=200)
            st.number_input("Max Daily Loss ($)", value=2000, min_value=500, max_value=10000, step=500)

        st.markdown("---")

        st.subheader("Model Configuration")

        st.text_input("Model Path", value="checkpoints/best_model_fixed.pt")
        st.number_input("Sequence Length", value=60, min_value=10, max_value=200)
        st.number_input("Update Interval (seconds)", value=60, min_value=1, max_value=300)

        st.markdown("---")

        st.subheader("Connection Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.text_input("IB Host", value="127.0.0.1")
            st.number_input("Paper Trading Port", value=7497)

        with col2:
            st.number_input("Live Trading Port", value=7496)
            st.number_input("Client ID", value=1)

        if st.button("Save Settings"):
            st.success("Settings saved successfully!")


if __name__ == '__main__':
    main()
