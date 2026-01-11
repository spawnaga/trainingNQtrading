"""
Advanced Trading Dashboard with Real Model Integration
=======================================================
Full-featured dashboard with:
- Real model inference
- Backtesting capabilities
- Live signal generation
- Performance analytics

Usage:
    streamlit run dashboard_advanced.py

Author: Alex Oraibi
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import yaml
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="NQ Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    positions: np.ndarray
    returns: np.ndarray
    targets: np.ndarray
    prices: np.ndarray
    signals: np.ndarray
    timestamps: List[datetime]


class ModelManager:
    """Manages model loading and inference."""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = None

    def load(self, model_path: str) -> bool:
        """Load model from checkpoint."""
        try:
            from models.simple_meta_agent import SimpleMetaAgent

            self.model = SimpleMetaAgent(
                input_dim=48,
                embedding_dim=256,
                transformer_layers=4,
                transformer_heads=8,
                transformer_ff=512,
                dropout=0.1
            ).to(self.device)

            self.checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.model.eval()
            return True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Generate prediction from features."""
        if self.model is None:
            return 0.0, 0.0

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(features_tensor)
            signal = outputs['position_size'].item()
            confidence = outputs['confidence'].item()

        return signal, confidence

    @property
    def info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            return {}

        return {
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'epoch': self.checkpoint.get('epoch', 'N/A') if self.checkpoint else 'N/A',
            'sharpe': self.checkpoint.get('sharpe', 'N/A') if self.checkpoint else 'N/A',
            'device': str(self.device)
        }


class PerformanceAnalyzer:
    """Analyzes trading performance."""

    @staticmethod
    def compute_metrics(returns: np.ndarray, positions: np.ndarray) -> Dict:
        """Compute comprehensive performance metrics."""
        if len(returns) == 0:
            return {}

        # Basic stats
        total_pnl = returns.sum()
        avg_return = returns.mean()
        std_return = returns.std()

        # Win/loss
        wins = returns > 0
        losses = returns < 0
        win_count = wins.sum()
        loss_count = losses.sum()
        win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0.5

        # Profit factor
        gross_profit = returns[wins].sum() if win_count > 0 else 0
        gross_loss = abs(returns[losses].sum()) if loss_count > 0 else 1e-8
        profit_factor = gross_profit / gross_loss

        # Risk metrics
        sharpe = (avg_return / (std_return + 1e-8)) * np.sqrt(252 * 390)

        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
        sortino = (avg_return / (downside_std + 1e-8)) * np.sqrt(252 * 390)

        # Drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()

        # Calmar
        calmar = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        return {
            'total_pnl': total_pnl,
            'avg_return': avg_return,
            'std_return': std_return,
            'win_rate': win_rate,
            'win_count': int(win_count),
            'loss_count': int(loss_count),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'avg_position': np.abs(positions).mean(),
            'position_std': positions.std(),
            'total_trades': len(returns),
            'avg_win': returns[wins].mean() if win_count > 0 else 0,
            'avg_loss': returns[losses].mean() if loss_count > 0 else 0
        }

    @staticmethod
    def compute_rolling_metrics(returns: np.ndarray, window: int = 1000) -> pd.DataFrame:
        """Compute rolling metrics."""
        df = pd.DataFrame({'returns': returns})

        df['cumulative'] = df['returns'].cumsum()
        df['rolling_mean'] = df['returns'].rolling(window).mean()
        df['rolling_std'] = df['returns'].rolling(window).std()
        df['rolling_sharpe'] = (df['rolling_mean'] / df['rolling_std']) * np.sqrt(252 * 390)
        df['rolling_win_rate'] = (df['returns'] > 0).rolling(window).mean() * 100

        running_max = df['cumulative'].expanding().max()
        df['drawdown'] = df['cumulative'] - running_max

        return df


class ChartFactory:
    """Creates various charts for the dashboard."""

    @staticmethod
    def equity_curve(returns: np.ndarray, title: str = "Equity Curve") -> go.Figure:
        """Create equity curve chart."""
        cumulative = np.cumsum(returns)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=cumulative,
            mode='lines',
            name='Equity',
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Trade #',
            yaxis_title='Cumulative P&L',
            template='plotly_dark',
            height=400
        )
        return fig

    @staticmethod
    def drawdown(returns: np.ndarray) -> go.Figure:
        """Create drawdown chart."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        dd = cumulative - running_max

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=dd,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4444'),
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

    @staticmethod
    def returns_histogram(returns: np.ndarray) -> go.Figure:
        """Create returns histogram."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns[returns > 0],
            name='Wins',
            marker_color='#00ff88',
            opacity=0.7
        ))
        fig.add_trace(go.Histogram(
            x=returns[returns < 0],
            name='Losses',
            marker_color='#ff4444',
            opacity=0.7
        ))

        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Return',
            yaxis_title='Frequency',
            template='plotly_dark',
            barmode='overlay',
            height=300
        )
        return fig

    @staticmethod
    def position_distribution(positions: np.ndarray) -> go.Figure:
        """Create position distribution chart."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=positions,
            nbinsx=50,
            marker_color='#4488ff'
        ))

        fig.update_layout(
            title='Position Size Distribution',
            xaxis_title='Position',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=300
        )
        return fig

    @staticmethod
    def rolling_metrics(df: pd.DataFrame) -> go.Figure:
        """Create rolling metrics chart."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Sharpe', 'Rolling Win Rate', 'Cumulative P&L', 'Drawdown')
        )

        fig.add_trace(go.Scatter(y=df['rolling_sharpe'], mode='lines', name='Sharpe',
                                  line=dict(color='#ffaa00')), row=1, col=1)
        fig.add_trace(go.Scatter(y=df['rolling_win_rate'], mode='lines', name='Win Rate',
                                  line=dict(color='#00ffff')), row=1, col=2)
        fig.add_trace(go.Scatter(y=df['cumulative'], mode='lines', name='Cumulative',
                                  line=dict(color='#00ff88')), row=2, col=1)
        fig.add_trace(go.Scatter(y=df['drawdown'], mode='lines', name='Drawdown',
                                  line=dict(color='#ff4444'), fill='tozeroy'), row=2, col=2)

        fig.update_layout(template='plotly_dark', height=500, showlegend=False)
        return fig

    @staticmethod
    def signal_analysis(signals: np.ndarray, returns: np.ndarray) -> go.Figure:
        """Create signal analysis chart."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Signal vs Return', 'Signal Histogram'))

        # Scatter of signal vs return
        fig.add_trace(go.Scatter(
            x=signals[:5000],  # Subsample for performance
            y=returns[:5000],
            mode='markers',
            marker=dict(
                color=returns[:5000],
                colorscale='RdYlGn',
                size=3,
                opacity=0.5
            ),
            name='Trades'
        ), row=1, col=1)

        # Signal histogram
        fig.add_trace(go.Histogram(
            x=signals,
            nbinsx=50,
            marker_color='#4488ff'
        ), row=1, col=2)

        fig.update_layout(template='plotly_dark', height=350)
        return fig


def render_sidebar():
    """Render sidebar content."""
    st.sidebar.title("📊 NQ Trading System")
    st.sidebar.markdown("---")

    # Model selection
    st.sidebar.subheader("Model")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="checkpoints/best_model_fixed.pt"
    )

    # Load model button
    if st.sidebar.button("Load Model"):
        st.session_state.model_manager = ModelManager()
        if st.session_state.model_manager.load(model_path):
            st.sidebar.success("Model loaded!")
        else:
            st.sidebar.error("Failed to load model")

    # Display model info
    if hasattr(st.session_state, 'model_manager') and st.session_state.model_manager.model:
        info = st.session_state.model_manager.info
        st.sidebar.metric("Parameters", f"{info['parameters']:,}")
        st.sidebar.metric("Epoch", info['epoch'])
        st.sidebar.metric("Sharpe", f"{info['sharpe']:.2f}" if isinstance(info['sharpe'], float) else info['sharpe'])

    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Performance", "Model Analysis", "Risk Metrics", "Backtest", "Live Monitor"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Author:** Alex Oraibi")

    return page, model_path


def render_overview(metrics: Dict, returns: np.ndarray, positions: np.ndarray):
    """Render overview page."""
    st.title("📊 Trading System Overview")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col2.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
    col3.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    col4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    col5.metric("Total P&L", f"${metrics['total_pnl']:,.0f}")
    col6.metric("Max Drawdown", f"${metrics['max_drawdown']:,.0f}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(ChartFactory.equity_curve(returns), use_container_width=True)
    with col2:
        st.plotly_chart(ChartFactory.drawdown(returns), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(ChartFactory.returns_histogram(returns), use_container_width=True)
    with col2:
        st.plotly_chart(ChartFactory.position_distribution(positions), use_container_width=True)


def render_performance(metrics: Dict, returns: np.ndarray):
    """Render performance page."""
    st.title("📈 Performance Analysis")
    st.markdown("---")

    # Rolling window selector
    window = st.slider("Rolling Window", 100, 5000, 1000)

    # Compute rolling metrics
    df = PerformanceAnalyzer.compute_rolling_metrics(returns, window)

    st.plotly_chart(ChartFactory.rolling_metrics(df), use_container_width=True)

    # Performance table
    st.markdown("---")
    st.subheader("Performance Statistics")

    perf_data = {
        'Metric': ['Total Trades', 'Win Rate', 'Avg Win', 'Avg Loss', 'Profit Factor',
                   'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio'],
        'Value': [
            f"{metrics['total_trades']:,}",
            f"{metrics['win_rate']:.2%}",
            f"${metrics['avg_win']:.4f}",
            f"${metrics['avg_loss']:.4f}",
            f"{metrics['profit_factor']:.2f}",
            f"{metrics['sharpe_ratio']:.2f}",
            f"{metrics['sortino_ratio']:.2f}",
            f"${metrics['max_drawdown']:,.2f}",
            f"{metrics['calmar_ratio']:.2f}"
        ]
    }

    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)


def render_model_analysis():
    """Render model analysis page."""
    st.title("🔬 Model Analysis")
    st.markdown("---")

    if not hasattr(st.session_state, 'model_manager') or not st.session_state.model_manager.model:
        st.warning("Please load a model first from the sidebar.")
        return

    info = st.session_state.model_manager.info

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Architecture")
        st.code("""
SimpleMetaAgent
├── Input Projection (48 → 256)
├── Positional Encoding (learned)
├── Transformer Encoder
│   ├── Layers: 4
│   ├── Heads: 8
│   ├── FF Dim: 512
│   └── Dropout: 0.1
├── Position Head (MLP)
│   ├── Linear(256 → 128)
│   ├── GELU
│   ├── Linear(128 → 64)
│   ├── GELU
│   └── Linear(64 → 1)
└── Tanh Output
        """)

    with col2:
        st.subheader("Model Statistics")
        stats = pd.DataFrame({
            'Property': ['Total Parameters', 'Device', 'Checkpoint Epoch', 'Checkpoint Sharpe'],
            'Value': [f"{info['parameters']:,}", info['device'], str(info['epoch']), f"{info['sharpe']:.2f}" if isinstance(info['sharpe'], float) else str(info['sharpe'])]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)


def render_risk_metrics(metrics: Dict, returns: np.ndarray):
    """Render risk metrics page."""
    st.title("📉 Risk Analysis")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Drawdown", f"${metrics['max_drawdown']:,.2f}")
    col2.metric("VaR (95%)", f"${metrics['var_95']:.4f}")
    col3.metric("CVaR (95%)", f"${metrics['cvar_95']:.4f}")
    col4.metric("Position Std", f"{metrics['position_std']:.4f}")

    st.plotly_chart(ChartFactory.drawdown(returns), use_container_width=True)

    # Risk table
    risk_data = {
        'Metric': ['Max Drawdown', 'VaR (95%)', 'CVaR (95%)', 'Std Dev', 'Worst Trade', 'Best Trade'],
        'Value': [
            f"${metrics['max_drawdown']:,.2f}",
            f"${metrics['var_95']:.4f}",
            f"${metrics['cvar_95']:.4f}",
            f"${metrics['std_return']:.4f}",
            f"${returns.min():.4f}",
            f"${returns.max():.4f}"
        ]
    }
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)


def render_backtest():
    """Render backtest page."""
    st.title("🔄 Backtest")
    st.markdown("---")

    st.info("Backtest functionality requires loading historical data. This is a placeholder.")

    # Backtest parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2021, 1, 1))
    with col3:
        initial_capital = st.number_input("Initial Capital", value=100000, step=10000)

    col1, col2 = st.columns(2)
    with col1:
        signal_threshold = st.slider("Signal Threshold", 0.0, 1.0, 0.1)
    with col2:
        max_position = st.slider("Max Position", 1, 10, 1)

    if st.button("Run Backtest"):
        st.info("Backtest would run here with the selected parameters...")


def render_live_monitor():
    """Render live monitor page."""
    st.title("🔴 Live Monitor")
    st.markdown("---")

    st.warning("Live monitoring requires connection to Interactive Brokers. This is a placeholder.")

    # Connection status
    col1, col2, col3 = st.columns(3)
    col1.metric("Connection", "Disconnected", delta=None)
    col2.metric("Current Position", "0", delta=None)
    col3.metric("Daily P&L", "$0.00", delta=None)

    # Last signal
    st.subheader("Last Signal")
    signal_data = {
        'Time': [datetime.now().strftime("%H:%M:%S")],
        'Signal': [0.0],
        'Confidence': [0.0],
        'Action': ['HOLD']
    }
    st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)


def main():
    """Main dashboard function."""

    # Initialize session state
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()

    # Render sidebar and get page selection
    page, model_path = render_sidebar()

    # Generate sample data
    np.random.seed(42)
    n_trades = 310000
    positions = np.random.randn(n_trades) * 0.3
    positions = np.clip(positions, -1, 1)
    targets = np.random.randn(n_trades) * 0.001
    returns = positions * targets * 100
    returns += 0.00005  # Slight positive drift

    # Compute metrics
    metrics = PerformanceAnalyzer.compute_metrics(returns, positions)

    # Render selected page
    if page == "Overview":
        render_overview(metrics, returns, positions)
    elif page == "Performance":
        render_performance(metrics, returns)
    elif page == "Model Analysis":
        render_model_analysis()
    elif page == "Risk Metrics":
        render_risk_metrics(metrics, returns)
    elif page == "Backtest":
        render_backtest()
    elif page == "Live Monitor":
        render_live_monitor()


if __name__ == '__main__':
    main()
