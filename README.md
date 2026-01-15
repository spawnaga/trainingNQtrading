# NQ Trading System

A sophisticated **Meta-Agent Trading System** using Transformer-based neural networks optimized via **Island-based Genetic Algorithms** for trading NQ (Nasdaq 100) and CL (Crude Oil) futures.

**DISCLAIMER: This software is for educational and research purposes only. It is NOT intended for actual trading. Trading involves substantial risk of loss and is not suitable for all investors. The author assumes no responsibility for any financial losses incurred through the use of this software.**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Configuration](#configuration)
7. [Data Preparation](#data-preparation)
8. [Training](#training)
   - [Basic Training](#basic-training)
   - [Multi-GPU Genetic Algorithm Training](#multi-gpu-genetic-algorithm-training)
   - [Remote Training](#remote-training)
9. [Backtesting](#backtesting)
10. [Paper Trading](#paper-trading)
11. [Live Trading](#live-trading)
12. [Model Architecture Details](#model-architecture-details)
13. [Genetic Algorithm Optimization](#genetic-algorithm-optimization)
14. [Loss Function Design](#loss-function-design)
15. [Troubleshooting](#troubleshooting)
16. [References](#references)

---

## Overview

This system implements a **Meta-Agent architecture** that combines multiple specialized neural network agents:

- **Transformer Agent**: Processes sequential market data using multi-head self-attention
- **Profit Maximizer Agent**: Determines optimal position sizing using actor-critic methods
- **Risk Controller Agent**: Computes dynamic risk parameters (stop-loss, take-profit)
- **Duration Agent**: Classifies expected holding periods

The system uses **Island-based Genetic Algorithms** to optimize hyperparameters across multiple GPUs in parallel, preventing premature convergence while exploring a large search space efficiently.

### Key Features

- Multi-GPU training with GPU memory-aware batch sizing
- Island-based genetic algorithm for hyperparameter optimization
- Signal variation loss to prevent mode collapse
- Support for NQ and CL futures
- Paper trading and live trading via Interactive Brokers
- Comprehensive backtesting with performance metrics

---

## Architecture

```
                    ┌─────────────────┐
                    │  Market Data    │
                    │  (OHLCV + Ind)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Transformer    │
                    │     Agent       │
                    │  (Embedding)    │
                    └────────┬────────┘
                             │
         ┌───────────┬───────┴───────┬───────────┐
         │           │               │           │
    ┌────▼────┐ ┌────▼────┐   ┌─────▼─────┐ ┌───▼────┐
    │ Profit  │ │  Risk   │   │ Duration  │ │ Meta   │
    │Maximizer│ │Controller│   │  Agent    │ │Attention│
    └────┬────┘ └────┬────┘   └─────┬─────┘ └───┬────┘
         │           │               │           │
         └───────────┴───────┬───────┴───────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Decision │
                    │  (Action, Size, │
                    │   Stop, Target) │
                    └─────────────────┘
```

---

## Requirements

### Hardware Requirements

**Minimum:**
- CPU: 8+ cores
- RAM: 32GB
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- Storage: 50GB+ SSD

**Recommended for Multi-GPU Training:**
- CPU: 16+ cores
- RAM: 64GB+
- GPU: 2-4 GPUs (e.g., 2x RTX 5090 + 2x RTX 3090)
- Storage: 100GB+ NVMe SSD

### Software Requirements

- Python 3.10+
- CUDA 12.0+ (for GPU training)
- Linux (recommended for multi-GPU) or Windows 10/11
- Interactive Brokers Gateway (for live/paper trading)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/spawnaga/trainingNQtrading.git
cd trainingNQtrading
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install PyTorch with CUDA

Visit https://pytorch.org/get-started/locally/ and select your configuration.

**Example for CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Full dependencies include:**
```
# Core ML/DL
torch>=2.0.0
torchvision>=0.15.0

# Trading
ib_insync>=0.9.86

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Technical Analysis
ta>=0.11.0

# Genetic Algorithm
deap>=1.4.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

# Logging & Monitoring
tensorboard>=2.14.0
loguru>=0.7.0

# Visualization
plotly>=5.18.0
matplotlib>=3.7.0

# Utilities
tqdm>=4.66.0
joblib>=1.3.0

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
GPUs: 4
```

---

## Project Structure

```
nq_trading_system/
├── config/                          # Configuration files
│   ├── config.yaml                  # Default configuration
│   ├── config_cl_optimized.yaml     # Optimized config for CL futures
│   └── config_remote.yaml           # Remote training configuration
│
├── data/                            # Data directory
│   └── csv/                         # Place your OHLCV CSV files here
│
├── data_pipeline/                   # Data processing
│   ├── __init__.py
│   ├── loader.py                    # Dataset and DataLoader
│   ├── preprocessor.py              # Feature engineering
│   └── time_encoder.py              # Cyclical time encoding
│
├── models/                          # Neural network models
│   ├── __init__.py
│   ├── meta_agent.py                # Main Meta-Agent (full version)
│   ├── simple_meta_agent.py         # Simplified Meta-Agent
│   ├── transformer_agent.py         # Transformer encoder
│   ├── profit_agent.py              # Profit maximizer (actor-critic)
│   ├── risk_agent.py                # Risk controller
│   └── duration_agent.py            # Trade duration classifier
│
├── training/                        # Training modules
│   ├── __init__.py
│   ├── trainer.py                   # Main training loop with loss function
│   ├── fitness.py                   # Fitness evaluation for GA
│   ├── genetic_optimizer.py         # Single-GPU genetic optimizer
│   └── multi_gpu_genetic_trainer.py # Multi-GPU island-based GA
│
├── trading/                         # Trading infrastructure
│   ├── __init__.py
│   ├── ib_connector.py              # Interactive Brokers connection
│   ├── order_manager.py             # Order execution
│   ├── position_manager.py          # Position tracking
│   └── paper_trader.py              # Paper trading simulation
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── gpu_utils.py                 # GPU memory management
│   ├── logger.py                    # Logging utilities
│   └── metrics.py                   # Performance metrics
│
├── scripts/                         # Helper scripts
│   ├── remote_train.py              # Remote training launcher
│   └── train_multigpu.sh            # Multi-GPU training script
│
├── paper/                           # Research documentation
│   ├── meta_agent_ga_approach.md    # Algorithm documentation
│   └── nq_trading_system_paper.md   # System overview paper
│
├── checkpoints/                     # Model checkpoints (created during training)
├── logs/                            # Training logs
│
├── main.py                          # Main entry point
├── train.py                         # Training script
├── backtest.py                      # Backtesting script
├── dashboard.py                     # Trading dashboard
├── live_trading.py                  # Live trading script
├── launch_remote_training.py        # Remote training launcher
├── start_ga_training.sh             # GA training shell script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Configuration

### Main Configuration File (`config/config.yaml`)

```yaml
# Data settings
data:
  csv_folder: "data/csv"        # Path to OHLCV data
  sequence_length: 60           # Lookback window (bars)
  train_split: 0.7              # Training data ratio
  val_split: 0.15               # Validation data ratio
  test_split: 0.15              # Test data ratio
  features:
    - open
    - high
    - low
    - close
    - volume

# Model architecture
model:
  transformer:
    d_model: 256                # Embedding dimension
    n_heads: 8                  # Attention heads
    n_layers: 4                 # Transformer layers
    d_ff: 1024                  # Feedforward dimension
    dropout: 0.1                # Dropout rate

  profit_agent:
    hidden_dim: 128             # Hidden layer size
    n_layers: 2                 # Number of layers

  risk_agent:
    hidden_dim: 64
    n_layers: 2

  duration_agent:
    hidden_dim: 64
    n_layers: 2
    num_classes: 3              # scalp, intraday, swing

  meta_agent:
    hidden_dim: 256
    n_heads: 4

# Training settings
training:
  batch_size: 64                # Batch size per GPU
  epochs: 100                   # Training epochs
  learning_rate: 0.0001         # Learning rate
  weight_decay: 0.0001          # L2 regularization
  gradient_clip: 1.0            # Gradient clipping
  device: "cuda"                # cuda or cpu
  mixed_precision: true         # Use FP16/BF16
  checkpoint_dir: "checkpoints"
  log_dir: "logs"

# Genetic algorithm settings
genetic:
  population_size: 50           # Total population
  generations: 100              # Max generations
  mutation_rate: 0.1            # Mutation probability
  crossover_rate: 0.7           # Crossover probability
  tournament_size: 3            # Tournament selection size
  elite_size: 5                 # Elite individuals to preserve

# Trading settings
trading:
  ib_host: "127.0.0.1"
  ib_port: 4002
  paper_trading: true
  max_position: 2               # Maximum contracts
  risk_per_trade: 0.02          # 2% risk per trade
  max_daily_loss: 0.05          # 5% max daily drawdown

# Backtest settings
backtest:
  initial_capital: 100000
  commission: 2.25              # Per contract per side
  slippage: 0.25                # Points
```

### Multi-GPU Configuration (`config/config_cl_optimized.yaml`)

For multi-GPU training, use the optimized configuration:

```yaml
# Multi-GPU settings for 2x5090 + 2x3090
training:
  distributed:
    enabled: true
    backend: "nccl"             # NCCL for Linux
    world_size: 4               # Total GPUs
    num_workers: 4              # DataLoader workers

# Island-based GA settings
genetic:
  population_size: 80           # Total population
  n_islands: 4                  # One island per GPU
  pop_per_island: 20            # Population per island
  generations: 100
  mutation_rate: 0.15
  crossover_rate: 0.80
  migration_interval: 10        # Migrate every N generations
  migration_size: 2             # Individuals to migrate
  epochs_per_eval: 10           # Quick evaluation epochs
  final_epochs: 200             # Full training for best
```

---

## Data Preparation

### Required Data Format

Place your OHLCV data as CSV files in `data/csv/`. Each CSV should have:

```csv
datetime,open,high,low,close,volume
2024-01-02 09:30:00,17000.00,17005.50,16998.25,17003.75,1250
2024-01-02 09:31:00,17003.75,17010.00,17002.00,17008.50,980
...
```

**Required columns:**
- `datetime`: Timestamp (parsed automatically)
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Volume

### Data Sources

You can obtain futures data from:
- Interactive Brokers (via `ib_insync`)
- Polygon.io API
- Databento
- QuantConnect
- Your broker's data export

### Combining Multiple Files

The data loader automatically combines all CSV files in the folder:

```bash
data/
└── csv/
    ├── NQ_2023_01.csv
    ├── NQ_2023_02.csv
    ├── NQ_2024_01.csv
    └── ...
```

---

## Training

### Basic Training

**Single GPU training with default configuration:**

```bash
python main.py --mode train --config config/config.yaml
```

**Or directly:**

```bash
python train.py --config config/config.yaml
```

**Arguments:**
- `--config`: Path to configuration file
- `--epochs`: Override epochs from config
- `--batch-size`: Override batch size
- `--learning-rate`: Override learning rate

### Multi-GPU Genetic Algorithm Training

This is the **recommended training method** for best results.

#### Step 1: Configure GPUs

Edit `config/config_cl_optimized.yaml`:

```yaml
genetic:
  n_islands: 4              # Set to your GPU count
  pop_per_island: 20        # Population per island
  generations: 100          # Number of generations
  epochs_per_eval: 10       # Quick epochs for fitness
  final_epochs: 200         # Full training for best model
```

#### Step 2: Launch Training

**Linux:**
```bash
chmod +x start_ga_training.sh
./start_ga_training.sh
```

**Or manually:**
```bash
python -c "
from training.multi_gpu_genetic_trainer import MultiGPUGeneticTrainer
import yaml

with open('config/config_cl_optimized.yaml') as f:
    config = yaml.safe_load(f)

trainer = MultiGPUGeneticTrainer(config, data_path='data/csv')
best_config, best_fitness = trainer.run(
    generations=100,
    epochs_per_eval=10
)
print(f'Best fitness: {best_fitness}')
"
```

#### Step 3: Monitor Progress

Training creates logs in `logs/`:
```bash
tail -f logs/training.log
```

**Or view with TensorBoard:**
```bash
tensorboard --logdir logs/
```

### Remote Training

For training on a remote server with multiple GPUs:

#### Step 1: Configure Remote Machine

Edit `launch_remote_training.py`:

```python
REMOTE_HOST = "192.168.0.129"  # Your server IP
REMOTE_USER = "alex"           # SSH username
REMOTE_PROJECT_PATH = "/home/alex/nq_speed"
```

#### Step 2: Launch Remote Training

```bash
python launch_remote_training.py
```

This will:
1. SSH to the remote machine
2. Sync the latest code
3. Start multi-GPU GA training
4. Stream logs back to your terminal

#### GPU Memory Management

The trainer automatically adjusts batch sizes based on GPU memory:

| GPU Memory | Max Batch Size (small model) | Max Batch Size (large model) |
|------------|------------------------------|------------------------------|
| 32GB+ (RTX 5090) | 384 | 192 |
| 22-32GB (RTX 3090) | 128 | 64 |
| 10-22GB (RTX 3080) | 64 | 32 |

---

## Backtesting

### Run Backtest

```bash
python main.py --mode backtest --model checkpoints/best_model.pt --data data/csv
```

**Or directly:**

```bash
python backtest.py \
    --model checkpoints/best_model.pt \
    --data data/csv \
    --config config/config.yaml \
    --initial-capital 100000 \
    --commission 2.25 \
    --slippage 0.25
```

### Backtest Output

The backtest produces:
- Trade-by-trade log
- Equity curve plot
- Performance metrics:
  - Total Return
  - Sharpe Ratio
  - Calmar Ratio
  - Maximum Drawdown
  - Win Rate
  - Profit Factor
  - Total Trades

### Example Output

```
=== Backtest Results ===
Period: 2024-01-01 to 2024-12-31
Initial Capital: $100,000.00
Final Capital: $127,450.00
Total Return: 27.45%
Sharpe Ratio: 1.85
Calmar Ratio: 2.41
Max Drawdown: 11.38%
Win Rate: 54.2%
Profit Factor: 1.67
Total Trades: 342
Avg Trade Duration: 45 minutes
```

---

## Paper Trading

### Setup

1. **No broker connection needed** - simulates execution locally
2. Configure initial capital in `config/config.yaml`:

```yaml
backtest:
  initial_capital: 100000
  commission: 2.25
  slippage: 0.25
```

### Run Paper Trading

```bash
python main.py --mode paper --model checkpoints/best_model.pt
```

### Live Dashboard

Run the trading dashboard for real-time monitoring:

```bash
python dashboard.py --model checkpoints/best_model.pt
```

Access at `http://localhost:8050`

---

## Live Trading

**WARNING: Use at your own risk. Start with paper trading first.**

### Prerequisites

1. **Interactive Brokers Account** with:
   - TWS or IB Gateway installed
   - API connections enabled
   - Futures trading permissions

2. **IB Gateway/TWS Configuration:**
   - API port: 4002 (paper) or 4001 (live)
   - Enable "Read-Only API": OFF
   - Enable "Download open orders on connection": ON

### Configure Connection

Edit `config/config.yaml`:

```yaml
trading:
  ib_host: "127.0.0.1"
  ib_port: 4001          # 4001 for live, 4002 for paper
  client_id: 1
  paper_trading: false   # Set to false for live
  symbol: "NQ"
  exchange: "CME"
  max_position: 2
  risk_per_trade: 0.02
  max_daily_loss: 0.05
```

### Run Live Trading

```bash
python main.py --mode live --model checkpoints/best_model.pt --config config/config.yaml
```

### Monitoring

```bash
python trading_monitor.py --config config/config.yaml
```

---

## Model Architecture Details

### Transformer Agent

Processes market sequences using multi-head self-attention:

```
Input: (batch, seq_len, features) → Embedding → Positional Encoding →
       N × TransformerEncoderLayer → Output: (batch, d_model)
```

**Key hyperparameters:**
- `d_model`: Embedding dimension (64-512)
- `n_heads`: Attention heads (2-16)
- `n_layers`: Encoder layers (2-8)
- `dropout`: Regularization (0.05-0.3)

### Profit Maximizer Agent

Actor-critic architecture for position sizing:

```
Market Embedding → Actor Network → Position Size (continuous)
                → Critic Network → Value Estimate
```

Uses Gaussian policy with learned variance.

### Risk Controller Agent

Computes dynamic risk parameters:

```
Market Embedding → Risk Network → Stop Distance (ATR multiple)
                               → Target Distance
                               → Risk Multiplier
```

### Duration Agent

Classifies expected holding period:

```
Market Embedding → Duration Network → Softmax → {Scalp, Intraday, Swing}
```

### Meta-Agent Attention

Combines sub-agent outputs using attention:

```
Agent Outputs → Multi-Head Attention → Weighted Combination → Final Decision
```

---

## Genetic Algorithm Optimization

### Island-based GA

The system uses an island model for parallel evolution:

```
  Island 0        Island 1        Island 2        Island 3
  ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
  │Pop 0 │◄──────►│Pop 1 │◄──────►│Pop 2 │◄──────►│Pop 3 │
  │GPU 0 │        │GPU 1 │        │GPU 2 │        │GPU 3 │
  └──────┘        └──────┘        └──────┘        └──────┘
      │               │               │               │
      └───────────────┴───────┬───────┴───────────────┘
                              │
                    Migration every N generations
```

### Search Space

| Hyperparameter | Range | Type |
|----------------|-------|------|
| d_model | [64, 128, 192, 256, 384, 512] | Categorical |
| n_layers | [2, 3, 4, 5, 6, 8] | Categorical |
| n_heads | [2, 4, 8] | Categorical |
| dropout | [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] | Categorical |
| learning_rate | [1e-5, 1e-3] | Log-uniform |
| batch_size | [64, 128, 192, 256, 384] | Categorical |
| weight_decay | [1e-4, 1e-1] | Log-uniform |

### Fitness Function

```
Fitness = TradeBonus + 0.4×Sharpe + 0.3×Calmar + 0.2×ProfitFactor + 0.1×WinRate
          - LowTradePenalty - HighDrawdownPenalty
```

Where:
- **TradeBonus**: Encourages trading activity (0-0.3)
- **Sharpe**: Risk-adjusted return (-1 to 1)
- **Calmar**: Return / Max Drawdown (-1 to 1)
- **LowTradePenalty**: Applied if trades < 20
- **HighDrawdownPenalty**: Applied if drawdown > 15%

---

## Loss Function Design

The training uses a **multi-component loss** to prevent mode collapse:

### Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| Direction Classification | 30% | Classify UP/NEUTRAL/DOWN |
| Directional Accuracy | 20% | Reward correct sign predictions |
| Signal Variation | 20% | **Prevent constant outputs** |
| Saturation Penalty | 10% | Prevent outputs stuck at ±1 |
| Position Change | 10% | Encourage varying positions |
| Value Loss | 5% | Critic estimation |
| Entropy | 3% | Exploration |
| Action Diversity | 2% | Balanced action distribution |

### Signal Variation Penalty

The key innovation preventing mode collapse:

```python
position_std = position_outputs.std()
target_std = 0.3  # Target standard deviation
variation_penalty = max(0, target_std - position_std) ** 2
```

This ensures the model outputs vary rather than converging to constant values.

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size in config
- The trainer automatically adjusts batch size per GPU memory
- For large models, use smaller d_model and n_layers

#### 2. Mode Collapse (Constant Outputs)

**Symptoms:**
- Signal std ≈ 0.0001
- All outputs near 0.99 or -0.99
- Zero trades in backtest

**Solution:**
The signal variation loss is already implemented. If still occurring:
- Increase `variation_penalty` weight in `trainer.py`
- Check that targets have sufficient variation
- Ensure balanced class distribution

#### 3. NaN Loss Values

**Error:**
```
Loss is NaN
```

**Solution:**
- Check for extreme values in data
- Reduce learning rate
- Enable gradient clipping
- Check Calmar ratio calculation for zero drawdown

#### 4. No GPU Detected

**Error:**
```
CUDA available: False
```

**Solution:**
- Install CUDA toolkit
- Reinstall PyTorch with CUDA support
- Check NVIDIA drivers: `nvidia-smi`

#### 5. DataParallel Errors on Mixed GPUs

**Error:**
```
CUBLAS_STATUS_EXECUTION_FAILED when calling cublasSgemmStridedBatched
```

**Solution:**
The trainer uses single GPU for final training when GPUs are different. For evaluation:
- Each island uses a single GPU
- Mixed GPU types are supported through batch size adjustment

#### 6. Connection Refused (IB Gateway)

**Error:**
```
Connection refused to 127.0.0.1:4002
```

**Solution:**
1. Ensure IB Gateway/TWS is running
2. Enable API in configuration:
   - File → Global Configuration → API → Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Check "Socket port" matches config
3. Disable firewall for the port

### Getting Help

1. Check logs in `logs/training.log`
2. Enable debug logging:
   ```bash
   python main.py --mode train --log-level DEBUG
   ```
3. Open an issue at https://github.com/spawnaga/trainingNQtrading/issues

---

## References

1. Kim, K., & Han, I. (2000). Genetic algorithms approach to feature discretization in artificial neural networks for the prediction of stock price index. *Expert Systems with Applications*.

2. Ding, Q., et al. (2020). Hierarchical multi-scale Gaussian transformer for stock movement prediction. *IJCAI*.

3. Wang, X., et al. (2025). A Transformer-Based Reinforcement Learning Framework for Sequential Strategy Optimization. *MDPI Applied Sciences*.

4. Chen, Y., et al. (2025). A Self-Rewarding Mechanism in Deep Reinforcement Learning for Trading Strategy Optimization. *MDPI Mathematics*.

5. ScienceDirect (2024). Deep reinforcement learning applied to a sparse-reward trading environment with intraday data.

See `paper/meta_agent_ga_approach.md` for detailed algorithm documentation.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Important Notice

- This system is for **educational and research purposes only**
- **DO NOT** use this for actual trading with real money without extensive testing
- Past performance does not guarantee future results
- Trading futures involves substantial risk of loss
- The authors assume no responsibility for any financial losses

---

*Maintained by Alex Oraibi*
