# Meta-Agent Architecture with Genetic Algorithm Optimization

**Alex Oraibi**

*January 2026*

---

## Abstract

This document describes the Meta-Agent trading architecture combined with Island-based Genetic Algorithm (GA) optimization for hyperparameter tuning. We address critical challenges in training deep learning trading systems, particularly the "mode collapse" problem where models output constant predictions. Our solution incorporates novel loss function components that encourage signal variation and proper trading behavior.

**Keywords:** Meta-Agent, Genetic Algorithm, Transformer, Trading, Hyperparameter Optimization, Mode Collapse

---

## 1. Introduction

### 1.1 Problem Statement

Training neural networks for trading poses unique challenges:

1. **Mode Collapse**: Models tend to output constant values that optimize average return but produce no trades
2. **Hyperparameter Sensitivity**: Performance varies dramatically with architecture choices
3. **Computational Cost**: Full training runs are expensive, making grid search impractical

### 1.2 Our Approach

We combine:
- **Meta-Agent Architecture**: Modular design with specialized sub-agents
- **Island-based Genetic Algorithm**: Parallel hyperparameter optimization
- **Signal Variation Loss**: Novel loss components to prevent mode collapse

---

## 2. Related Work

### 2.1 Genetic Algorithms for Trading

Genetic algorithms have proven effective for trading strategy optimization:

- **Kim & Han (2000)**: Pioneered GA-based feature selection for neural network trading [1]
- **Ghandar et al. (2023)**: Trading Strategy Hyper-parameter Optimization using Genetic Algorithm [2]
- **Learning-Gate (2025)**: Comprehensive review of GA-optimized neural network trading systems [3]

### 2.2 Transformers for Financial Prediction

Recent advances in transformer-based trading:

- **Ding et al. (2020)**: Hierarchical multi-scale Gaussian transformer for stock prediction [4]
- **Arxiv (2025)**: Transformer-Based Reinforcement Learning for Sequential Strategy Optimization [5]
- **MDPI (2025)**: Self-Rewarding Mechanism in Deep RL for Trading [6]

### 2.3 Mode Collapse in Trading Models

The problem of constant outputs is well-documented:

> "Some DRL-based trading systems face limitations that often manifest as constant feedback signals... This stems from the intricate exploration and complex reward adjustment procedures required to steer the algorithm towards a profitable policy." [7]

**Key insight**: Standard regression loss (minimizing MSE on returns) encourages constant bias rather than varying signals.

---

## 3. Meta-Agent Architecture

### 3.1 Overview

The Meta-Agent orchestrates four specialized sub-agents:

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

### 3.2 Component Details

#### 3.2.1 Transformer Agent

Processes sequential market data using multi-head self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Architecture:**
- Input: $(B, T, d_{input})$ where $T=60$ time steps
- Embedding: Linear projection to $d_{model}$
- Encoder: $L$ transformer layers with $H$ attention heads
- Output: Market embedding $(B, d_{model})$

#### 3.2.2 Profit Maximizer

Determines position sizing using actor-critic approach:

$$\pi(a|s) = \mathcal{N}(\mu(s), \sigma(s))$$

where $\mu(s)$ is the mean action and $\sigma(s)$ is learned standard deviation.

#### 3.2.3 Risk Controller

Computes dynamic risk parameters:
- Risk multiplier based on regime
- Stop-loss distance (ATR-based)
- Take-profit distance

$$\text{stop\_distance} = \text{ATR} \times \text{multiplier}$$

#### 3.2.4 Duration Agent

Classifies expected holding periods:
- Short-term (< 15 min)
- Medium-term (15-60 min)
- Long-term (> 60 min)

### 3.3 Agent Attention

Multi-head attention combines sub-agent outputs:

$$\text{Combined} = \text{Attention}([E_{market}, E_{profit}, E_{risk}, E_{duration}])$$

---

## 4. Genetic Algorithm Optimization

### 4.1 Island-Based GA

We use an island model for parallel evolution:

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

**Benefits:**
- Maintains population diversity
- Parallel evaluation on multiple GPUs
- Prevents premature convergence

### 4.2 Search Space

| Hyperparameter | Range | Type |
|----------------|-------|------|
| d_model | [64, 128, 192, 256, 384, 512] | Categorical |
| n_layers | [2, 3, 4, 5, 6, 8] | Categorical |
| n_heads | [2, 4, 8] | Categorical |
| dropout | [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] | Categorical |
| learning_rate | [1e-5, 1e-3] | Log-uniform |
| batch_size | [64, 128, 192, 256, 384] | Categorical |
| weight_decay | [1e-4, 1e-1] | Log-uniform |

### 4.3 Fitness Function

The fitness function balances multiple objectives:

$$F = F_{trade} + w_1 \cdot \text{Sharpe} + w_2 \cdot \text{Calmar} + w_3 \cdot \text{PF} + w_4 \cdot \text{WR} - P_{trades} - P_{dd}$$

where:
- $F_{trade}$: Trade activity bonus (encourages trading)
- Sharpe: Risk-adjusted return
- Calmar: Return / Max Drawdown
- PF: Profit Factor
- WR: Win Rate
- $P_{trades}$: Penalty for too few trades
- $P_{dd}$: Penalty for excessive drawdown

---

## 5. Signal Variation Loss (Key Contribution)

### 5.1 The Mode Collapse Problem

**Problem:** Standard regression loss optimizes for average return:

$$\mathcal{L}_{return} = -\mathbb{E}[p_i \cdot r_i]$$

This encourages the model to learn a constant bias (always long or short) rather than varying signals.

**Evidence:**
```
Before fix: Signal stats: min=0.9995, max=0.9997, std=0.0001
            Fitness: 0.0000, Trades: 0

After fix:  Signal stats: min=-0.67, max=0.66, std=0.33
            Fitness: 0.0024, Trades: 47
```

### 5.2 Our Solution: Multi-Component Loss

We design a loss function that explicitly encourages signal variation:

#### 5.2.1 Direction Classification Loss (30%)

$$\mathcal{L}_{dir} = \text{CrossEntropy}(\hat{y}, y_{direction})$$

with class weights to handle imbalance:
- UP: weight = 1.5
- NEUTRAL: weight = 0.5
- DOWN: weight = 1.5

#### 5.2.2 Signal Variation Penalty (20%)

$$\mathcal{L}_{var} = \max(0, \sigma_{target} - \sigma_{outputs})^2$$

where $\sigma_{target} = 0.3$ ensures sufficient signal variation within each batch.

#### 5.2.3 Saturation Penalty (10%)

$$\mathcal{L}_{sat} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[|p_i| > 0.95]$$

Penalizes outputs stuck at ±1.

#### 5.2.4 Position Change Encouragement (10%)

$$\mathcal{L}_{change} = \mathbb{E}[\mathbf{1}[|r_i - r_{i-1}| > \tau] \cdot \max(0, \delta - |p_i - p_{i-1}|)]$$

If consecutive targets differ significantly, positions should also differ.

#### 5.2.5 Directional Accuracy (20%)

$$\mathcal{L}_{acc} = -\mathbb{E}[\text{sign}(p_i) \cdot \text{sign}(r_i) \cdot |r_i|]$$

Rewards correct direction weighted by magnitude.

### 5.3 Complete Loss Function

$$\mathcal{L}_{total} = \sum_{k} w_k \cdot \mathcal{L}_k$$

| Component | Weight | Purpose |
|-----------|--------|---------|
| Direction | 0.30 | Classify market direction |
| Accuracy | 0.20 | Reward correct directional calls |
| Variation | 0.20 | **Prevent constant outputs** |
| Saturation | 0.10 | Prevent output saturation |
| Change | 0.10 | Encourage position changes |
| Value | 0.05 | Critic estimation |
| Entropy | 0.03 | Exploration |
| Action Diversity | 0.02 | Balanced action distribution |

---

## 6. GPU Memory-Aware Training

### 6.1 Challenge

Training on heterogeneous GPUs (RTX 5090 32GB + RTX 3090 24GB) causes OOM errors when batch sizes are too large for smaller GPUs.

### 6.2 Solution

Dynamic batch size adjustment based on GPU memory:

```python
def adjust_batch_size_for_gpu(batch_size, gpu_memory_gb, model_factor):
    if gpu_memory_gb >= 30:  # RTX 5090
        max_batch = 384 if model_factor < 1.5 else 192
    elif gpu_memory_gb >= 22:  # RTX 3090
        if model_factor > 1.5:
            max_batch = 64
        elif model_factor > 1.0:
            max_batch = 96
        else:
            max_batch = 128
    return min(batch_size, max_batch)
```

**Model factor** estimates memory usage:
$$\text{model\_factor} = \frac{d_{model}}{256} \times \frac{n_{layers}}{4} \times \frac{d_{ff}}{1024}$$

---

## 7. Experimental Results

### 7.1 Before vs After Signal Variation Loss

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Signal Std | 0.0001 | 0.33 |
| Signal Range | [0.999, 1.0] | [-0.67, +0.66] |
| Total Trades | 0 | 47+ |
| Fitness | 0.0000 | 0.0024 |
| Model Status | Mode Collapse | **Working** |

### 7.2 GA Optimization Progress

| Generation | Best Fitness | Avg Fitness | Improvement |
|------------|--------------|-------------|-------------|
| 0 | -0.2475 | -0.3551 | Baseline |
| 1 | **0.0024** | -0.3148 | +12.8% avg |

### 7.3 Training Configuration

- 4 GPUs: 2× RTX 5090 (32GB) + 2× RTX 3090 (24GB NVLinked)
- Islands: 4
- Population per island: 2-3
- Epochs per evaluation: 3-5
- Data: 100K rows (quick test)

---

## 8. References

[1] Kim, K., & Han, I. (2000). Genetic algorithms approach to feature discretization in artificial neural networks for the prediction of stock price index. *Expert Systems with Applications*.

[2] Ghandar, A., et al. (2023). Trading Strategy Hyper-parameter Optimization using Genetic Algorithm. *IEEE Conference*.

[3] Learning-Gate (2025). Optimizing trading strategies using genetic algorithms. https://learning-gate.com/

[4] Ding, Q., et al. (2020). Hierarchical multi-scale Gaussian transformer for stock movement prediction. *IJCAI*.

[5] Wang, X., et al. (2025). A Transformer-Based Reinforcement Learning Framework for Sequential Strategy Optimization in Sparse Data. *MDPI Applied Sciences*.

[6] Chen, Y., et al. (2025). A Self-Rewarding Mechanism in Deep Reinforcement Learning for Trading Strategy Optimization. *MDPI Mathematics*.

[7] ScienceDirect (2024). Deep reinforcement learning applied to a sparse-reward trading environment with intraday data.

[8] Medium (2025). Using Reinforcement Learning to Optimize Stock Trading Strategies.

[9] Arxiv (2025). Risk-Aware Reinforcement Learning Reward for Financial Trading.

---

## Appendix A: Loss Function Implementation

```python
def _compute_loss(self, outputs, targets, position_state):
    position_size = outputs['position_size']

    # 1. Direction Classification (30%)
    target_direction = classify_direction(targets, threshold=0.0005)
    direction_loss = CrossEntropyLoss(weight=[1.5, 0.5, 1.5])(
        outputs['action_probs'][:, :3], target_direction
    )

    # 2. Signal Variation Penalty (20%)
    position_std = position_size.std()
    variation_penalty = relu(0.3 - position_std) ** 2

    # 3. Saturation Penalty (10%)
    saturation = (abs(position_size) > 0.95).float().mean()

    # 4. Directional Accuracy (20%)
    direction_match = sign(position_size) * sign(targets)
    accuracy_loss = -(direction_match * abs(targets) * 100).mean()

    # 5. Position Change Encouragement (10%)
    position_diff = abs(position_size[1:] - position_size[:-1])
    target_diff = abs(targets[1:] - targets[:-1])
    should_differ = (target_diff > 0.001).float()
    change_penalty = (should_differ * relu(0.1 - position_diff)).mean()

    # Combined loss
    total_loss = (
        0.30 * direction_loss +
        0.20 * accuracy_loss +
        0.20 * variation_penalty +
        0.10 * saturation +
        0.10 * change_penalty +
        0.05 * value_loss +
        0.03 * entropy_loss +
        0.02 * action_diversity_loss
    )

    return total_loss
```

## Appendix B: Fitness Evaluation

```python
def calculate_fitness(metrics):
    if metrics.total_trades == 0:
        return -0.5  # Penalize but don't zero out

    # Trade activity bonus
    trade_bonus = min(metrics.total_trades / 50.0, 1.0) * 0.3

    # Performance metrics
    sharpe_score = clip(metrics.sharpe_ratio / 2.0, -1, 1)
    calmar_score = clip(metrics.calmar_ratio / 3.0, -1, 1)
    pf_score = clip(metrics.profit_factor / 2.0, 0, 1)
    wr_score = clip(metrics.win_rate / 0.55, 0, 1.2)

    fitness = (
        trade_bonus +
        0.4 * sharpe_score +
        0.3 * calmar_score +
        0.2 * pf_score +
        0.1 * wr_score
    )

    # Gradual penalties (not multiplicative)
    if metrics.total_trades < 20:
        fitness -= 0.2 * (1.0 - metrics.total_trades / 20.0)

    if metrics.max_drawdown > 0.15:
        fitness -= 0.3 * min((metrics.max_drawdown - 0.15) / 0.35, 1.0)

    return fitness
```

---

*Correspondence: Alex Oraibi*
