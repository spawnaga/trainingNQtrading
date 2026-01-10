#!/bin/bash
# Multi-GPU Training Script for 2x RTX 5090
# Run on alex@192.168.0.129

# Activate conda environment
source ~/.bashrc
conda activate openHands

# Set CUDA devices (using GPUs 0 and 2)
export CUDA_VISIBLE_DEVICES=0,2

# Navigate to project directory
cd /home/alex/nq_trading_system

# Check GPU status
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    exit 0
fi

# Run training with PyTorch DDP (Distributed Data Parallel)
# Using torchrun for multi-GPU training
echo "🚀 Starting Multi-GPU Training with 2x RTX 5090..."
echo "   Config: config/config_remote.yaml"
echo ""

# Option 1: Using torchrun (recommended for PyTorch 2.x)
torchrun --nproc_per_node=2 \
         --master_port=29500 \
         train.py \
         --config config/config_remote.yaml \
         --distributed \
         "$@"

# Option 2: Alternative using python -m torch.distributed.launch (legacy)
# python -m torch.distributed.launch --nproc_per_node=2 train.py --config config/config_remote.yaml --distributed "$@"
