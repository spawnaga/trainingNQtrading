#!/bin/bash
# Multi-GPU Genetic Algorithm Training Script
# For 2x RTX 5090 + 2x RTX 3090 (NVLinked)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Training parameters (can be overridden via environment)
N_ISLANDS=${N_ISLANDS:-4}
POP_PER_ISLAND=${POP_PER_ISLAND:-20}
GENERATIONS=${GENERATIONS:-100}
EPOCHS_PER_EVAL=${EPOCHS_PER_EVAL:-10}
FINAL_EPOCHS=${FINAL_EPOCHS:-200}
SEED=${SEED:-42}

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/ga_optimized_${TIMESTAMP}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Multi-GPU Genetic Algorithm Training${NC}"
echo -e "${BLUE}============================================${NC}"

# Check CUDA availability
echo -e "\n${YELLOW}Checking GPU configuration...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo ""
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}Found $GPU_COUNT GPUs${NC}"
else
    echo -e "${RED}Warning: nvidia-smi not found${NC}"
    GPU_COUNT=0
fi

# Setup Python environment
echo -e "\n${YELLOW}Setting up Python environment...${NC}"
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}Activated existing virtual environment${NC}"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}Activated existing virtual environment${NC}"
else
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    echo -e "${GREEN}Created and activated virtual environment${NC}"
fi

# Install dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"
pip install --quiet --upgrade pip

# Check if PyTorch with CUDA is installed
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# Install other requirements
pip install --quiet -r requirements.txt

# Verify PyTorch CUDA
echo -e "\n${YELLOW}Verifying PyTorch CUDA...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory: {props.total_memory / 1024**3:.1f} GB')
"

# Display configuration
echo -e "\n${BLUE}============================================${NC}"
echo -e "${BLUE}  Training Configuration${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Islands: ${GREEN}$N_ISLANDS${NC}"
echo -e "Population per island: ${GREEN}$POP_PER_ISLAND${NC}"
echo -e "Total population: ${GREEN}$((N_ISLANDS * POP_PER_ISLAND))${NC}"
echo -e "Generations: ${GREEN}$GENERATIONS${NC}"
echo -e "Epochs per evaluation: ${GREEN}$EPOCHS_PER_EVAL${NC}"
echo -e "Final training epochs: ${GREEN}$FINAL_EPOCHS${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Log file: ${GREEN}$LOG_FILE${NC}"

# Check data exists
echo -e "\n${YELLOW}Checking data...${NC}"
if [ -d "data/csv" ] && [ "$(ls -A data/csv/*.csv 2>/dev/null)" ]; then
    CSV_COUNT=$(ls data/csv/*.csv 2>/dev/null | wc -l)
    echo -e "${GREEN}Found $CSV_COUNT CSV files in data/csv/${NC}"
    ls -la data/csv/*.csv | head -5
else
    echo -e "${RED}Error: No CSV files found in data/csv/${NC}"
    exit 1
fi

# Confirm start
echo -e "\n${YELLOW}Ready to start training.${NC}"
read -p "Press Enter to start (or Ctrl+C to cancel)..."

# Start training
echo -e "\n${GREEN}Starting training...${NC}"
echo -e "Logging to: $LOG_FILE"
echo -e "Use 'tail -f $LOG_FILE' to monitor progress"
echo ""

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training
python -u training/multi_gpu_genetic_trainer.py \
    --data data/csv \
    --config config/config_cl_optimized.yaml \
    --output "$OUTPUT_DIR" \
    --n-islands $N_ISLANDS \
    --pop-per-island $POP_PER_ISLAND \
    --generations $GENERATIONS \
    --epochs-per-eval $EPOCHS_PER_EVAL \
    --final-epochs $FINAL_EPOCHS \
    --seed $SEED \
    2>&1 | tee "$LOG_FILE"

# Training complete
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Results saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "Log file: ${BLUE}$LOG_FILE${NC}"

# Show best results
if [ -f "$OUTPUT_DIR/ga_results.json" ]; then
    echo -e "\n${YELLOW}Best Configuration:${NC}"
    python -c "
import json
with open('$OUTPUT_DIR/ga_results.json') as f:
    results = json.load(f)
print(f'Best Fitness: {results[\"best_fitness\"]:.4f}')
print('Best Genes:')
for k, v in results['best_genes'].items():
    print(f'  {k}: {v}')
"
fi

# Show final model
if [ -f "$OUTPUT_DIR/best_model_ga_optimized.pt" ]; then
    echo -e "\n${GREEN}Final model saved: $OUTPUT_DIR/best_model_ga_optimized.pt${NC}"
fi
