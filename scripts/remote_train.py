#!/usr/bin/env python3
"""
Remote Training Helper Script

Syncs code to remote machine and runs training via SSH.

Usage:
    python scripts/remote_train.py --sync          # Sync files only
    python scripts/remote_train.py --train         # Sync and start training
    python scripts/remote_train.py --status        # Check GPU status
    python scripts/remote_train.py --logs          # Tail training logs
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Remote machine configuration
REMOTE_HOST = "alex@192.168.0.129"
REMOTE_DIR = "/home/alex/nq_trading_system"
LOCAL_DIR = Path(__file__).parent.parent
CUDA_DEVICES = "0,2"  # GPUs to use on remote machine
CONDA_ENV = "openHands"  # Conda environment on remote machine

# Files/folders to exclude from sync
EXCLUDE_PATTERNS = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "*.egg-info",
    ".venv",
    "venv",
    "checkpoints/*.pt",  # Don't sync large model files
    "logs",
    ".idea",
]


def run_cmd(cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f">>> {cmd}")
    if capture:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return subprocess.run(cmd, shell=True)


def sync_to_remote():
    """Sync local code to remote machine using rsync."""
    exclude_args = " ".join(f'--exclude="{p}"' for p in EXCLUDE_PATTERNS)

    # Use rsync for efficient sync (requires rsync on Windows via WSL or Git Bash)
    cmd = f'rsync -avz --progress {exclude_args} "{LOCAL_DIR}/" "{REMOTE_HOST}:{REMOTE_DIR}/"'

    print(f"\n🔄 Syncing to {REMOTE_HOST}...")
    result = run_cmd(cmd)

    if result.returncode == 0:
        print("✅ Sync completed successfully!")
    else:
        print("❌ Sync failed. Make sure rsync is installed.")
        print("   On Windows, install via: winget install rsync")
        print("   Or use Git Bash which includes rsync")

    return result.returncode == 0


def sync_to_remote_scp():
    """Alternative sync using scp (if rsync not available)."""
    print(f"\n🔄 Syncing to {REMOTE_HOST} using scp...")

    # Create remote directory
    run_cmd(f'ssh {REMOTE_HOST} "mkdir -p {REMOTE_DIR}"')

    # Copy important directories
    dirs_to_sync = ["config", "data_pipeline", "models", "training", "trading", "utils", "scripts"]
    files_to_sync = ["train.py", "backtest.py", "main.py", "requirements.txt"]

    for d in dirs_to_sync:
        local_path = LOCAL_DIR / d
        if local_path.exists():
            run_cmd(f'scp -r "{local_path}" "{REMOTE_HOST}:{REMOTE_DIR}/"')

    for f in files_to_sync:
        local_path = LOCAL_DIR / f
        if local_path.exists():
            run_cmd(f'scp "{local_path}" "{REMOTE_HOST}:{REMOTE_DIR}/"')

    print("✅ Sync completed!")
    return True


def check_gpu_status():
    """Check GPU status on remote machine."""
    print(f"\n🖥️  GPU Status on {REMOTE_HOST}:")
    run_cmd(f'ssh {REMOTE_HOST} "nvidia-smi"')


def install_dependencies():
    """Install Python dependencies on remote machine."""
    print(f"\n📦 Installing dependencies on {REMOTE_HOST}...")
    print(f"   Using conda environment: {CONDA_ENV}")
    run_cmd(f'ssh {REMOTE_HOST} "source ~/.bashrc && conda activate {CONDA_ENV} && cd {REMOTE_DIR} && pip install -r requirements.txt"')


def start_training(config: str = "config/config.yaml", epochs: int = None, resume: str = None):
    """Start training on remote machine."""
    print(f"\n🚀 Starting training on {REMOTE_HOST}...")
    print(f"   Using GPUs: {CUDA_DEVICES}")
    print(f"   Using conda env: {CONDA_ENV}")

    # Build training command with conda activation and CUDA_VISIBLE_DEVICES
    cmd_parts = ["python train.py"]

    if config:
        cmd_parts.extend(["--config", config])
    if epochs:
        cmd_parts.extend(["--epochs", str(epochs)])
    if resume:
        cmd_parts.extend(["--resume", resume])

    train_cmd = " ".join(cmd_parts)

    # Full command with conda activation and nohup
    remote_cmd = f'source ~/.bashrc && conda activate {CONDA_ENV} && cd {REMOTE_DIR} && CUDA_VISIBLE_DEVICES={CUDA_DEVICES} nohup {train_cmd} > training.log 2>&1 &'

    run_cmd(f"ssh {REMOTE_HOST} '{remote_cmd}'")
    print("✅ Training started in background!")
    print(f"   View logs: python scripts/remote_train.py --logs")
    print(f"   Check GPU: python scripts/remote_train.py --status")


def start_training_interactive(config: str = "config/config.yaml"):
    """Start training interactively (stays connected)."""
    print(f"\n🚀 Starting interactive training on {REMOTE_HOST}...")
    print(f"   Using GPUs: {CUDA_DEVICES}")
    print(f"   Using conda env: {CONDA_ENV}")
    cmd = f'ssh -t {REMOTE_HOST} "source ~/.bashrc && conda activate {CONDA_ENV} && cd {REMOTE_DIR} && CUDA_VISIBLE_DEVICES={CUDA_DEVICES} python train.py --config {config}"'
    run_cmd(cmd)


def tail_logs():
    """Tail training logs on remote machine."""
    print(f"\n📜 Tailing logs from {REMOTE_HOST} (Ctrl+C to stop)...")
    run_cmd(f'ssh {REMOTE_HOST} "tail -f {REMOTE_DIR}/training.log"')


def check_training_status():
    """Check if training is running."""
    print(f"\n🔍 Checking training status on {REMOTE_HOST}...")
    result = run_cmd(f'ssh {REMOTE_HOST} "pgrep -f \'python train.py\' && echo \'Training is running\' || echo \'No training process found\'"')


def download_checkpoints():
    """Download checkpoints from remote machine."""
    print(f"\n📥 Downloading checkpoints from {REMOTE_HOST}...")
    local_checkpoint_dir = LOCAL_DIR / "checkpoints"
    local_checkpoint_dir.mkdir(exist_ok=True)

    run_cmd(f'scp -r "{REMOTE_HOST}:{REMOTE_DIR}/checkpoints/*" "{local_checkpoint_dir}/"')
    print(f"✅ Checkpoints downloaded to {local_checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Remote training helper")
    parser.add_argument("--sync", action="store_true", help="Sync code to remote")
    parser.add_argument("--sync-scp", action="store_true", help="Sync using scp (fallback)")
    parser.add_argument("--train", action="store_true", help="Start training (background)")
    parser.add_argument("--train-interactive", action="store_true", help="Start training (interactive)")
    parser.add_argument("--status", action="store_true", help="Check GPU status")
    parser.add_argument("--logs", action="store_true", help="Tail training logs")
    parser.add_argument("--check", action="store_true", help="Check if training is running")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--download", action="store_true", help="Download checkpoints")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--config", default="config/config.yaml", help="Config file")

    args = parser.parse_args()

    if args.sync:
        sync_to_remote()
    elif args.sync_scp:
        sync_to_remote_scp()
    elif args.status:
        check_gpu_status()
    elif args.install:
        install_dependencies()
    elif args.train:
        sync_to_remote()
        start_training(args.config, args.epochs)
    elif args.train_interactive:
        sync_to_remote()
        start_training_interactive(args.config)
    elif args.logs:
        tail_logs()
    elif args.check:
        check_training_status()
    elif args.download:
        download_checkpoints()
    else:
        parser.print_help()
        print("\n📋 Quick Start:")
        print("  1. python scripts/remote_train.py --status      # Check GPUs")
        print("  2. python scripts/remote_train.py --sync        # Sync code")
        print("  3. python scripts/remote_train.py --install     # Install deps")
        print("  4. python scripts/remote_train.py --train       # Start training")
        print("  5. python scripts/remote_train.py --logs        # View logs")


if __name__ == "__main__":
    main()
