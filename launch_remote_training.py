#!/usr/bin/env python
"""
Remote Training Launcher for Multi-GPU Genetic Algorithm Optimization.

This script:
1. Syncs code to the remote machine (192.168.0.129)
2. Launches the multi-GPU genetic algorithm training
3. Monitors progress and retrieves results

Usage:
    python launch_remote_training.py --host 192.168.0.129 --user <username>
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"[CMD] {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture_output,
        text=True,
        check=check
    )
    return result


def sync_to_remote(local_path: str, remote_host: str, remote_user: str, remote_path: str):
    """Sync local directory to remote machine using rsync or scp."""
    print(f"\n{'='*50}")
    print("Syncing code to remote machine...")
    print(f"{'='*50}")

    # Try rsync first, fall back to scp
    try:
        # Use rsync for efficient sync
        cmd = f'rsync -avz --progress --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" --exclude=".venv" --exclude="checkpoints/*.pt" "{local_path}/" {remote_user}@{remote_host}:{remote_path}/'
        run_command(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("rsync not available, using scp...")
        # Fallback to scp
        cmd = f'scp -r "{local_path}" {remote_user}@{remote_host}:{remote_path}'
        run_command(cmd)


def sync_data_to_remote(local_data: str, remote_host: str, remote_user: str, remote_path: str):
    """Sync data files to remote."""
    print(f"\n{'='*50}")
    print("Syncing data to remote machine...")
    print(f"{'='*50}")

    remote_data_path = f"{remote_path}/data"

    # Create remote data directory
    cmd = f'ssh {remote_user}@{remote_host} "mkdir -p {remote_data_path}"'
    run_command(cmd)

    # Sync data
    try:
        cmd = f'rsync -avz --progress "{local_data}/" {remote_user}@{remote_host}:{remote_data_path}/'
        run_command(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        cmd = f'scp -r "{local_data}" {remote_user}@{remote_host}:{remote_data_path}'
        run_command(cmd)


def setup_remote_environment(remote_host: str, remote_user: str, remote_path: str):
    """Setup Python environment on remote machine."""
    print(f"\n{'='*50}")
    print("Setting up remote environment...")
    print(f"{'='*50}")

    setup_commands = f'''
cd {remote_path} && \\
python -m venv .venv 2>/dev/null || true && \\
source .venv/bin/activate && \\
pip install --upgrade pip && \\
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \\
pip install -r requirements.txt 2>/dev/null || pip install numpy pandas scikit-learn pyyaml tqdm deap ta ib_insync loguru
'''

    cmd = f'ssh {remote_user}@{remote_host} "{setup_commands}"'
    run_command(cmd, check=False)


def launch_training(
    remote_host: str,
    remote_user: str,
    remote_path: str,
    n_islands: int = 4,
    pop_per_island: int = 20,
    generations: int = 100,
    epochs_per_eval: int = 10,
    final_epochs: int = 200,
    screen_name: str = "ga_training"
):
    """Launch training on remote machine using screen/tmux."""
    print(f"\n{'='*50}")
    print("Launching training on remote machine...")
    print(f"{'='*50}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"checkpoints/ga_optimized_{timestamp}"
    log_file = f"logs/training_{timestamp}.log"

    training_cmd = f'''
cd {remote_path} && \\
source .venv/bin/activate && \\
mkdir -p logs && \\
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u training/multi_gpu_genetic_trainer.py \\
    --data data/csv \\
    --config config/config.yaml \\
    --output {output_dir} \\
    --n-islands {n_islands} \\
    --pop-per-island {pop_per_island} \\
    --generations {generations} \\
    --epochs-per-eval {epochs_per_eval} \\
    --final-epochs {final_epochs} \\
    --seed 42 \\
    2>&1 | tee {log_file}
'''

    # Use screen for background execution
    screen_cmd = f'ssh {remote_user}@{remote_host} "screen -dmS {screen_name} bash -c \\"{training_cmd}\\""'

    try:
        run_command(screen_cmd)
        print(f"\nTraining launched in screen session: {screen_name}")
        print(f"Output directory: {output_dir}")
        print(f"Log file: {log_file}")
    except subprocess.CalledProcessError:
        # Fallback: use nohup
        print("Screen not available, using nohup...")
        nohup_cmd = f'ssh {remote_user}@{remote_host} "cd {remote_path} && nohup bash -c \\"{training_cmd}\\" > {log_file} 2>&1 &"'
        run_command(nohup_cmd)

    return output_dir, log_file


def monitor_training(remote_host: str, remote_user: str, remote_path: str, log_file: str, interval: int = 30):
    """Monitor training progress."""
    print(f"\n{'='*50}")
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring (training continues)")
    print(f"{'='*50}\n")

    try:
        while True:
            # Get latest log lines
            cmd = f'ssh {remote_user}@{remote_host} "tail -20 {remote_path}/{log_file} 2>/dev/null || echo \'Waiting for log...\'"'
            result = run_command(cmd, capture_output=True, check=False)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Latest progress:")
            print("-" * 40)
            print(result.stdout)

            # Check GPU utilization
            cmd = f'ssh {remote_user}@{remote_host} "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo \'GPU info unavailable\'"'
            result = run_command(cmd, capture_output=True, check=False)
            print("\nGPU Status:")
            print("-" * 40)
            print(result.stdout)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped. Training continues on remote machine.")


def retrieve_results(remote_host: str, remote_user: str, remote_path: str, output_dir: str, local_output: str):
    """Retrieve training results from remote machine."""
    print(f"\n{'='*50}")
    print("Retrieving results from remote machine...")
    print(f"{'='*50}")

    Path(local_output).mkdir(parents=True, exist_ok=True)

    try:
        cmd = f'rsync -avz --progress {remote_user}@{remote_host}:{remote_path}/{output_dir}/ "{local_output}/"'
        run_command(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        cmd = f'scp -r {remote_user}@{remote_host}:{remote_path}/{output_dir}/* "{local_output}/"'
        run_command(cmd)

    print(f"\nResults saved to: {local_output}")


def check_training_status(remote_host: str, remote_user: str, screen_name: str = "ga_training"):
    """Check if training is still running."""
    cmd = f'ssh {remote_user}@{remote_host} "screen -list | grep {screen_name} || echo \'Not running\'"'
    result = run_command(cmd, capture_output=True, check=False)
    return screen_name in result.stdout


def main():
    parser = argparse.ArgumentParser(description="Launch Remote Multi-GPU Training")

    # Connection settings
    parser.add_argument("--host", type=str, default="192.168.0.129", help="Remote host IP")
    parser.add_argument("--user", type=str, required=True, help="SSH username")
    parser.add_argument("--remote-path", type=str, default="~/nq_trading_system", help="Remote project path")

    # Training settings
    parser.add_argument("--n-islands", type=int, default=4, help="Number of GA islands")
    parser.add_argument("--pop-per-island", type=int, default=20, help="Population per island")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--epochs-per-eval", type=int, default=10, help="Training epochs per fitness evaluation")
    parser.add_argument("--final-epochs", type=int, default=200, help="Epochs for final best model training")

    # Actions
    parser.add_argument("--sync-only", action="store_true", help="Only sync code, don't launch training")
    parser.add_argument("--monitor", action="store_true", help="Monitor running training")
    parser.add_argument("--retrieve", type=str, help="Retrieve results from specified output dir")
    parser.add_argument("--status", action="store_true", help="Check training status")

    args = parser.parse_args()

    local_path = str(Path(__file__).parent.absolute())
    local_data = str(Path(local_path) / "data")

    if args.status:
        is_running = check_training_status(args.host, args.user)
        print(f"Training status: {'Running' if is_running else 'Not running'}")
        return

    if args.monitor:
        # Find latest log file
        cmd = f'ssh {args.user}@{args.host} "ls -t {args.remote_path}/logs/training_*.log 2>/dev/null | head -1"'
        result = run_command(cmd, capture_output=True, check=False)
        log_file = result.stdout.strip().replace(f"{args.remote_path}/", "")
        if log_file:
            monitor_training(args.host, args.user, args.remote_path, log_file)
        else:
            print("No log files found")
        return

    if args.retrieve:
        retrieve_results(
            args.host, args.user, args.remote_path,
            args.retrieve,
            str(Path(local_path) / "results" / Path(args.retrieve).name)
        )
        return

    # Full setup and launch
    print(f"\n{'='*60}")
    print("MULTI-GPU GENETIC ALGORITHM TRAINING LAUNCHER")
    print(f"{'='*60}")
    print(f"Remote host: {args.host}")
    print(f"User: {args.user}")
    print(f"Remote path: {args.remote_path}")
    print(f"\nGA Configuration:")
    print(f"  Islands: {args.n_islands}")
    print(f"  Population per island: {args.pop_per_island}")
    print(f"  Total population: {args.n_islands * args.pop_per_island}")
    print(f"  Generations: {args.generations}")
    print(f"  Epochs per evaluation: {args.epochs_per_eval}")
    print(f"  Final training epochs: {args.final_epochs}")

    # Sync code
    sync_to_remote(local_path, args.host, args.user, args.remote_path)

    # Sync data
    sync_data_to_remote(local_data, args.host, args.user, args.remote_path)

    if args.sync_only:
        print("\nCode synced. Use --monitor to check progress.")
        return

    # Setup environment
    setup_remote_environment(args.host, args.user, args.remote_path)

    # Launch training
    output_dir, log_file = launch_training(
        args.host, args.user, args.remote_path,
        n_islands=args.n_islands,
        pop_per_island=args.pop_per_island,
        generations=args.generations,
        epochs_per_eval=args.epochs_per_eval,
        final_epochs=args.final_epochs
    )

    print(f"\n{'='*60}")
    print("TRAINING LAUNCHED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nTo monitor progress:")
    print(f"  python launch_remote_training.py --host {args.host} --user {args.user} --monitor")
    print(f"\nTo check status:")
    print(f"  python launch_remote_training.py --host {args.host} --user {args.user} --status")
    print(f"\nTo retrieve results:")
    print(f"  python launch_remote_training.py --host {args.host} --user {args.user} --retrieve {output_dir}")
    print(f"\nTo attach to screen session on remote:")
    print(f"  ssh {args.user}@{args.host}")
    print(f"  screen -r ga_training")

    # Ask if user wants to monitor
    response = input("\nStart monitoring now? [Y/n]: ").strip().lower()
    if response != 'n':
        monitor_training(args.host, args.user, args.remote_path, log_file)


if __name__ == "__main__":
    main()
