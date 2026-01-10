"""
Training script for the NQ Multi-Agent Trading System.

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --genetic

Multi-GPU Training (2x RTX 5090):
    torchrun --nproc_per_node=2 train.py --config config/config_remote.yaml --distributed
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data_pipeline import create_data_loaders
from models import MetaAgent
from training import Trainer, GeneticOptimizer, HyperparameterSpace, FitnessEvaluator
from training.trainer import TrainingConfig
from training.genetic_optimizer import create_fitness_function
from utils import setup_logger, get_logger, get_device, print_gpu_info, clear_gpu_memory


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL for NVIDIA GPUs
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        # Set device for this process
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(config: dict, checkpoint: str = None, distributed: bool = False):
    """
    Train the multi-agent trading model.

    Args:
        config: Configuration dictionary
        checkpoint: Optional checkpoint path to resume from
        distributed: Whether to use distributed training
    """
    # Setup distributed training if enabled
    rank, world_size, local_rank = 0, 1, 0
    if distributed:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"🚀 Distributed training with {world_size} GPUs")
    else:
        device = get_device(prefer_gpu=config['training']['device'] == 'cuda')

    logger = get_logger("train")

    if is_main_process():
        print_gpu_info()

    # Load data
    logger.info("Loading data...")

    # Adjust batch size for distributed training (per GPU)
    batch_size = config['training']['batch_size']
    if distributed:
        batch_size = batch_size // world_size

    train_loader, val_loader, test_loader, feature_info = create_data_loaders(
        csv_folder=config['data']['csv_folder'],
        sequence_length=config['data']['sequence_length'],
        batch_size=batch_size,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        start_date="2008-01-01",
        end_date="2024-06-01"
    )

    # For distributed training, wrap data loaders with DistributedSampler
    if distributed:
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=config['training'].get('distributed', {}).get('num_workers', 4),
            pin_memory=True
        )

    if is_main_process():
        logger.info(f"Feature info: {feature_info}")

    # Create model
    if is_main_process():
        logger.info("Creating model...")

    model = MetaAgent(
        input_dim=feature_info['total_features'],
        embedding_dim=config['model']['transformer']['d_model'],
        n_heads=4,
        transformer_layers=config['model']['transformer']['n_layers'],
        transformer_heads=config['model']['transformer']['n_heads'],
        transformer_ff=config['model']['transformer']['d_ff'],
        dropout=config['model']['transformer']['dropout'],
        profit_hidden=config['model']['profit_agent']['hidden_dim'],
        risk_hidden=config['model']['risk_agent']['hidden_dim']
    )

    # Move model to device and wrap with DDP for distributed training
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process():
            logger.info(f"Model wrapped with DistributedDataParallel")

    # Count parameters
    base_model = model.module if distributed else model
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create training config
    training_config = TrainingConfig(
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        gradient_clip=config['training']['gradient_clip'],
        batch_size=config['training']['batch_size'],
        mixed_precision=config['training'].get('mixed_precision', True),
        checkpoint_dir=config['training']['checkpoint_dir'],
        log_dir=config['training']['log_dir'],
        device=str(device)
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )

    # Resume from checkpoint if provided
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        trainer.load_checkpoint(checkpoint)

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    fitness_eval = FitnessEvaluator()
    test_fitness = fitness_eval.calculate_fitness(test_metrics)

    logger.info(f"Test Results:")
    logger.info(f"  Fitness: {test_fitness:.4f}")
    logger.info(f"  Sharpe Ratio: {test_metrics.sharpe_ratio:.2f}")
    logger.info(f"  Total Return: {test_metrics.total_return:.2%}")
    logger.info(f"  Max Drawdown: {test_metrics.max_drawdown:.2%}")
    logger.info(f"  Win Rate: {test_metrics.win_rate:.2%}")
    logger.info(f"  Profit Factor: {test_metrics.profit_factor:.2f}")

    return trainer


def train_with_genetic_optimization(config: dict):
    """
    Train with genetic algorithm hyperparameter optimization.

    Args:
        config: Configuration dictionary
    """
    logger = get_logger("genetic")

    device = get_device(prefer_gpu=config['training']['device'] == 'cuda')
    print_gpu_info()

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, _, feature_info = create_data_loaders(
        csv_folder=config['data']['csv_folder'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        start_date="2008-01-01",
        end_date="2024-06-01"
    )

    # Define search space
    search_space = HyperparameterSpace(
        d_model=config['genetic']['search_space'].get('d_model', [128, 256, 512]),
        n_heads=config['genetic']['search_space'].get('n_heads', [4, 8, 16]),
        n_layers=config['genetic']['search_space'].get('n_layers', [2, 4, 6]),
        dropout=config['genetic']['search_space'].get('dropout', [0.05, 0.1, 0.2]),
        learning_rate=config['genetic']['search_space'].get('learning_rate', [1e-5, 1e-4, 1e-3]),
        batch_size=config['genetic']['search_space'].get('batch_size', [32, 64, 128])
    )

    # Create genetic optimizer
    optimizer = GeneticOptimizer(
        search_space=search_space,
        population_size=config['genetic']['population_size'],
        n_generations=config['genetic']['generations'],
        mutation_rate=config['genetic']['mutation_rate'],
        crossover_rate=config['genetic']['crossover_rate'],
        tournament_size=config['genetic'].get('tournament_size', 3),
        elite_size=config['genetic'].get('elite_size', 5)
    )

    # Create fitness function
    fitness_fn = create_fitness_function(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=feature_info['total_features'],
        device=str(device),
        max_epochs=10  # Quick training for evaluation
    )

    # Run optimization
    logger.info("Starting genetic optimization...")
    best_hp, history = optimizer.optimize(
        fitness_function=fitness_fn,
        early_stopping_generations=20,
        target_fitness=0.8
    )

    logger.info(f"Best hyperparameters: {best_hp}")

    # Save results
    results_path = Path(config['training']['checkpoint_dir']) / "genetic_results.json"
    optimizer.save_results(str(results_path), best_hp)
    logger.info(f"Results saved to {results_path}")

    # Train final model with best hyperparameters
    logger.info("Training final model with best hyperparameters...")

    final_model = MetaAgent(
        input_dim=feature_info['total_features'],
        embedding_dim=best_hp['d_model'],
        n_heads=4,
        transformer_layers=best_hp['n_layers'],
        transformer_heads=best_hp['n_heads'],
        transformer_ff=best_hp['d_ff'],
        dropout=best_hp['dropout'],
        profit_hidden=best_hp['profit_hidden'],
        risk_hidden=best_hp['risk_hidden']
    )

    training_config = TrainingConfig(
        epochs=config['training']['epochs'],
        learning_rate=best_hp['learning_rate'],
        weight_decay=best_hp['weight_decay'],
        batch_size=best_hp['batch_size'],
        mixed_precision=True,
        checkpoint_dir=config['training']['checkpoint_dir'],
        log_dir=config['training']['log_dir'],
        device=str(device)
    )

    trainer = Trainer(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )

    trainer.train()

    return best_hp, trainer


def main():
    parser = argparse.ArgumentParser(description="Train NQ Trading System")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--genetic",
        action="store_true",
        help="Use genetic algorithm for hyperparameter optimization"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (use with torchrun)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_level=args.log_level)
    logger = get_logger("main")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Create default config
        logger.warning(f"Config file not found at {config_path}, using defaults")
        config = {
            'data': {
                'csv_folder': 'data/csv',
                'sequence_length': 60,
                'train_split': 0.7,
                'val_split': 0.15
            },
            'model': {
                'transformer': {
                    'd_model': 256,
                    'n_heads': 8,
                    'n_layers': 4,
                    'd_ff': 1024,
                    'dropout': 0.1
                },
                'profit_agent': {'hidden_dim': 128},
                'risk_agent': {'hidden_dim': 64}
            },
            'training': {
                'epochs': 100,
                'batch_size': 64,
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'gradient_clip': 1.0,
                'mixed_precision': True,
                'device': 'cuda',
                'checkpoint_dir': 'checkpoints',
                'log_dir': 'logs'
            },
            'genetic': {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'search_space': {}
            }
        }
    else:
        config = load_config(str(config_path))

    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs

    # Run training
    try:
        if args.genetic:
            if is_main_process():
                logger.info("Starting training with genetic optimization...")
            best_hp, trainer = train_with_genetic_optimization(config)
        else:
            if is_main_process():
                logger.info("Starting standard training...")
            trainer = train_model(config, args.checkpoint, distributed=args.distributed)

        if is_main_process():
            logger.info("Training complete!")

    finally:
        # Clean up distributed training
        cleanup_distributed()


if __name__ == "__main__":
    main()
