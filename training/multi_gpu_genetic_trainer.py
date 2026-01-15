"""
Multi-GPU Genetic Algorithm Training Pipeline.

Optimized for distributed training across multiple GPUs:
- 2x RTX 5090 (primary compute)
- 2x RTX 3090 NVLinked (secondary compute)

Uses parallel fitness evaluation with island-based genetic algorithm
for faster convergence and better exploration.
"""

import os

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE importing torch
# This is required for CUDA to work with multiprocessing
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import random
import json
import time
from pathlib import Path
from queue import Queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import yaml
from datetime import datetime

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import MetaAgent
from data_pipeline import CSVDataLoader, OHLCVPreprocessor, CyclicalTimeEncoder, TradingDataset
from training.fitness import FitnessEvaluator, PerformanceMetrics
from training.trainer import Trainer, TrainingConfig
from utils import setup_logger, get_logger


# Global variables for worker processes (set during initialization)
_worker_train_data = None
_worker_val_data = None
_worker_input_dim = None
_worker_epochs = None
_worker_use_mixed_precision = None


def _init_worker(train_data, val_data, input_dim, epochs, use_mixed_precision):
    """Initialize worker process with shared data."""
    global _worker_train_data, _worker_val_data, _worker_input_dim, _worker_epochs, _worker_use_mixed_precision
    _worker_train_data = train_data
    _worker_val_data = val_data
    _worker_input_dim = input_dim
    _worker_epochs = epochs
    _worker_use_mixed_precision = use_mixed_precision


def _get_gpu_memory_gb(gpu_id: int) -> float:
    """Get total GPU memory in GB."""
    try:
        props = torch.cuda.get_device_properties(gpu_id)
        return props.total_memory / (1024**3)
    except Exception:
        return 24.0  # Conservative default


def _get_available_gpu_memory_gb(gpu_id: int) -> float:
    """Get available (free) GPU memory in GB."""
    try:
        torch.cuda.set_device(gpu_id)
        free_memory = torch.cuda.mem_get_info(gpu_id)[0]
        return free_memory / (1024**3)
    except Exception:
        return 16.0  # Conservative default


def _adjust_config_for_gpu_memory(config: dict, gpu_id: int) -> dict:
    """
    Adjust model and training config based on GPU memory to avoid OOM.

    RTX 5090 (32GB): Can handle larger models and batch sizes
    RTX 3090 (24GB): Need to limit batch sizes for larger models

    IMPORTANT: We use very conservative limits because in multi-GPU training,
    memory can be consumed by other processes on the same GPU.
    """
    gpu_memory_gb = _get_gpu_memory_gb(gpu_id)
    available_memory_gb = _get_available_gpu_memory_gb(gpu_id)

    # Get model parameters that affect memory usage
    d_model = config['model']['transformer']['d_model']
    n_layers = config['model']['transformer']['n_layers']
    d_ff = config['model']['transformer']['d_ff']
    batch_size = config['training']['batch_size']

    # Calculate model size factor (rough estimate)
    # d_model=512, n_layers=8, d_ff=2048 is "large" (factor ~4)
    # d_model=128, n_layers=2, d_ff=512 is "small" (factor ~0.25)
    model_factor = (d_model / 256) * (n_layers / 4) * (d_ff / 1024)

    original_batch_size = batch_size

    if gpu_memory_gb >= 30:  # RTX 5090 (32GB)
        # Full capacity - can handle larger batches
        if model_factor > 3.0:  # Very large model
            max_batch = 192
        elif model_factor > 1.5:  # Large model
            max_batch = 256
        else:
            max_batch = 384  # Small/medium models
    elif gpu_memory_gb >= 22:  # RTX 3090 (24GB) - BE VERY CONSERVATIVE
        # In multi-GPU setups, other processes may consume memory
        # Use very conservative limits to avoid OOM
        if model_factor > 1.5:  # Large model
            max_batch = 32  # Very conservative for large models
        elif model_factor > 1.0:  # Medium-large model
            max_batch = 64
        elif model_factor > 0.5:  # Medium model
            max_batch = 96
        else:  # Small model
            max_batch = 128
    else:  # Smaller GPUs
        max_batch = 32

    # Additional constraint: if available memory is low, reduce further
    # This catches cases where other processes are using GPU memory
    if available_memory_gb < 22:  # Less than expected for 24GB GPU
        max_batch = min(max_batch, 64)
    if available_memory_gb < 18:
        max_batch = min(max_batch, 32)
    if available_memory_gb < 10:
        max_batch = 16  # Minimal batch size

    # Apply batch size limit
    adjusted_batch_size = min(batch_size, max_batch)

    if adjusted_batch_size != original_batch_size:
        print(f"GPU {gpu_id} ({gpu_memory_gb:.1f}GB, {available_memory_gb:.1f}GB free): "
              f"Adjusted batch_size {original_batch_size} -> {adjusted_batch_size} "
              f"(model_factor={model_factor:.2f})")

    config['training']['batch_size'] = adjusted_batch_size
    return config


def _evaluate_individual_worker(args):
    """
    Worker function for parallel GPU evaluation.
    Must be at module level for pickling.
    """
    individual_genes, gpu_id = args
    model = None
    trainer = None
    fitness = -1000.0

    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        # Clear any leftover memory from previous runs
        torch.cuda.empty_cache()
        torch.cuda.synchronize(gpu_id)

        # Check if GPU has enough memory to train
        available_memory_gb = _get_available_gpu_memory_gb(gpu_id)
        if available_memory_gb < 5.0:
            print(f"GPU {gpu_id}: Only {available_memory_gb:.1f}GB available - skipping (need at least 5GB)")
            return (individual_genes, -500.0, gpu_id)  # Return low but not worst fitness

        # Recreate individual from genes
        individual = Individual(genes=individual_genes)
        config = individual.to_config()

        # Adjust config based on GPU memory to avoid OOM
        config = _adjust_config_for_gpu_memory(config, gpu_id)

        # Create model
        model = MetaAgent(
            input_dim=_worker_input_dim,
            embedding_dim=config['model']['transformer']['d_model'],
            n_heads=config['model']['meta_agent']['n_heads'],
            transformer_layers=config['model']['transformer']['n_layers'],
            transformer_heads=config['model']['transformer']['n_heads'],
            transformer_ff=config['model']['transformer']['d_ff'],
            dropout=config['model']['transformer']['dropout'],
            profit_hidden=config['model']['profit_agent']['hidden_dim'],
            risk_hidden=config['model']['risk_agent']['hidden_dim']
        ).to(device)

        # Create data loaders
        train_loader = DataLoader(
            _worker_train_data,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # No sub-workers in worker process
            pin_memory=False
        )
        val_loader = DataLoader(
            _worker_val_data,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # Training config
        training_config = TrainingConfig(
            epochs=_worker_epochs,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            gradient_clip=config['training']['gradient_clip'],
            batch_size=config['training']['batch_size'],
            mixed_precision=_worker_use_mixed_precision
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device
        )

        # Train
        trainer.train(_worker_epochs)

        # Evaluate
        metrics = trainer.evaluate(val_loader)

        # Calculate fitness
        fitness_eval = FitnessEvaluator()
        fitness = fitness_eval.calculate_fitness(metrics)

    except Exception as e:
        print(f"Worker on GPU {gpu_id} failed: {e}")
        import traceback
        traceback.print_exc()
        fitness = -1000.0

    finally:
        # Always cleanup GPU resources
        try:
            if model is not None:
                del model
            if trainer is not None:
                del trainer
            # Synchronize and clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
                torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore cleanup errors

    return (individual_genes, fitness, gpu_id)


@dataclass
class GPUConfig:
    """Configuration for a single GPU."""
    device_id: int
    device_name: str
    memory_gb: float
    is_nvlinked: bool = False
    nvlink_partner: Optional[int] = None


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU setup."""
    gpus: List[GPUConfig] = field(default_factory=list)
    primary_gpus: List[int] = field(default_factory=list)  # For main training
    secondary_gpus: List[int] = field(default_factory=list)  # For parallel eval

    @classmethod
    def detect_gpus(cls) -> 'MultiGPUConfig':
        """Auto-detect GPU configuration."""
        config = cls()

        if not torch.cuda.is_available():
            return config

        # Check PyTorch's supported CUDA compute capabilities
        # RTX 5090 (Blackwell) needs sm_120, which may not be supported yet
        supported_archs = [50, 60, 70, 75, 80, 86, 90]  # Common supported architectures

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu = GPUConfig(
                device_id=i,
                device_name=props.name,
                memory_gb=props.total_memory / (1024**3)
            )
            config.gpus.append(gpu)

            # Check if GPU is compatible (test with a small tensor)
            is_compatible = True
            try:
                test_tensor = torch.zeros(1, device=f'cuda:{i}')
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"GPU {i} ({props.name}) not compatible with current PyTorch: {e}")
                is_compatible = False

            if not is_compatible:
                continue

            # Classify GPUs - use compatible GPUs
            if '5090' in props.name:
                # RTX 5090 (Blackwell) - use as primary (fastest)
                config.primary_gpus.append(i)
            elif '3090' in props.name:
                config.secondary_gpus.append(i)
                # Check for NVLink (assuming consecutive 3090s are linked)
                if len(config.secondary_gpus) == 2:
                    config.gpus[-2].is_nvlinked = True
                    config.gpus[-2].nvlink_partner = i
                    config.gpus[-1].is_nvlinked = True
                    config.gpus[-1].nvlink_partner = config.secondary_gpus[0]
            elif '4090' in props.name or '4080' in props.name:
                config.primary_gpus.append(i)
            else:
                # Default to secondary for other compatible GPUs
                config.secondary_gpus.append(i)

        # If no primary GPUs, promote secondary to primary
        if not config.primary_gpus and config.secondary_gpus:
            config.primary_gpus = config.secondary_gpus
            config.secondary_gpus = []

        return config


@dataclass
class EnhancedHyperparameterSpace:
    """Enhanced hyperparameter search space with architecture options."""

    # Transformer architecture
    d_model: List[int] = field(default_factory=lambda: [128, 192, 256, 384, 512])
    n_heads: List[int] = field(default_factory=lambda: [4, 8, 12, 16])
    n_layers: List[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])
    d_ff_multiplier: List[int] = field(default_factory=lambda: [2, 3, 4])
    dropout: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25])

    # Training parameters - conservative batch sizes for stability
    learning_rate: List[float] = field(default_factory=lambda: [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4])
    batch_size: List[int] = field(default_factory=lambda: [64, 128, 192, 256, 384])  # Conservative for stability
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 1e-6, 1e-5, 1e-4, 1e-3])

    # Profit agent parameters
    profit_hidden: List[int] = field(default_factory=lambda: [64, 96, 128, 192, 256])
    profit_layers: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Risk agent parameters
    risk_hidden: List[int] = field(default_factory=lambda: [32, 48, 64, 96, 128])
    risk_layers: List[int] = field(default_factory=lambda: [2, 3])

    # Duration agent parameters
    duration_hidden: List[int] = field(default_factory=lambda: [32, 48, 64, 96])
    duration_layers: List[int] = field(default_factory=lambda: [2, 3])

    # Meta-agent parameters
    meta_hidden: List[int] = field(default_factory=lambda: [128, 192, 256, 384])
    meta_heads: List[int] = field(default_factory=lambda: [2, 4, 8])

    # Risk management parameters
    max_risk_per_trade: List[float] = field(default_factory=lambda: [0.01, 0.015, 0.02, 0.025, 0.03])
    atr_multiplier: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0, 3.5])

    # Activation functions
    activation: List[str] = field(default_factory=lambda: ['gelu', 'relu', 'silu', 'mish'])

    # Regularization
    label_smoothing: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15])
    gradient_clip: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def sample_random(self) -> Dict:
        """Sample a random configuration."""
        config = {}
        for key, values in self.to_dict().items():
            config[key] = random.choice(values)

        # Ensure validity
        while config['d_model'] % config['n_heads'] != 0:
            config['n_heads'] = random.choice(self.n_heads)

        config['d_ff'] = config['d_model'] * config['d_ff_multiplier']
        return config


@dataclass
class Individual:
    """Enhanced individual representation."""
    genes: Dict
    fitness: float = 0.0
    metrics: Optional[PerformanceMetrics] = None
    generation: int = 0
    island: int = 0

    def to_config(self) -> Dict:
        """Convert to model configuration."""
        return {
            'model': {
                'transformer': {
                    'd_model': self.genes['d_model'],
                    'n_heads': self.genes['n_heads'],
                    'n_layers': self.genes['n_layers'],
                    'd_ff': self.genes.get('d_ff', self.genes['d_model'] * self.genes.get('d_ff_multiplier', 4)),
                    'dropout': self.genes['dropout'],
                },
                'profit_agent': {
                    'hidden_dim': self.genes['profit_hidden'],
                    'n_layers': self.genes.get('profit_layers', 2),
                },
                'risk_agent': {
                    'hidden_dim': self.genes['risk_hidden'],
                    'n_layers': self.genes.get('risk_layers', 2),
                },
                'duration_agent': {
                    'hidden_dim': self.genes.get('duration_hidden', 64),
                    'n_layers': self.genes.get('duration_layers', 2),
                },
                'meta_agent': {
                    'hidden_dim': self.genes.get('meta_hidden', 256),
                    'n_heads': self.genes.get('meta_heads', 4),
                },
            },
            'training': {
                'learning_rate': self.genes['learning_rate'],
                'batch_size': self.genes['batch_size'],
                'weight_decay': self.genes['weight_decay'],
                'gradient_clip': self.genes.get('gradient_clip', 1.0),
                'label_smoothing': self.genes.get('label_smoothing', 0.0),
            },
            'trading': {
                'max_risk_per_trade': self.genes.get('max_risk_per_trade', 0.02),
                'atr_multiplier': self.genes.get('atr_multiplier', 2.0),
            }
        }


class ParallelFitnessEvaluator:
    """
    Parallel fitness evaluation across multiple GPUs.

    Distributes population evaluation across available GPUs for
    maximum throughput.
    """

    def __init__(
        self,
        gpu_config: MultiGPUConfig,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
        input_dim: int,
        epochs_per_eval: int = 10,
        use_mixed_precision: bool = True
    ):
        self.gpu_config = gpu_config
        self.train_data = train_data
        self.val_data = val_data
        self.input_dim = input_dim
        self.epochs_per_eval = epochs_per_eval
        self.use_mixed_precision = use_mixed_precision
        self.fitness_evaluator = FitnessEvaluator()
        self.logger = get_logger("parallel_fitness")

        # Get all available GPUs
        self.all_gpus = gpu_config.primary_gpus + gpu_config.secondary_gpus
        if not self.all_gpus:
            self.all_gpus = [0] if torch.cuda.is_available() else []

    def evaluate_individual(self, individual: Individual, gpu_id: int) -> Individual:
        """Evaluate a single individual on specified GPU."""
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        try:
            config = individual.to_config()

            # Create model
            model = MetaAgent(
                input_dim=self.input_dim,
                embedding_dim=config['model']['transformer']['d_model'],
                n_heads=config['model']['meta_agent']['n_heads'],
                transformer_layers=config['model']['transformer']['n_layers'],
                transformer_heads=config['model']['transformer']['n_heads'],
                transformer_ff=config['model']['transformer']['d_ff'],
                dropout=config['model']['transformer']['dropout'],
                profit_hidden=config['model']['profit_agent']['hidden_dim'],
                risk_hidden=config['model']['risk_agent']['hidden_dim']
            ).to(device)

            # Create data loaders
            train_loader = DataLoader(
                self.train_data,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            val_loader = DataLoader(
                self.val_data,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            # Create trainer
            training_config = TrainingConfig(
                epochs=self.epochs_per_eval,
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                gradient_clip=config['training']['gradient_clip'],
                batch_size=config['training']['batch_size'],
                mixed_precision=self.use_mixed_precision
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config,
                device=device
            )

            # Train
            trainer.train(self.epochs_per_eval)

            # Evaluate
            metrics = trainer.evaluate(val_loader)

            # Calculate fitness
            fitness = self.fitness_evaluator.calculate_fitness(metrics)

            individual.fitness = fitness
            individual.metrics = metrics

            # Cleanup
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

        except Exception as e:
            self.logger.error(f"Evaluation failed on GPU {gpu_id}: {e}")
            individual.fitness = -1000.0

        return individual

    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """Evaluate population in parallel across GPUs using ProcessPoolExecutor."""
        if not self.all_gpus:
            # CPU fallback
            return [self.evaluate_individual(ind, -1) for ind in population]

        n_gpus = len(self.all_gpus)
        self.logger.info(f"Evaluating {len(population)} individuals across {n_gpus} GPUs in parallel")

        # Prepare work items - assign each individual to a GPU (round-robin)
        work_items = []
        for i, individual in enumerate(population):
            gpu_id = self.all_gpus[i % n_gpus]
            work_items.append((individual.genes.copy(), gpu_id))

        results = []

        # Use ProcessPoolExecutor with spawn context for CUDA compatibility
        # Pass initializer to set global vars in each worker process
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=n_gpus,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.train_data, self.val_data, self.input_dim,
                     self.epochs_per_eval, self.use_mixed_precision)
        ) as executor:
            # Submit all work
            futures = {executor.submit(_evaluate_individual_worker, item): i
                      for i, item in enumerate(work_items)}

            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    genes, fitness, gpu_id = future.result(timeout=700)
                    individual = population[idx]
                    individual.fitness = fitness
                    results.append((idx, individual))
                    self.logger.info(f"Individual {idx+1}/{len(population)} on GPU {gpu_id}: fitness={fitness:.4f}")
                except Exception as e:
                    self.logger.error(f"Future {idx} failed: {e}")
                    population[idx].fitness = -1000.0
                    results.append((idx, population[idx]))

        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class IslandGeneticAlgorithm:
    """
    Island-based Genetic Algorithm with migration.

    Multiple sub-populations (islands) evolve independently with
    periodic migration of best individuals between islands.
    This provides better exploration and prevents premature convergence.
    """

    def __init__(
        self,
        search_space: EnhancedHyperparameterSpace,
        n_islands: int = 4,
        population_per_island: int = 20,
        n_generations: int = 100,
        migration_interval: int = 10,
        migration_size: int = 2,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        elite_size: int = 2,
        seed: Optional[int] = None
    ):
        self.search_space = search_space
        self.n_islands = n_islands
        self.population_per_island = population_per_island
        self.total_population = n_islands * population_per_island
        self.n_generations = n_generations
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.logger = get_logger("island_ga")

        # History
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'island_best': [[] for _ in range(n_islands)],
            'best_individual': None,
            'generations': []
        }

    def _create_individual(self, island: int) -> Individual:
        """Create a random individual."""
        genes = self.search_space.sample_random()
        return Individual(genes=genes, island=island)

    def _initialize_population(self) -> List[List[Individual]]:
        """Initialize island populations."""
        islands = []
        for i in range(self.n_islands):
            island_pop = [self._create_individual(i) for _ in range(self.population_per_island)]
            islands.append(island_pop)
        return islands

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        child1_genes = {}
        child2_genes = {}

        for key in parent1.genes:
            if random.random() < 0.5:
                child1_genes[key] = parent1.genes[key]
                child2_genes[key] = parent2.genes[key]
            else:
                child1_genes[key] = parent2.genes[key]
                child2_genes[key] = parent1.genes[key]

        # Repair d_model/n_heads compatibility
        for genes in [child1_genes, child2_genes]:
            while genes['d_model'] % genes['n_heads'] != 0:
                genes['n_heads'] = random.choice(self.search_space.n_heads)
            genes['d_ff'] = genes['d_model'] * genes.get('d_ff_multiplier', 4)

        return (
            Individual(genes=child1_genes, island=parent1.island),
            Individual(genes=child2_genes, island=parent2.island)
        )

    def _mutate(self, individual: Individual) -> Individual:
        """Mutate individual."""
        space_dict = self.search_space.to_dict()

        for key, values in space_dict.items():
            if random.random() < self.mutation_rate:
                if isinstance(values[0], float):
                    # Continuous: Gaussian perturbation or resample
                    if random.random() < 0.5:
                        std = (max(values) - min(values)) * 0.15
                        new_val = individual.genes.get(key, values[0]) + random.gauss(0, std)
                        new_val = max(min(values), min(max(values), new_val))
                        individual.genes[key] = new_val
                    else:
                        individual.genes[key] = random.choice(values)
                else:
                    individual.genes[key] = random.choice(values)

        # Repair
        while individual.genes['d_model'] % individual.genes['n_heads'] != 0:
            individual.genes['n_heads'] = random.choice(self.search_space.n_heads)
        individual.genes['d_ff'] = individual.genes['d_model'] * individual.genes.get('d_ff_multiplier', 4)

        return individual

    def _migrate(self, islands: List[List[Individual]]) -> List[List[Individual]]:
        """Migrate best individuals between islands (ring topology)."""
        migrants = []

        # Collect best from each island
        for island_pop in islands:
            sorted_pop = sorted(island_pop, key=lambda x: x.fitness, reverse=True)
            migrants.append(sorted_pop[:self.migration_size])

        # Send to next island (ring)
        for i in range(self.n_islands):
            target = (i + 1) % self.n_islands
            # Replace worst individuals
            islands[target] = sorted(islands[target], key=lambda x: x.fitness, reverse=True)
            for j, migrant in enumerate(migrants[i]):
                migrant.island = target
                islands[target][-(j+1)] = migrant

        return islands

    def _evolve_island(self, population: List[Individual]) -> List[Individual]:
        """Evolve a single island for one generation."""
        # Elitism
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_pop[:self.elite_size]

        # Create offspring
        offspring = []
        while len(offspring) < self.population_per_island - self.elite_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1 = Individual(genes=parent1.genes.copy(), island=parent1.island)
                child2 = Individual(genes=parent2.genes.copy(), island=parent2.island)

            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            offspring.extend([child1, child2])

        offspring = offspring[:self.population_per_island - self.elite_size]
        return elite + offspring

    def optimize(
        self,
        fitness_evaluator: ParallelFitnessEvaluator,
        early_stopping_generations: int = 25,
        target_fitness: Optional[float] = None,
        checkpoint_dir: Optional[str] = None
    ) -> Tuple[Individual, Dict]:
        """
        Run island-based genetic algorithm optimization.

        Returns:
            (best_individual, history)
        """
        self.logger.info(f"Starting Island GA with {self.n_islands} islands, "
                        f"{self.population_per_island} individuals per island")

        # Initialize
        islands = self._initialize_population()
        best_fitness_ever = float('-inf')
        best_individual_ever = None
        generations_without_improvement = 0

        start_time = time.time()

        for gen in range(self.n_generations):
            gen_start = time.time()

            # Flatten population for parallel evaluation
            all_individuals = [ind for island in islands for ind in island]

            # Evaluate in parallel
            self.logger.info(f"Generation {gen}: Evaluating {len(all_individuals)} individuals...")
            evaluated = fitness_evaluator.evaluate_population(all_individuals)

            # Redistribute to islands
            idx = 0
            for i in range(self.n_islands):
                islands[i] = evaluated[idx:idx + self.population_per_island]
                idx += self.population_per_island

            # Statistics
            all_fitness = [ind.fitness for ind in evaluated if ind.fitness > -500]
            if all_fitness:
                best_fit = max(all_fitness)
                avg_fit = np.mean(all_fitness)
                std_fit = np.std(all_fitness)
            else:
                best_fit = -1000
                avg_fit = -1000
                std_fit = 0

            # Track best
            for ind in evaluated:
                if ind.fitness > best_fitness_ever:
                    best_fitness_ever = ind.fitness
                    best_individual_ever = ind
                    generations_without_improvement = 0

            self.history['best_fitness'].append(best_fit)
            self.history['avg_fitness'].append(avg_fit)

            for i, island in enumerate(islands):
                island_best = max(ind.fitness for ind in island)
                self.history['island_best'][i].append(island_best)

            gen_time = time.time() - gen_start
            self.logger.info(
                f"Gen {gen:3d}: Best={best_fit:.4f}, Avg={avg_fit:.4f}, "
                f"Std={std_fit:.4f}, Time={gen_time:.1f}s"
            )

            # Migration
            if gen > 0 and gen % self.migration_interval == 0:
                self.logger.info(f"Migration at generation {gen}")
                islands = self._migrate(islands)

            # Evolve islands
            for i in range(self.n_islands):
                islands[i] = self._evolve_island(islands[i])

            # Check early stopping
            generations_without_improvement += 1
            if generations_without_improvement >= early_stopping_generations:
                self.logger.info(f"Early stopping: no improvement for {early_stopping_generations} generations")
                break

            if target_fitness and best_fitness_ever >= target_fitness:
                self.logger.info(f"Target fitness {target_fitness} reached!")
                break

            # Checkpoint
            if checkpoint_dir and gen % 10 == 0:
                self._save_checkpoint(checkpoint_dir, gen, best_individual_ever)

        total_time = time.time() - start_time
        self.logger.info(f"Optimization complete in {total_time/60:.1f} minutes")
        self.logger.info(f"Best fitness: {best_fitness_ever:.4f}")

        self.history['best_individual'] = best_individual_ever

        return best_individual_ever, self.history

    def _save_checkpoint(self, checkpoint_dir: str, generation: int, best: Individual):
        """Save optimization checkpoint."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'generation': generation,
            'best_fitness': best.fitness if best else 0,
            'best_genes': best.genes if best else {},
            'history': {
                'best_fitness': self.history['best_fitness'],
                'avg_fitness': self.history['avg_fitness']
            }
        }

        filepath = Path(checkpoint_dir) / f"ga_checkpoint_gen{generation}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)


def train_best_model(
    best_individual: Individual,
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    test_data: torch.utils.data.Dataset,
    input_dim: int,
    gpu_config: MultiGPUConfig,
    output_dir: str,
    full_epochs: int = 200
) -> Tuple[nn.Module, Dict]:
    """
    Train the best model configuration with full resources.

    Uses primary GPUs (same memory class) with DataParallel for final training.
    Avoids mixing GPUs with different memory sizes to prevent OOM.
    """
    logger = get_logger("final_training")
    logger.info("Starting final training with best configuration...")

    config = best_individual.to_config()

    # Clear GPU memory before final training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Use single GPU for final training to avoid DataParallel CUBLAS issues
    # The GA optimization already uses all GPUs in parallel for evaluating individuals
    # Final training is just one model, so single GPU is fine
    all_gpus = gpu_config.primary_gpus + gpu_config.secondary_gpus

    # Prefer the fastest GPU (primary = RTX 5090)
    if gpu_config.primary_gpus:
        training_gpu = gpu_config.primary_gpus[0]
        logger.info(f"Using primary GPU (fastest) for final training: {training_gpu}")
    elif all_gpus:
        training_gpu = all_gpus[0]
        logger.info(f"Using GPU for final training: {training_gpu}")
    else:
        training_gpu = 0
        logger.info("No GPUs detected, using GPU 0")

    device = f"cuda:{training_gpu}"

    # Adjust config based on GPU memory
    config = _adjust_config_for_gpu_memory(config, training_gpu)

    # Create model
    model = MetaAgent(
        input_dim=input_dim,
        embedding_dim=config['model']['transformer']['d_model'],
        n_heads=config['model']['meta_agent']['n_heads'],
        transformer_layers=config['model']['transformer']['n_layers'],
        transformer_heads=config['model']['transformer']['n_heads'],
        transformer_ff=config['model']['transformer']['d_ff'],
        dropout=config['model']['transformer']['dropout'],
        profit_hidden=config['model']['profit_agent']['hidden_dim'],
        risk_hidden=config['model']['risk_agent']['hidden_dim']
    )

    model = model.to(device)

    # Use single GPU batch size (can be larger on 5090)
    batch_size = config['training']['batch_size']

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Training config
    training_config = TrainingConfig(
        epochs=full_epochs,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        gradient_clip=config['training']['gradient_clip'],
        batch_size=batch_size,
        mixed_precision=True,
        checkpoint_dir=output_dir
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )

    # Train
    trainer.train(full_epochs)

    # Final evaluation
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    # Save final model
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get the underlying model if DataParallel
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'config': config,
        'genes': best_individual.genes,
        'test_metrics': test_metrics.__dict__ if hasattr(test_metrics, '__dict__') else test_metrics,
        'timestamp': datetime.now().isoformat()
    }

    save_path = Path(output_dir) / "best_model_ga_optimized.pt"
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")

    return model_to_save, test_metrics


def main():
    """Main entry point for multi-GPU genetic optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-GPU Genetic Algorithm Training")
    parser.add_argument("--data", type=str, default="data/csv", help="Data directory")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="checkpoints/ga_optimized", help="Output directory")
    parser.add_argument("--n-islands", type=int, default=4, help="Number of islands")
    parser.add_argument("--pop-per-island", type=int, default=20, help="Population per island")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--epochs-per-eval", type=int, default=10, help="Training epochs per fitness evaluation")
    parser.add_argument("--final-epochs", type=int, default=200, help="Epochs for final training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-rows", type=int, default=2000000, help="Max data rows (use 100000 for quick testing)")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode: 100K rows, 3 epochs, 2 generations")

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger("main")

    # Quick test mode overrides
    if args.quick_test:
        logger.info("QUICK TEST MODE: Using reduced settings for fast validation")
        args.max_rows = 100000
        args.epochs_per_eval = 3
        args.generations = 2
        args.final_epochs = 5

    # Detect GPUs
    gpu_config = MultiGPUConfig.detect_gpus()
    logger.info(f"Detected GPUs: {[g.device_name for g in gpu_config.gpus]}")
    logger.info(f"Primary GPUs: {gpu_config.primary_gpus}")
    logger.info(f"Secondary GPUs: {gpu_config.secondary_gpus}")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found, using defaults")
        config = {'data': {'sequence_length': 60}}

    # Load data
    logger.info(f"Loading data from {args.data}...")
    loader = CSVDataLoader(args.data)
    data = loader.load_all_files()

    logger.info(f"Loaded {len(data)} rows of data")

    # Clean data - remove rows with zero/negative prices
    logger.info("Cleaning data...")
    data = data[(data['close'] > 0) & (data['open'] > 0) & (data['high'] > 0) & (data['low'] > 0)]
    data = data[(data['volume'] >= 0)]
    data = data.dropna()
    logger.info(f"After cleaning: {len(data)} rows")

    # Limit data size (use last N rows for recent patterns)
    max_rows = args.max_rows
    if len(data) > max_rows:
        logger.info(f"Limiting data to last {max_rows} rows")
        data = data.iloc[-max_rows:]
    logger.info(f"Using {len(data)} rows | Data range: {data.index[0]} to {data.index[-1]}")

    # Create preprocessors
    preprocessor = OHLCVPreprocessor()
    time_encoder = CyclicalTimeEncoder()

    # Create dataset - TradingDataset handles preprocessing internally
    sequence_length = config.get('data', {}).get('sequence_length', 60)
    dataset = TradingDataset(
        ohlcv_data=data,
        sequence_length=sequence_length,
        preprocessor=preprocessor,
        time_encoder=time_encoder,
        target_horizon=1,
        include_target=True
    )

    # Get input dimension from the dataset
    input_dim = dataset.features.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create search space
    search_space = EnhancedHyperparameterSpace()

    # Create parallel fitness evaluator
    fitness_evaluator = ParallelFitnessEvaluator(
        gpu_config=gpu_config,
        train_data=train_data,
        val_data=val_data,
        input_dim=input_dim,
        epochs_per_eval=args.epochs_per_eval,
        use_mixed_precision=True
    )

    # Create island GA
    ga = IslandGeneticAlgorithm(
        search_space=search_space,
        n_islands=args.n_islands,
        population_per_island=args.pop_per_island,
        n_generations=args.generations,
        migration_interval=10,
        migration_size=2,
        mutation_rate=0.15,
        crossover_rate=0.8,
        seed=args.seed
    )

    # Run optimization
    logger.info("Starting genetic algorithm optimization...")
    best_individual, history = ga.optimize(
        fitness_evaluator=fitness_evaluator,
        early_stopping_generations=25,
        checkpoint_dir=args.output
    )

    # Save optimization results
    results_path = Path(args.output) / "ga_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_genes': best_individual.genes,
            'best_fitness': best_individual.fitness,
            'history': {
                'best_fitness': history['best_fitness'],
                'avg_fitness': history['avg_fitness']
            }
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Train best model with full resources
    logger.info("Training best model with full epochs...")
    final_model, test_metrics = train_best_model(
        best_individual=best_individual,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        input_dim=input_dim,
        gpu_config=gpu_config,
        output_dir=args.output,
        full_epochs=args.final_epochs
    )

    logger.info("=" * 50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Best configuration: {json.dumps(best_individual.genes, indent=2)}")
    logger.info(f"Best fitness: {best_individual.fitness:.4f}")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
