"""
Genetic Algorithm Optimizer for hyperparameter optimization.

Uses DEAP library to evolve optimal hyperparameters for the
multi-agent trading system.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import copy
import json
from pathlib import Path

from deap import base, creator, tools, algorithms
import torch

from .fitness import FitnessEvaluator, PerformanceMetrics


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""

    # Transformer parameters
    d_model: List[int] = None
    n_heads: List[int] = None
    n_layers: List[int] = None
    d_ff_multiplier: List[int] = None
    dropout: List[float] = None

    # Training parameters
    learning_rate: List[float] = None
    batch_size: List[int] = None
    weight_decay: List[float] = None

    # Agent parameters
    profit_hidden: List[int] = None
    risk_hidden: List[int] = None

    # Risk parameters
    max_risk_per_trade: List[float] = None
    atr_multiplier: List[float] = None

    def __post_init__(self):
        # Set defaults
        self.d_model = self.d_model or [128, 256, 512]
        self.n_heads = self.n_heads or [4, 8, 16]
        self.n_layers = self.n_layers or [2, 4, 6]
        self.d_ff_multiplier = self.d_ff_multiplier or [2, 4]
        self.dropout = self.dropout or [0.05, 0.1, 0.15, 0.2]
        self.learning_rate = self.learning_rate or [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        self.batch_size = self.batch_size or [32, 64, 128]
        self.weight_decay = self.weight_decay or [0.0, 1e-5, 1e-4]
        self.profit_hidden = self.profit_hidden or [64, 128, 256]
        self.risk_hidden = self.risk_hidden or [32, 64, 128]
        self.max_risk_per_trade = self.max_risk_per_trade or [0.01, 0.015, 0.02, 0.025]
        self.atr_multiplier = self.atr_multiplier or [1.5, 2.0, 2.5, 3.0]

    def to_dict(self) -> Dict:
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff_multiplier': self.d_ff_multiplier,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'profit_hidden': self.profit_hidden,
            'risk_hidden': self.risk_hidden,
            'max_risk_per_trade': self.max_risk_per_trade,
            'atr_multiplier': self.atr_multiplier
        }


@dataclass
class Individual:
    """Represents a single hyperparameter configuration."""
    d_model: int
    n_heads: int
    n_layers: int
    d_ff_multiplier: int
    dropout: float
    learning_rate: float
    batch_size: int
    weight_decay: float
    profit_hidden: int
    risk_hidden: int
    max_risk_per_trade: float
    atr_multiplier: float
    fitness: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_model * self.d_ff_multiplier,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'profit_hidden': self.profit_hidden,
            'risk_hidden': self.risk_hidden,
            'max_risk_per_trade': self.max_risk_per_trade,
            'atr_multiplier': self.atr_multiplier
        }

    def is_valid(self) -> bool:
        """Check if hyperparameters are valid."""
        # d_model must be divisible by n_heads
        if self.d_model % self.n_heads != 0:
            return False
        return True


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for hyperparameter tuning.

    Uses tournament selection, uniform crossover, and Gaussian mutation.
    """

    def __init__(
        self,
        search_space: HyperparameterSpace,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elite_size: int = 5,
        seed: Optional[int] = None
    ):
        """
        Args:
            search_space: Hyperparameter search space
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
            elite_size: Number of best individuals to preserve
            seed: Random seed
        """
        self.search_space = search_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # History tracking
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual': [],
            'generation_stats': []
        }

        # Setup DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework."""
        # Create fitness class (maximize)
        if not hasattr(creator, 'FitnessMax'):
            creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create('Individual', list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Gene generators
        space = self.search_space
        self.toolbox.register('d_model', random.choice, space.d_model)
        self.toolbox.register('n_heads', random.choice, space.n_heads)
        self.toolbox.register('n_layers', random.choice, space.n_layers)
        self.toolbox.register('d_ff_mult', random.choice, space.d_ff_multiplier)
        self.toolbox.register('dropout', random.choice, space.dropout)
        self.toolbox.register('lr', random.choice, space.learning_rate)
        self.toolbox.register('batch_size', random.choice, space.batch_size)
        self.toolbox.register('weight_decay', random.choice, space.weight_decay)
        self.toolbox.register('profit_hidden', random.choice, space.profit_hidden)
        self.toolbox.register('risk_hidden', random.choice, space.risk_hidden)
        self.toolbox.register('max_risk', random.choice, space.max_risk_per_trade)
        self.toolbox.register('atr_mult', random.choice, space.atr_multiplier)

        # Individual creator
        self.toolbox.register(
            'individual',
            tools.initCycle,
            creator.Individual,
            (
                self.toolbox.d_model,
                self.toolbox.n_heads,
                self.toolbox.n_layers,
                self.toolbox.d_ff_mult,
                self.toolbox.dropout,
                self.toolbox.lr,
                self.toolbox.batch_size,
                self.toolbox.weight_decay,
                self.toolbox.profit_hidden,
                self.toolbox.risk_hidden,
                self.toolbox.max_risk,
                self.toolbox.atr_mult
            ),
            n=1
        )

        # Population creator
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register('select', tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register('mate', self._crossover)
        self.toolbox.register('mutate', self._mutate)

    def _individual_to_dict(self, ind: List) -> Dict:
        """Convert DEAP individual to hyperparameter dict."""
        return {
            'd_model': ind[0],
            'n_heads': ind[1],
            'n_layers': ind[2],
            'd_ff': ind[0] * ind[3],
            'dropout': ind[4],
            'learning_rate': ind[5],
            'batch_size': ind[6],
            'weight_decay': ind[7],
            'profit_hidden': ind[8],
            'risk_hidden': ind[9],
            'max_risk_per_trade': ind[10],
            'atr_multiplier': ind[11]
        }

    def _is_valid(self, ind: List) -> bool:
        """Check if individual has valid hyperparameters."""
        d_model = ind[0]
        n_heads = ind[1]
        return d_model % n_heads == 0

    def _repair(self, ind: List) -> List:
        """Repair invalid individual."""
        d_model = ind[0]
        n_heads = ind[1]

        # Adjust n_heads to be a divisor of d_model
        while d_model % n_heads != 0:
            valid_heads = [h for h in self.search_space.n_heads if d_model % h == 0]
            if valid_heads:
                ind[1] = random.choice(valid_heads)
            else:
                # Fallback: use closest valid d_model
                valid_d = [d for d in self.search_space.d_model if d % n_heads == 0]
                if valid_d:
                    ind[0] = min(valid_d, key=lambda x: abs(x - d_model))
            break

        return ind

    def _crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """Uniform crossover with repair."""
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Repair if necessary
        self._repair(ind1)
        self._repair(ind2)

        return ind1, ind2

    def _mutate(self, ind: List) -> Tuple[List]:
        """Mutate individual with parameter-specific mutations."""
        space = self.search_space.to_dict()
        param_names = list(space.keys())

        for i, (param, values) in enumerate(space.items()):
            if random.random() < self.mutation_rate:
                if isinstance(values[0], float):
                    # Continuous: add Gaussian noise or sample new
                    if random.random() < 0.5:
                        # Gaussian perturbation
                        std = (max(values) - min(values)) * 0.1
                        new_val = ind[i] + random.gauss(0, std)
                        new_val = max(min(values), min(max(values), new_val))
                        ind[i] = new_val
                    else:
                        ind[i] = random.choice(values)
                else:
                    # Discrete: sample new value
                    ind[i] = random.choice(values)

        # Repair if necessary
        self._repair(ind)

        return (ind,)

    def optimize(
        self,
        fitness_function: Callable[[Dict], float],
        callback: Optional[Callable[[int, Dict], None]] = None,
        early_stopping_generations: int = 20,
        target_fitness: Optional[float] = None
    ) -> Tuple[Dict, List[Dict]]:
        """
        Run genetic algorithm optimization.

        Args:
            fitness_function: Function that takes hyperparameters and returns fitness
            callback: Optional callback(generation, stats) called each generation
            early_stopping_generations: Stop if no improvement for N generations
            target_fitness: Stop if fitness reaches this value

        Returns:
            (best_hyperparameters, optimization_history)
        """
        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        print(f"Evaluating initial population of {len(population)} individuals...")
        fitnesses = []
        for i, ind in enumerate(population):
            if not self._is_valid(ind):
                ind = self._repair(ind)

            hp = self._individual_to_dict(ind)
            try:
                fitness = fitness_function(hp)
            except Exception as e:
                print(f"  Individual {i} failed: {e}")
                fitness = -1000.0

            ind.fitness.values = (fitness,)
            fitnesses.append(fitness)

        print(f"Initial best fitness: {max(fitnesses):.4f}")

        # Evolution loop
        best_fitness_ever = max(fitnesses)
        generations_without_improvement = 0

        for gen in range(self.n_generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                hp = self._individual_to_dict(ind)
                try:
                    fitness = fitness_function(hp)
                except Exception as e:
                    fitness = -1000.0
                ind.fitness.values = (fitness,)

            # Elitism: keep best from previous generation
            elite = tools.selBest(population, self.elite_size)
            population[:] = offspring + elite

            # Statistics
            fits = [ind.fitness.values[0] for ind in population]
            best_fit = max(fits)
            avg_fit = np.mean(fits)
            std_fit = np.std(fits)

            stats = {
                'generation': gen,
                'best_fitness': best_fit,
                'avg_fitness': avg_fit,
                'std_fitness': std_fit,
                'best_individual': self._individual_to_dict(tools.selBest(population, 1)[0])
            }

            self.history['best_fitness'].append(best_fit)
            self.history['avg_fitness'].append(avg_fit)
            self.history['generation_stats'].append(stats)

            # Check for improvement
            if best_fit > best_fitness_ever:
                best_fitness_ever = best_fit
                generations_without_improvement = 0
                self.history['best_individual'].append(stats['best_individual'])
            else:
                generations_without_improvement += 1

            # Callback
            if callback:
                callback(gen, stats)

            print(f"Gen {gen:3d}: Best={best_fit:.4f}, Avg={avg_fit:.4f}, Std={std_fit:.4f}")

            # Early stopping
            if target_fitness and best_fit >= target_fitness:
                print(f"Target fitness {target_fitness} reached!")
                break

            if generations_without_improvement >= early_stopping_generations:
                print(f"No improvement for {early_stopping_generations} generations. Stopping.")
                break

        # Return best individual
        best_ind = tools.selBest(population, 1)[0]
        best_hp = self._individual_to_dict(best_ind)

        return best_hp, self.history

    def save_results(self, filepath: str, best_hp: Dict):
        """Save optimization results to file."""
        results = {
            'best_hyperparameters': best_hp,
            'history': {
                'best_fitness': self.history['best_fitness'],
                'avg_fitness': self.history['avg_fitness']
            },
            'config': {
                'population_size': self.population_size,
                'n_generations': self.n_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def load_results(filepath: str) -> Dict:
        """Load optimization results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def create_fitness_function(
    train_loader,
    val_loader,
    input_dim: int,
    device: str,
    max_epochs: int = 10,
    fitness_evaluator: Optional[FitnessEvaluator] = None
) -> Callable[[Dict], float]:
    """
    Create a fitness function for the genetic algorithm.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input dimension
        device: Device to use
        max_epochs: Maximum training epochs per evaluation
        fitness_evaluator: Fitness evaluator instance

    Returns:
        Fitness function
    """
    from .trainer import Trainer
    from ..models import MetaAgent

    fitness_eval = fitness_evaluator or FitnessEvaluator()

    def fitness_function(hyperparameters: Dict) -> float:
        try:
            # Create model with hyperparameters
            model = MetaAgent(
                input_dim=input_dim,
                embedding_dim=hyperparameters['d_model'],
                n_heads=4,  # Fixed for meta-agent
                transformer_layers=hyperparameters['n_layers'],
                transformer_heads=hyperparameters['n_heads'],
                transformer_ff=hyperparameters['d_ff'],
                dropout=hyperparameters['dropout'],
                profit_hidden=hyperparameters['profit_hidden'],
                risk_hidden=hyperparameters['risk_hidden']
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=hyperparameters['learning_rate'],
                weight_decay=hyperparameters['weight_decay'],
                device=device
            )

            # Quick training
            trainer.train(max_epochs)

            # Evaluate on validation set
            metrics = trainer.evaluate(val_loader)

            # Calculate fitness
            fitness = fitness_eval.calculate_fitness(metrics)

            # Cleanup
            del model
            del trainer
            torch.cuda.empty_cache() if device == 'cuda' else None

            return fitness

        except Exception as e:
            print(f"Fitness evaluation failed: {e}")
            return -1000.0

    return fitness_function
