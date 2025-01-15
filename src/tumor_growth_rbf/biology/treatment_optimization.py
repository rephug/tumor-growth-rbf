"""
treatment_optimization.py

Advanced treatment optimization module for tumor growth model.
Analyzes treatment response patterns and optimizes treatment scheduling
based on tumor characteristics, tissue properties, and cell cycle states.
Incorporates uncertainty quantification for robust optimization.
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TreatmentConstraints:
    """Constraints for treatment optimization."""
    # Dose constraints
    max_daily_dose: float = 2.0  # Gy for radiation
    max_cumulative_dose: float = 60.0  # Gy for radiation
    min_interval: float = 1.0  # Minimum days between treatments
    
    # Time constraints
    max_treatment_duration: float = 60.0  # Days
    allowed_days: List[int] = None  # Days of week when treatment is allowed (1-7)
    
    # Tissue constraints
    max_normal_tissue_dose: float = 45.0  # Gy for critical structures
    max_organ_at_risk_dose: Dict[str, float] = None  # Organ-specific limits

@dataclass
class OptimizationParameters:
    """Parameters for treatment optimization."""
    # Optimization goals
    target_reduction: float = 0.5  # Target tumor volume reduction
    normal_tissue_weight: float = 1.0  # Weight for normal tissue protection
    immune_response_weight: float = 0.5  # Weight for immune system considerations
    
    # Algorithm parameters
    population_size: int = 100  # For genetic algorithm
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Robustness parameters
    n_samples: int = 20  # Number of samples for uncertainty analysis
    confidence_level: float = 0.95  # For robust optimization

class TreatmentOptimizer:
    """
    Optimizes treatment schedules considering tumor biology,
    tissue properties, and practical constraints.
    
    Features:
    - Multi-objective optimization
    - Robust scheduling under uncertainty
    - Tissue-specific dose adaptation
    - Cell cycle-aware timing
    """
    
    def __init__(self,
                 model,
                 constraints: Optional[TreatmentConstraints] = None,
                 optimization_params: Optional[OptimizationParameters] = None):
        """
        Initialize treatment optimizer.
        
        Args:
            model: TumorModel instance
            constraints: Treatment constraints
            optimization_params: Optimization parameters
        """
        self.model = model
        self.constraints = constraints or TreatmentConstraints()
        self.params = optimization_params or OptimizationParameters()
        
        # Initialize treatment history tracking
        self.treatment_history = []
        self.optimization_results = {}
        
    def optimize_schedule(self,
                         initial_state: Dict,
                         treatment_types: List[str],
                         duration: float) -> Dict:
        """
        Optimize treatment schedule for given conditions.
        
        Args:
            initial_state: Initial tumor and tissue state
            treatment_types: List of available treatment types
            duration: Total treatment duration in days
            
        Returns:
            Optimized treatment schedule with timing and doses
        """
        # Set up optimization problem
        n_timepoints = int(duration / self.constraints.min_interval)
        
        # Define decision variables:
        # - Treatment timing (binary variables for each possible timepoint)
        # - Treatment doses (continuous variables for each treatment)
        n_variables = n_timepoints * len(treatment_types)
        
        # Define bounds and constraints
        bounds = self._create_bounds(n_timepoints, treatment_types)
        constraints = self._create_constraints(n_timepoints, treatment_types)
        
        # Run optimization using genetic algorithm
        result = self._run_genetic_algorithm(
            n_variables,
            bounds,
            constraints,
            initial_state,
            treatment_types
        )
        
        # Convert optimization result to treatment schedule
        schedule = self._create_schedule(
            result.x,
            n_timepoints,
            treatment_types
        )
        
        # Validate and refine schedule
        schedule = self._refine_schedule(schedule, initial_state)
        
        return schedule
        
    def analyze_response_pattern(self,
                               treatment_history: List[Dict],
                               metrics_history: List[Dict]) -> Dict:
        """
        Analyze tumor response patterns to previous treatments.
        
        Args:
            treatment_history: List of previous treatments
            metrics_history: Corresponding tumor metrics
            
        Returns:
            Dictionary of response pattern analysis
        """
        analysis = {
            'response_by_type': {},
            'timing_effects': {},
            'tissue_specific_response': {},
            'cell_cycle_effects': {}
        }
        
        # Analyze response by treatment type
        for treatment_type in set(t['type'] for t in treatment_history):
            responses = self._analyze_treatment_type(
                treatment_type,
                treatment_history,
                metrics_history
            )
            analysis['response_by_type'][treatment_type] = responses
            
        # Analyze timing effects
        timing_effects = self._analyze_timing_effects(
            treatment_history,
            metrics_history
        )
        analysis['timing_effects'] = timing_effects
        
        # Analyze tissue-specific responses
        if hasattr(self.model, 'tissue_model'):
            tissue_responses = self._analyze_tissue_responses(
                treatment_history,
                metrics_history
            )
            analysis['tissue_specific_response'] = tissue_responses
            
        # Analyze cell cycle effects
        cell_cycle_effects = self._analyze_cell_cycle_effects(
            treatment_history,
            metrics_history
        )
        analysis['cell_cycle_effects'] = cell_cycle_effects
        
        return analysis
        
    def predict_outcomes(self,
                        schedule: Dict,
                        initial_state: Dict,
                        n_samples: int = None) -> Dict:
        """
        Predict treatment outcomes with uncertainty quantification.
        
        Args:
            schedule: Proposed treatment schedule
            initial_state: Initial tumor state
            n_samples: Number of samples for uncertainty analysis
            
        Returns:
            Dictionary of predicted outcomes with confidence intervals
        """
        if n_samples is None:
            n_samples = self.params.n_samples
            
        predictions = {
            'volume_reduction': [],
            'normal_tissue_dose': [],
            'immune_response': []
        }
        
        # Run multiple simulations with parameter variation
        with ProcessPoolExecutor() as executor:
            futures = []
            for _ in range(n_samples):
                perturbed_state = self._perturb_parameters(initial_state)
                futures.append(
                    executor.submit(
                        self._simulate_schedule,
                        schedule,
                        perturbed_state
                    )
                )
                
            # Collect results
            for future in futures:
                result = future.result()
                for key in predictions:
                    predictions[key].append(result[key])
                    
        # Calculate statistics
        statistics = {}
        for key, values in predictions.items():
            values = np.array(values)
            statistics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, 2.5)),
                'ci_upper': float(np.percentile(values, 97.5))
            }
            
        return statistics
        
    def _run_genetic_algorithm(self,
                             n_variables: int,
                             bounds: List[Tuple],
                             constraints: List[Dict],
                             initial_state: Dict,
                             treatment_types: List[str]) -> optimize.OptimizeResult:
        """Run genetic algorithm optimization."""
        # Initialize population
        population = self._initialize_population(
            n_variables,
            bounds,
            self.params.population_size
        )
        
        best_fitness = float('-inf')
        best_solution = None
        
        for generation in range(self.params.generations):
            # Evaluate fitness
            fitness_values = [
                self._evaluate_fitness(
                    solution,
                    initial_state,
                    treatment_types,
                    constraints
                )
                for solution in population
            ]
            
            # Update best solution
            max_fitness_idx = np.argmax(fitness_values)
            if fitness_values[max_fitness_idx] > best_fitness:
                best_fitness = fitness_values[max_fitness_idx]
                best_solution = population[max_fitness_idx]
                
            # Create next generation
            new_population = []
            
            while len(new_population) < self.params.population_size:
                # Selection
                parent1 = self._tournament_select(population, fitness_values)
                parent2 = self._tournament_select(population, fitness_values)
                
                # Crossover
                if np.random.random() < self.params.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                    
                # Mutation
                self._mutate(child1)
                self._mutate(child2)
                
                new_population.extend([child1, child2])
                
            population = new_population[:self.params.population_size]
            
        return optimize.OptimizeResult(
            x=best_solution,
            fun=-best_fitness,  # Convert maximization to minimization
            success=True,
            message="Optimization completed successfully"
        )
        
    def _evaluate_fitness(self,
                         solution: np.ndarray,
                         initial_state: Dict,
                         treatment_types: List[str],
                         constraints: List[Dict]) -> float:
        """
        Evaluate fitness of a treatment schedule.
        
        Considers:
        - Tumor volume reduction
        - Normal tissue protection
        - Treatment timing practicality
        - Robustness to uncertainty
        """
        # Convert solution to schedule
        schedule = self._create_schedule(
            solution,
            len(solution) // len(treatment_types),
            treatment_types
        )
        
        # Check constraint violations
        if not self._check_constraints(schedule, constraints):
            return float('-inf')
            
        # Predict outcomes
        outcomes = self.predict_outcomes(schedule, initial_state)
        
        # Calculate fitness components
        tumor_reduction = outcomes['volume_reduction']['mean']
        normal_tissue_score = 1.0 - (
            outcomes['normal_tissue_dose']['mean'] /
            self.constraints.max_normal_tissue_dose
        )
        immune_score = outcomes['immune_response']['mean']
        
        # Combine components with weights
        fitness = (tumor_reduction +
                  self.params.normal_tissue_weight * normal_tissue_score +
                  self.params.immune_response_weight * immune_score)
                  
        # Penalize uncertainty
        uncertainty_penalty = (
            outcomes['volume_reduction']['std'] +
            outcomes['normal_tissue_dose']['std']
        )
        
        return fitness - 0.1 * uncertainty_penalty
        
    def _analyze_treatment_type(self,
                              treatment_type: str,
                              history: List[Dict],
                              metrics: List[Dict]) -> Dict:
        """Analyze response patterns for specific treatment type."""
        type_indices = [
            i for i, t in enumerate(history)
            if t['type'] == treatment_type
        ]
        
        responses = []
        for idx in type_indices:
            # Calculate response metrics
            pre_vol = metrics[idx]['tumor']['total_mass']
            post_vol = metrics[idx + 1]['tumor']['total_mass']
            response = (pre_vol - post_vol) / pre_vol
            
            # Get treatment parameters
            treatment = history[idx]
            
            responses.append({
                'response_ratio': float(response),
                'dose': treatment.get('dose', None),
                'timing': treatment['time'],
                'cell_cycle_state': {
                    phase: metrics[idx]['cell_populations'][f'{phase.lower()}_fraction']
                    for phase in ['G1', 'S', 'G2', 'M']
                }
            })
            
        return responses
        
    def _analyze_timing_effects(self,
                              history: List[Dict],
                              metrics: List[Dict]) -> Dict:
        """Analyze effects of treatment timing."""
        intervals = np.diff([t['time'] for t in history])
        responses = []
        
        for i in range(len(intervals)):
            pre_vol = metrics[i]['tumor']['total_mass']
            post_vol = metrics[i + 1]['tumor']['total_mass']
            response = (pre_vol - post_vol) / pre_vol
            
            responses.append({
                'interval': float(intervals[i]),
                'response': float(response)
            })
            
        # Calculate timing correlations
        intervals = [r['interval'] for r in responses]
        responses = [r['response'] for r in responses]
        
        correlation = np.corrcoef(intervals, responses)[0, 1]
        
        return {
            'interval_response_correlation': float(correlation),
            'optimal_interval': float(intervals[np.argmax(responses)])
        }
        
    def _analyze_tissue_responses(self,
                                history: List[Dict],
                                metrics: List[Dict]) -> Dict:
        """Analyze tissue-specific treatment responses."""
        tissue_responses = {}
        
        if not hasattr(self.model, 'tissue_model'):
            return tissue_responses
            
        for tissue_type in self.model.tissue_model.tissue_map.unique():
            responses = []
            
            for i in range(len(history)):
                tissue_mask = self.model.tissue_model.tissue_map == tissue_type
                pre_vol = metrics[i]['tumor']['total_mass']
                post_vol = metrics[i + 1]['tumor']['total_mass']
                
                response = (pre_vol - post_vol) / pre_vol
                responses.append(float(response))
                
            tissue_responses[tissue_type.value] = {
                'mean_response': float(np.mean(responses)),
                'response_std': float(np.std(responses))
            }
            
        return tissue_responses
        
    def _analyze_cell_cycle_effects(self,
                                  history: List[Dict],
                                  metrics: List[Dict]) -> Dict:
        """Analyze cell cycle-specific treatment responses."""
        phase_responses = {}
        
        for phase in ['G1', 'S', 'G2', 'M']:
            responses = []
            
            for i in range(len(history)):
                phase_fraction = metrics[i]['cell_populations'][f'{phase.lower()}_fraction']
                pre_vol = metrics[i]['tumor']['total_mass']
                post_vol = metrics[i + 1]['tumor']['total_mass']
                
                response = (pre_vol - post_vol) / pre_vol
                responses.append({
                    'phase_fraction': float(phase_fraction),
                    'response': float(response)
                })
                
            # Calculate correlation