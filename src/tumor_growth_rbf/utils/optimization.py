#!/usr/bin/env python3
"""
optimization.py

Performance optimization utilities for tumor growth simulations.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import multiprocessing as mp
from functools import partial
import concurrent.futures
import time
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Optimizes computational performance of tumor simulations.
    """
    
    def __init__(self, 
                 n_processes: Optional[int] = None):
        self.n_processes = n_processes or mp.cpu_count()
        
    def optimize_rbf_weights(self,
                           points: np.ndarray,
                           neighbor_lists: list,
                           epsilon: float) -> np.ndarray:
        """Optimize RBF weight computation using parallel processing."""
        return self._compute_weights_parallel(points, neighbor_lists, epsilon)
            
    def _compute_weights_parallel(self,
                                points: np.ndarray,
                                neighbor_lists: list,
                                epsilon: float) -> np.ndarray:
        """Compute RBF weights using parallel CPU processing."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Create partial function with fixed parameters
            compute_func = partial(self._compute_single_weight,
                                 points=points,
                                 epsilon=epsilon)
            
            # Compute weights in parallel
            weights = list(executor.map(compute_func, neighbor_lists))
            
        return np.array(weights)
        
    @staticmethod
    def _compute_single_weight(neighbors: list,
                             points: np.ndarray,
                             epsilon: float) -> np.ndarray:
        """Compute RBF weights for a single point."""
        # Extract neighbor coordinates
        neighbor_coords = points[neighbors]
        
        # Compute distances
        center = neighbor_coords[0]
        distances = np.linalg.norm(neighbor_coords - center, axis=1)
        
        # Compute RBF matrix
        rbf_matrix = np.exp(-(epsilon * distances[:, np.newaxis])**2)
        
        # Solve for weights
        rhs = np.zeros(len(neighbors))
        rhs[0] = 1.0  # Delta function at center
        
        try:
            weights = np.linalg.solve(rbf_matrix, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(rbf_matrix, rhs, rcond=None)[0]
            
        return weights

class AdaptiveTimestepper:
    """
    Implements adaptive timestepping for efficient simulation.
    """
    
    def __init__(self,
                 initial_dt: float = 0.1,
                 tolerance: float = 1e-3,
                 safety_factor: float = 0.9):
        self.dt = initial_dt
        self.tolerance = tolerance
        self.safety_factor = safety_factor
        self.min_dt = initial_dt / 100
        self.max_dt = initial_dt * 10
        
    def compute_timestep(self,
                        model,
                        previous_state: np.ndarray) -> Tuple[float, float]:
        """
        Compute optimal timestep based on error estimate.
        """
        # Try current timestep
        state_copy = model.tumor_density.copy()
        model.update(self.dt)
        first_step = model.tumor_density.copy()
        
        # Take two half steps
        model.tumor_density = state_copy
        model.update(self.dt/2)
        model.update(self.dt/2)
        two_steps = model.tumor_density.copy()
        
        # Compute error estimate
        error = np.max(np.abs(two_steps - first_step))
        
        # Compute new timestep using PI control
        if error > 0:
            dt_new = self.safety_factor * self.dt * (self.tolerance/error)**0.5
            dt_new = min(max(dt_new, self.min_dt), self.max_dt)
        else:
            dt_new = self.max_dt
            
        # Restore original state
        model.tumor_density = state_copy
        
        return dt_new, error
        
    def step(self, model) -> Tuple[float, float]:
        """
        Take one adaptive step.
        """
        previous_state = model.tumor_density.copy()
        
        while True:
            dt_new, error = self.compute_timestep(model, previous_state)
            
            if error <= self.tolerance:
                # Step accepted
                self.dt = dt_new
                return self.dt, error
            else:
                # Step rejected, try again with smaller step
                self.dt = max(dt_new, self.min_dt)
                model.tumor_density = previous_state.copy()