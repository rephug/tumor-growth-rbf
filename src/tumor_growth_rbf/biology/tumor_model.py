#!/usr/bin/env python3
"""
tumor_model.py

Integrated tumor growth model incorporating immune response, treatments,
and adaptive mesh refinement.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

from ..core.rbf_solver import RBFSolver
from ..core.pde_assembler import PDEAssembler
from ..core.mesh_handler import MeshHandler
from .immune_response import ImmuneResponse, ImmuneParameters
from .treatments import TreatmentModule, TreatmentParameters

logger = logging.getLogger(__name__)

@dataclass
class TumorParameters:
    """Parameters for tumor growth model."""
    # Growth parameters
    growth_rate: float = 0.1
    carrying_capacity: float = 1.0
    
    # Diffusion parameters
    diffusion_white: float = 0.1  # mm²/day in white matter
    diffusion_grey: float = 0.01  # mm²/day in grey matter
    
    # Oxygen dynamics
    oxygen_consumption: float = 0.1
    oxygen_diffusion: float = 1.0
    hypoxia_threshold: float = 0.1
    
    # Mesh parameters
    min_spacing: float = 0.01
    max_spacing: float = 0.1
    refinement_threshold: float = 0.1

    def validate(self):
        """Validate parameter values."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")

class TumorModel:
    """
    Comprehensive tumor growth model with immune response,
    treatments, and adaptive refinement.
    """
    
    def __init__(self,
                 domain_size: Tuple[float, float],
                 params: Optional[TumorParameters] = None,
                 immune_params: Optional[ImmuneParameters] = None,
                 treatment_params: Optional[TreatmentParameters] = None,
                 n_initial_points: int = 1000):
        """
        Initialize tumor model with all components.
        
        Args:
            domain_size: Physical domain size (Lx, Ly)
            params: Tumor growth parameters
            immune_params: Immune response parameters
            treatment_params: Treatment parameters
            n_initial_points: Initial number of mesh points
        """
        self.domain_size = domain_size
        self.params = params or TumorParameters()
        self.params.validate()
        
        # Initialize mesh and RBF solver
        self.mesh = MeshHandler(
            domain_size,
            min_spacing=self.params.min_spacing,
            max_spacing=self.params.max_spacing
        )
        self.mesh.initialize_points(n_initial_points, distribution="halton")
        
        self.rbf_solver = RBFSolver(epsilon=1.0, poly_degree=2)
        self.pde_assembler = PDEAssembler(self.rbf_solver)
        
        # Initialize biological components
        self.immune_system = ImmuneResponse(immune_params)
        self.treatment_module = TreatmentModule(treatment_params)
        
        # Initialize state variables
        self.tumor_density = None
        self.oxygen = None
        self.vessels = None
        self._initialize_state()
        
    def _initialize_state(self):
        """Initialize spatial distributions."""
        n_points = len(self.mesh.points)
        
        # Initial tumor: Gaussian distribution
        center = np.array(self.domain_size) / 2
        points = self.mesh.points
        distances = np.linalg.norm(points - center, axis=1)
        sigma = min(self.domain_size) / 10
        
        self.tumor_density = np.exp(-distances**2 / (2*sigma**2))
        self.oxygen = np.ones(n_points)
        self.vessels = np.zeros(n_points)
        
        # Initialize other components
        self.immune_system.initialize((n_points,))
        self.treatment_module.initialize((n_points,))
        
    def update(self, dt: float):
        """
        Update tumor state for one time step.
        
        Args:
            dt: Time step size
        """
        # Store previous state for refinement
        prev_density = self.tumor_density.copy()
        
        # 1. Update oxygen distribution
        self._update_oxygen(dt)
        
        # 2. Compute tumor growth and diffusion
        growth = self._compute_growth()
        diffusion = self._solve_diffusion()
        
        # 3. Update immune response
        immune_effect, immune_density = self.immune_system.update(
            dt, self.tumor_density, self.oxygen
        )
        
        # 4. Combine effects and update tumor density
        self.tumor_density += dt * (growth + diffusion + immune_effect)
        np.clip(self.tumor_density, 0, self.params.carrying_capacity, 
                out=self.tumor_density)
                
        # 5. Adapt mesh if needed
        self._adapt_mesh(prev_density)
        
    def _update_oxygen(self, dt: float):
        """Update oxygen distribution."""
        # Assemble oxygen diffusion operator
        oxygen_operator = self.pde_assembler.build_operator(
            self.mesh.points,
            self.mesh.neighbor_lists,
            operator="laplacian"
        )
        
        # Compute consumption and production terms
        consumption = (self.params.oxygen_consumption * 
                      self.tumor_density * self.oxygen)
        production = self.vessels
        
        # Update oxygen concentration
        rhs = (self.params.oxygen_diffusion * 
               (oxygen_operator @ self.oxygen) -
               consumption + production)
               
        self.oxygen += dt * rhs
        np.clip(self.oxygen, 0, 1, out=self.oxygen)
        
    def _compute_growth(self) -> np.ndarray:
        """Compute tumor growth term."""
        # Logistic growth modulated by oxygen
        growth = (self.params.growth_rate * 
                 self.tumor_density * 
                 (1 - self.tumor_density / self.params.carrying_capacity))
        
        # Oxygen dependence
        hypoxic = self.oxygen < self.params.hypoxia_threshold
        growth[hypoxic] *= 0.1  # Reduced growth in hypoxic regions
        
        return growth
        
    def _solve_diffusion(self) -> np.ndarray:
        """Solve diffusion using RBF-FD."""
        # Build diffusion operator
        diffusion_operator = self.pde_assembler.build_operator(
            self.mesh.points,
            self.mesh.neighbor_lists,
            operator="laplacian"
        )
        
        # Apply tissue-dependent diffusion coefficient
        # (simplified here - could be more complex based on tissue type)
        D = np.full_like(self.tumor_density, self.params.diffusion_white)
        
        return D * (diffusion_operator @ self.tumor_density)
        
    def _adapt_mesh(self, previous_density: np.ndarray):
        """Adapt mesh based on tumor density changes."""
        # Compute refinement indicator
        density_change = np.abs(self.tumor_density - previous_density)
        gradient = self._compute_gradient_magnitude()
        
        refinement_indicator = (
            density_change / np.max(density_change) +
            gradient / np.max(gradient)
        )
        
        # Refine and coarsen mesh
        self.mesh.refine_points(
            refinement_indicator,
            threshold=self.params.refinement_threshold
        )
        
        self.mesh.coarsen_points(
            refinement_indicator,
            threshold=self.params.refinement_threshold * 0.1
        )
        
        # Interpolate fields to new mesh if points changed
        if len(self.mesh.points) != len(previous_density):
            self._interpolate_to_new_mesh(previous_density)
            
    def _compute_gradient_magnitude(self) -> np.ndarray:
        """Compute magnitude of tumor density gradient."""
        # Build gradient operators
        grad_x = self.pde_assembler.build_operator(
            self.mesh.points,
            self.mesh.neighbor_lists,
            operator="gradient_x"
        )
        
        grad_y = self.pde_assembler.build_operator(
            self.mesh.points,
            self.mesh.neighbor_lists,
            operator="gradient_y"
        )
        
        # Compute gradient components
        dx = grad_x @ self.tumor_density
        dy = grad_y @ self.tumor_density
        
        return np.sqrt(dx**2 + dy**2)
        
    def _interpolate_to_new_mesh(self, previous_density: np.ndarray):
        """Interpolate fields to new mesh points."""
        # Use RBF interpolation to transfer fields to new points
        old_points = self.mesh.points[:-1]  # Previous points
        new_points = self.mesh.points  # Updated points
        
        # Interpolate each field
        self.tumor_density = self.rbf_solver.interpolate(
            old_points, new_points, previous_density
        )
        
        self.oxygen = self.rbf_solver.interpolate(
            old_points, new_points, self.oxygen
        )
        
        self.vessels = self.rbf_solver.interpolate(
            old_points, new_points, self.vessels
        )
        
    def apply_treatment(self, 
                       treatment_type: str,
                       **treatment_params):
        """Apply specified treatment."""
        effect, metrics = self.treatment_module.apply_treatment(
            treatment_type,
            self.tumor_density,
            self.oxygen,
            self.immune_system.immune_density,
            **treatment_params
        )
        
        self.tumor_density += effect
        np.clip(self.tumor_density, 0, None, out=self.tumor_density)
        
        return metrics
        
    def get_metrics(self) -> Dict:
        """Get comprehensive model metrics."""
        metrics = {
            "tumor": {
                "total_mass": float(np.sum(self.tumor_density)),
                "max_density": float(np.max(self.tumor_density)),
                "hypoxic_fraction": float(np.mean(
                    self.oxygen < self.params.hypoxia_threshold
                ))
            },
            "mesh": self.mesh.get_metrics(),
            "immune": self.immune_system.get_metrics(),
            "treatment": self.treatment_module.get_metrics()
        }
        
        return metrics

if __name__ == "__main__":
    # Example usage
    domain_size = (10.0, 10.0)  # 10mm x 10mm domain
    model = TumorModel(domain_size)
    
    # Simulate for 10 days
    dt = 0.1  # 0.1 day time step
    for t in range(100):
        model.update(dt)
        if t % 10 == 0:
            metrics = model.get_metrics()
            print(f"Day {t*dt:.1f}: {metrics}")
            
        # Apply treatment every 5 days
        if t % 50 == 0:
            model.apply_treatment("radiation", dose=2.0)