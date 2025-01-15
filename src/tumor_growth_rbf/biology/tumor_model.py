#!/usr/bin/env python3
"""
tumor_model.py

Integrated tumor growth model incorporating cell populations, tissue heterogeneity,
immune response, treatments, and adaptive mesh refinement.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging

from ..core.rbf_solver import RBFSolver
from ..core.pde_assembler import PDEAssembler
from ..core.mesh_handler import MeshHandler
from .immune_response import ImmuneResponse, ImmuneParameters
from .treatments import TreatmentModule, TreatmentParameters
from .cell_populations import CellPopulationModel, CellCycleParameters
from .tissue_properties import TissueModel, TissueParameters, TissueType

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
    Comprehensive tumor growth model with tissue heterogeneity, cell populations,
    immune response, treatments, and adaptive refinement.
    """
    
    def __init__(self,
                 domain_size: Tuple[float, float],
                 params: Optional[TumorParameters] = None,
                 immune_params: Optional[ImmuneParameters] = None,
                 treatment_params: Optional[TreatmentParameters] = None,
                 cell_cycle_params: Optional[CellCycleParameters] = None,
                 tissue_params: Optional[TissueParameters] = None,
                 n_initial_points: int = 1000):
        """
        Initialize tumor model with all components.
        
        Args:
            domain_size: Physical domain size (Lx, Ly)
            params: Tumor growth parameters
            immune_params: Immune response parameters
            treatment_params: Treatment parameters
            cell_cycle_params: Cell cycle parameters
            tissue_params: Tissue-specific parameters
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
        self.cell_populations = CellPopulationModel(cell_cycle_params)
        self.tissue_model = TissueModel(tissue_params)
        
        # Initialize state variables
        self.tumor_density = None
        self.oxygen = None
        self.vessels = None
        
        # Initialize tissue-specific property maps
        self.diffusion_map = None
        self.growth_modifier_map = None
        self.oxygen_perfusion_map = None
        
        self._initialize_state()

    def load_tissue_data(self,
                        tissue_image: np.ndarray,
                        tissue_labels: Dict[int, TissueType],
                        vessel_image: Optional[np.ndarray] = None):
        """Load tissue type information from medical imaging data."""
        self.tissue_model.initialize_from_image(
            tissue_image, tissue_labels, vessel_image
        )
        self._update_tissue_properties()
        
    def _initialize_state(self):
        """Initialize spatial distributions."""
        n_points = len(self.mesh.points)
        
        # Create initial Gaussian distribution
        center = np.array(self.domain_size) / 2
        points = self.mesh.points
        distances = np.linalg.norm(points - center, axis=1)
        sigma = min(self.domain_size) / 10
        initial_density = np.exp(-distances**2 / (2*sigma**2))
        
        # Initialize cell populations with realistic proportions
        self.cell_populations.initialize(initial_density.shape)
        self.cell_populations.populations['G1'] = initial_density * 0.6  # 60% in G1
        self.cell_populations.populations['S'] = initial_density * 0.2   # 20% in S
        self.cell_populations.populations['G2'] = initial_density * 0.15 # 15% in G2
        self.cell_populations.populations['M'] = initial_density * 0.05  # 5% in M
        
        # Set total tumor density from cell populations
        self.tumor_density = self.cell_populations.get_total_density()
        
        # Initialize other state variables
        self.oxygen = np.ones(n_points)
        self.vessels = np.zeros(n_points)
        
        # Initialize tissue properties if not already set
        if self.diffusion_map is None:
            self.diffusion_map = np.full(n_points, 1.0)
            self.growth_modifier_map = np.full(n_points, 1.0)
            self.oxygen_perfusion_map = np.full(n_points, 1.0)
        
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
        
        # 1. Update oxygen distribution with tissue-specific perfusion
        self._update_oxygen(dt)
        
        # 2. Update cell populations with current oxygen levels
        self.cell_populations.update(dt, self.oxygen)
        
        # 3. Get total density for transport calculations
        self.tumor_density = self.cell_populations.get_total_density()
        
        # 4. Compute tissue-modified diffusion and growth
        diffusion = self._solve_diffusion()
        growth = self._compute_growth()
        
        # 5. Update immune response
        immune_effect, immune_density = self.immune_system.update(
            dt, self.tumor_density, self.oxygen
        )
        
        # 6. Apply combined effects to tumor density
        self.tumor_density += dt * (growth + diffusion + immune_effect)
        np.clip(self.tumor_density, 0, self.params.carrying_capacity, 
                out=self.tumor_density)
        
        # 7. Update tissue state based on tumor evolution
        self._update_tissue_state()
        
        # 8. Redistribute total density across populations
        self._redistribute_populations()
        
        # 9. Adapt mesh if needed
        self._adapt_mesh(prev_density)
        
    def _update_oxygen(self, dt: float):
        """Update oxygen distribution with tissue-specific perfusion."""
        oxygen_operator = self.pde_assembler.build_operator(
            self.mesh.points,
            self.mesh.neighbor_lists,
            operator="laplacian"
        )
        
        # Calculate oxygen consumption based on cell cycle phases
        proliferating_density = sum(
            self.cell_populations.populations[phase]
            for phase in ['S', 'G2', 'M']  # High oxygen consumption phases
        ) + 0.5 * self.cell_populations.populations['G1']  # Lower consumption in G1
        
        # Include tissue-specific oxygen dynamics
        consumption = (self.params.oxygen_consumption * 
                      proliferating_density * self.oxygen)
        
        # Production includes vessels and tissue-specific perfusion
        production = (self.vessels + 
                     self.oxygen_perfusion_map * (1 - self.tumor_density))
        
        rhs = (self.params.oxygen_diffusion * 
               (oxygen_operator @ self.oxygen) -
               consumption + production)
               
        self.oxygen += dt * rhs
        np.clip(self.oxygen, 0, 1, out=self.oxygen)
        
    def _compute_growth(self) -> np.ndarray:
        """Compute tumor growth term with tissue-specific modifiers."""
        # Base logistic growth
        growth = (self.params.growth_rate * 
                 self.tumor_density * 
                 (1 - self.tumor_density / self.params.carrying_capacity))
        
        # Apply tissue-specific growth modifiers
        growth *= self.growth_modifier_map
        
        # Oxygen dependence
        hypoxic = self.oxygen < self.params.hypoxia_threshold
        growth[hypoxic] *= 0.1
        
        return growth
        
    def _solve_diffusion(self) -> np.ndarray:
        """Solve diffusion using RBF-FD with tissue-specific coefficients."""
        diffusion_operator = self.pde_assembler.build_operator(
            self.mesh.points,
            self.mesh.neighbor_lists,
            operator="laplacian"
        )
        
        # Base diffusion coefficient modified by tissue properties
        D = self.params.diffusion_white * self.diffusion_map
        
        return D * (diffusion_operator @ self.tumor_density)
        
    def _update_tissue_state(self):
        """Update tissue state based on tumor evolution."""
        if self.tissue_model is None:
            return
            
        # Identify necrotic regions
        necrotic = (self.oxygen < self.params.hypoxia_threshold/5) & \
                  (self.tumor_density > 0.1)
                  
        # Update tissue model
        self.tissue_model.update_tissue_state(necrotic)
        
        # Update tissue properties
        self._update_tissue_properties()
        
    def _update_tissue_properties(self):
        """Update spatial maps of tissue-dependent properties."""
        if self.tissue_model is None:
            return
            
        # Get current tissue property maps
        self.diffusion_map = self.tissue_model.get_diffusion_coefficient_map()
        self.growth_modifier_map = self.tissue_model.get_growth_modifier_map()
        self.oxygen_perfusion_map = self.tissue_model.get_oxygen_perfusion_map()
        
        # Update vessel information if available
        if self.tissue_model.vessel_map is not None:
            self.vessels = self.tissue_model.vessel_map.astype(float)
        
    def _redistribute_populations(self):
        """Redistribute total density across cell populations."""
        if np.any(self.tumor_density > 0):
            total_before = self.cell_populations.get_total_density()
            for pop_name in self.cell_populations.populations:
                if np.any(total_before > 0):
                    fraction = self.cell_populations.populations[pop_name] / \
                             (total_before + 1e-10)
                    self.cell_populations.populations[pop_name] = \
                        fraction * self.tumor_density
                        
    def _adapt_mesh(self, previous_density: np.ndarray):
        """Adapt mesh based on tumor density changes."""
        # Compute refinement indicator
        density_change = np.abs(self.tumor_density - previous_density)
        gradient = self._compute_gradient_magnitude()
        
        refinement_indicator = (
            density_change / (np.max(density_change) + 1e-10) +
            gradient / (np.max(gradient) + 1e-10)
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
        
        # Interpolate fields if mesh changed
        if len(self.mesh.points) != len(previous_density):
            self._interpolate_to_new_mesh(previous_density)
            
    def _interpolate_to_new_mesh(self, previous_density: np.ndarray):
        """Interpolate fields to new mesh points."""
        old_points = self.mesh.points[:-1]
        new_points = self.mesh.points
        
        # Interpolate each cell population
        for pop_name in self.cell_populations.populations:
            self.cell_populations.populations[pop_name] = self.rbf_solver.interpolate(
                old_points, new_points, 
                self.cell_populations.populations[pop_name]
            )
        
        # Interpolate tissue property maps
        if self.diffusion_map is not None:
            self.diffusion_map = self.rbf_solver.interpolate(
                old_points, new_points, self.diffusion_map
            )
            self.growth_modifier_map = self.rbf_solver.interpolate(
                old_points, new_points, self.growth_modifier_map
            )
            self.oxygen_perfusion_map = self.rbf_solver.interpolate(
                old_points, new_points, self.oxygen_perfusion_map
            )
        
        # Interpolate other fields
        self.oxygen = self.rbf_solver.interpolate(
            old_points, new_points, self.oxygen
        )
        
        self.vessels = self.rbf_solver.interpolate(
            old_points, new_points, self.vessels
        )
        
        # Update total density
        self.tumor_density = self.cell_populations.get_total_density()
        
    def _compute_gradient_magnitude(self) -> np.ndarray:
        """Compute magnitude of tumor density gradient."""
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
        
        dx = grad_x @ self.tumor_density
        dy = grad_y @ self.tumor_density
        
        return np.sqrt(dx**2 + dy**2)
        
    def apply_treatment(self,
                       treatment_type: str,
                       **treatment_params):
        """
        Apply specified treatment with tissue-specific modifiers.
        
        Args:
            treatment_type: Type of treatment
            **treatment_params: Additional treatment parameters
        
        Returns:
            Treatment metrics
        """
        # Get tissue-specific treatment modifiers if available
        radiation_mod = drug_mod = None
        if self.tissue_model is not None:
            radiation_mod, drug_mod = self.tissue_model.get_treatment_modifier_maps()
            
        # Apply treatment with modifiers
        if treatment_type == "radiation" and radiation_mod is not None:
            treatment_params["radiation_modifier"] = radiation_mod
        elif treatment_type in ["chemo", "chemotherapy"] and drug_mod is not None:
            treatment_params["drug_modifier"] = drug_mod
            
        # Apply treatment to cell populations
        effects, metrics = self.treatment_module.apply_treatment(
            treatment_type,
            self.cell_populations.populations,
            self.oxygen,
            self.immune_system.immune_density,
            **treatment_params
        )
        
        # Apply effects to each population
        if isinstance(effects, dict):
            for pop_name, effect in effects.items():
                self.cell_populations.populations[pop_name] += effect
                np.clip(self.cell_populations.populations[pop_name], 0, None, 
                       out=self.cell_populations.populations[pop_name])
        else:
            # Handle legacy single-effect treatment responses
            effect_fraction = effects / (self.tumor_density + 1e-10)
            for pop_name in self.cell_populations.populations:
                self.cell_populations.populations[pop_name] *= (1 + effect_fraction)
                np.clip(self.cell_populations.populations[pop_name], 0, None, 
                       out=self.cell_populations.populations[pop_name])
        
        # Update total density
        self.tumor_density = self.cell_populations.get_total_density()
        
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
            "cell_populations": self.cell_populations.get_metrics(),
            "mesh": self.mesh.get_metrics(),
            "immune": self.immune_system.get_metrics(),
            "treatment": self.treatment_module.get_metrics()
        }
        
        # Add tissue metrics if tissue model is active
        if self.tissue_model is not None:
            metrics["tissue"] = self.tissue_model.get_metrics()
            
        return metrics

if __name__ == "__main__":
    # Example usage
    domain_size = (10.0, 10.0)  # 10mm x 10mm domain
    model = TumorModel(domain_size)
    
    # Optional: Load tissue data from medical imaging
    # tissue_image = ... # Load tissue type map
    # vessel_image = ... # Load vessel map
    # tissue_labels = {
    #     1: TissueType.WHITE_MATTER,
    #     2: TissueType.GRAY_MATTER,
    #     3: TissueType.CSF
    # }
    # model.load_tissue_data(tissue_image, tissue_labels, vessel_image)
    
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