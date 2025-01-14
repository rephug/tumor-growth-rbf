#!/usr/bin/env python3
"""
immune_response.py

Models immune system response to tumor growth.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ImmuneParameters:
    """Parameters for immune response model."""
    recruitment_rate: float = 0.1  # Rate of immune cell recruitment
    killing_rate: float = 0.2      # Rate at which immune cells kill tumor cells
    immune_death_rate: float = 0.1 # Natural death rate of immune cells
    saturation_constant: float = 0.5  # Saturation constant for immune response
    chemokine_diffusion: float = 0.1  # Diffusion rate of chemokines
    activation_threshold: float = 0.2  # Threshold for immune activation

    def validate(self):
        """Validate parameter values."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")

class ImmuneResponse:
    """
    Models immune system response to tumor growth.
    Includes:
    - Immune cell recruitment and migration
    - Tumor cell killing by immune cells
    - Chemokine signaling
    """
    
    def __init__(self, params: Optional[ImmuneParameters] = None):
        self.params = params or ImmuneParameters()
        self.params.validate()
        
        # State variables
        self.immune_density = None
        self.chemokine_concentration = None
        
    def initialize(self, shape: Tuple[int, int]):
        """Initialize immune system state variables."""
        self.immune_density = np.zeros(shape)
        self.chemokine_concentration = np.zeros(shape)
        
    def update(self, 
              dt: float,
              tumor_density: np.ndarray,
              oxygen_concentration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update immune response for one time step.
        
        Args:
            dt: Time step
            tumor_density: Current tumor density
            oxygen_concentration: Current oxygen concentration
            
        Returns:
            Tuple of (immune effect on tumor, updated immune density)
        """
        if self.immune_density is None:
            self.initialize(tumor_density.shape)
            
        # Update chemokine concentration
        self._update_chemokines(dt, tumor_density)
        
        # Update immune cell density
        self._update_immune_cells(dt, tumor_density, oxygen_concentration)
        
        # Calculate immune effect on tumor
        immune_effect = self._calculate_immune_effect(tumor_density)
        
        return immune_effect, self.immune_density
        
    def _update_chemokines(self, dt: float, tumor_density: np.ndarray):
        """Update chemokine concentration."""
        # Chemokine production by tumor cells
        production = tumor_density
        
        # Diffusion of chemokines
        diffusion = self._compute_diffusion(self.chemokine_concentration)
        
        # Natural decay
        decay = 0.1 * self.chemokine_concentration
        
        # Update concentration
        self.chemokine_concentration += dt * (
            production + 
            self.params.chemokine_diffusion * diffusion - 
            decay
        )
        
        # Ensure non-negativity
        np.clip(self.chemokine_concentration, 0, None, out=self.chemokine_concentration)
        
    def _update_immune_cells(self,
                           dt: float,
                           tumor_density: np.ndarray,
                           oxygen_concentration: np.ndarray):
        """Update immune cell density."""
        # Recruitment based on chemokine gradient
        recruitment = (self.params.recruitment_rate * 
                     self.chemokine_concentration /
                     (self.params.saturation_constant + 
                      self.chemokine_concentration))
        
        # Movement towards chemokine gradient
        migration = self._compute_migration()
        
        # Death rate modulated by oxygen
        death_rate = self.params.immune_death_rate * (
            1.0 + 0.5 * (1.0 - oxygen_concentration)
        )
        
        # Update density
        self.immune_density += dt * (
            recruitment + 
            migration - 
            death_rate * self.immune_density
        )
        
        # Ensure non-negativity
        np.clip(self.immune_density, 0, None, out=self.immune_density)
        
    def _calculate_immune_effect(self, tumor_density: np.ndarray) -> np.ndarray:
        """Calculate immune system effect on tumor cells."""
        # Killing rate modulated by local immune cell density
        killing = (self.params.killing_rate * 
                  self.immune_density * 
                  tumor_density /
                  (self.params.saturation_constant + tumor_density))
        
        return -killing  # Negative effect on tumor growth
        
    def _compute_diffusion(self, field: np.ndarray) -> np.ndarray:
        """Compute diffusion term using finite differences."""
        return (np.roll(field, 1, axis=0) + 
                np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) + 
                np.roll(field, -1, axis=1) - 
                4 * field)
                
    def _compute_migration(self) -> np.ndarray:
        """Compute immune cell migration based on chemokine gradient."""
        # Compute chemokine gradients
        dx = np.roll(self.chemokine_concentration, -1, axis=1) - \
             np.roll(self.chemokine_concentration, 1, axis=1)
        dy = np.roll(self.chemokine_concentration, -1, axis=0) - \
             np.roll(self.chemokine_concentration, 1, axis=0)
             
        # Migration flux
        flux_x = self.immune_density * dx
        flux_y = self.immune_density * dy
        
        # Divergence of flux
        return -(np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1) +
                np.roll(flux_y, -1, axis=0) - np.roll(flux_y, 1, axis=0))
                
    def get_metrics(self) -> dict:
        """Calculate immune response metrics."""
        return {
            'total_immune_cells': float(np.sum(self.immune_density)),
            'max_immune_density': float(np.max(self.immune_density)),
            'mean_chemokine_conc': float(np.mean(self.chemokine_concentration)),
            'immune_coverage': float(np.mean(self.immune_density > 0.1))
        }