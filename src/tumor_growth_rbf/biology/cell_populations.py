"""
cell_populations.py
Handles multiple cell populations and cell cycle dynamics.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class CellCycleParameters:
    """Parameters for cell cycle transitions and population dynamics."""
    # Cell cycle transition rates (hours^-1)
    # Based on typical mammalian cell cycle durations
    g1_to_s_rate: float = 1/10.0  # G1->S: ~10 hours
    s_to_g2_rate: float = 1/8.0   # S->G2: ~8 hours
    g2_to_m_rate: float = 1/4.0   # G2->M: ~4 hours
    m_to_g1_rate: float = 1/2.0   # M->G1: ~2 hours
    
    # Oxygen-dependent parameters
    hypoxia_threshold: float = 0.1  # O2 level below which cells become quiescent
    severe_hypoxia_threshold: float = 0.01  # O2 level for necrosis
    
    # Survival parameters
    necrosis_rate: float = 0.1  # Rate of necrotic cell clearance
    quiescent_survival_time: float = 48.0  # Hours cells can survive in quiescence
    
    def validate(self):
        """Validate parameter values."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")

class CellPopulationModel:
    """
    Manages multiple cell populations including cell cycle states.
    Populations tracked:
    - Proliferating cells in each phase (G1, S, G2, M)
    - Quiescent cells
    - Necrotic cells
    """
    
    def __init__(self, params: Optional[CellCycleParameters] = None):
        self.params = params or CellCycleParameters()
        self.params.validate()
        
        # Initialize population arrays
        self.populations = {
            'G1': None,
            'S': None,
            'G2': None,
            'M': None,
            'Q': None,  # Quiescent
            'N': None   # Necrotic
        }
        
        # Track time in quiescence for each spatial point
        self.quiescence_time = None
        
    def initialize(self, shape: Tuple[int, ...]):
        """Initialize cell populations."""
        for key in self.populations:
            self.populations[key] = np.zeros(shape)
            
        self.quiescence_time = np.zeros(shape)
        
        # Start with cells in G1
        self.populations['G1'] = np.ones(shape) * 0.1
        
    def update(self, dt: float, oxygen: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update cell populations for one time step.
        
        Args:
            dt: Time step in hours
            oxygen: Oxygen concentration field
            
        Returns:
            Dictionary of updated population fields
        """
        # Handle transitions based on oxygen levels
        self._handle_oxygen_effects(oxygen, dt)
        
        # Update cell cycle transitions
        self._update_cell_cycle(dt)
        
        # Update quiescence and necrosis
        self._update_quiescent_cells(dt)
        self._clear_necrotic_cells(dt)
        
        return self.populations
        
    def _handle_oxygen_effects(self, oxygen: np.ndarray, dt: float):
        """Handle transitions due to oxygen levels."""
        # Severe hypoxia -> necrosis
        severe_hypoxia = oxygen < self.params.severe_hypoxia_threshold
        for phase in ['G1', 'S', 'G2', 'M', 'Q']:
            self.populations['N'][severe_hypoxia] += self.populations[phase][severe_hypoxia]
            self.populations[phase][severe_hypoxia] = 0
            
        # Moderate hypoxia -> quiescence
        hypoxia = (oxygen >= self.params.severe_hypoxia_threshold) & \
                 (oxygen < self.params.hypoxia_threshold)
        for phase in ['G1', 'S', 'G2', 'M']:
            self.populations['Q'][hypoxia] += self.populations[phase][hypoxia]
            self.populations[phase][hypoxia] = 0
            
        # Recovery from quiescence under normal oxygen
        normal_oxygen = oxygen >= self.params.hypoxia_threshold
        recovering = normal_oxygen & (self.populations['Q'] > 0)
        self.populations['G1'][recovering] += self.populations['Q'][recovering]
        self.populations['Q'][recovering] = 0
        self.quiescence_time[recovering] = 0
        
    def _update_cell_cycle(self, dt: float):
        """Update cell cycle transitions."""
        # G1 -> S
        g1_to_s = self.populations['G1'] * (1 - np.exp(-self.params.g1_to_s_rate * dt))
        self.populations['G1'] -= g1_to_s
        self.populations['S'] += g1_to_s
        
        # S -> G2
        s_to_g2 = self.populations['S'] * (1 - np.exp(-self.params.s_to_g2_rate * dt))
        self.populations['S'] -= s_to_g2
        self.populations['G2'] += s_to_g2
        
        # G2 -> M
        g2_to_m = self.populations['G2'] * (1 - np.exp(-self.params.g2_to_m_rate * dt))
        self.populations['G2'] -= g2_to_m
        self.populations['M'] += g2_to_m
        
        # M -> G1 (division)
        m_to_g1 = self.populations['M'] * (1 - np.exp(-self.params.m_to_g1_rate * dt))
        self.populations['M'] -= m_to_g1
        self.populations['G1'] += 2 * m_to_g1  # Multiplication by 2 for cell division
        
    def _update_quiescent_cells(self, dt: float):
        """Update quiescent cell population."""
        quiescent_mask = self.populations['Q'] > 0
        self.quiescence_time[quiescent_mask] += dt
        
        # Death of cells in prolonged quiescence
        death_mask = self.quiescence_time > self.params.quiescent_survival_time
        self.populations['N'][death_mask] += self.populations['Q'][death_mask]
        self.populations['Q'][death_mask] = 0
        self.quiescence_time[death_mask] = 0
        
    def _clear_necrotic_cells(self, dt: float):
        """Clear necrotic cells."""
        clearance = self.populations['N'] * (1 - np.exp(-self.params.necrosis_rate * dt))
        self.populations['N'] -= clearance
        
    def get_total_density(self) -> np.ndarray:
        """Get total cell density across all populations."""
        return sum(pop for pop in self.populations.values())
        
    def get_metrics(self) -> Dict:
        """Calculate population metrics."""
        total_cells = self.get_total_density()
        return {
            'total_cells': float(np.sum(total_cells)),
            'proliferating_fraction': float(np.mean(
                sum(self.populations[p] for p in ['G1', 'S', 'G2', 'M']) / 
                (total_cells + 1e-10)
            )),
            'quiescent_fraction': float(np.mean(
                self.populations['Q'] / (total_cells + 1e-10)
            )),
            'necrotic_fraction': float(np.mean(
                self.populations['N'] / (total_cells + 1e-10)
            )),
            'mean_quiescence_time': float(np.mean(
                self.quiescence_time[self.populations['Q'] > 0]
            )) if np.any(self.populations['Q'] > 0) else 0.0
        }