"""
treatments.py

Models various treatment modalities for tumor therapy with cell cycle-specific effects.
Based on experimental data from radiation biology and chemotherapy studies.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class TreatmentParameters:
    """Parameters for different treatment modalities with cell cycle specificity."""
    # Radiation therapy parameters
    radiation_sensitivity: float = 0.3      # Base sensitivity to radiation
    oxygen_enhancement: float = 2.0         # Oxygen enhancement ratio
    fractionation_alpha: float = 0.15      # Linear term in LQ model
    fractionation_beta: float = 0.05       # Quadratic term in LQ model
    
    # Cell cycle-specific radiation sensitivity multipliers
    # Based on experimental measurements of phase-specific radiosensitivity
    radiation_g1_factor: float = 1.0       # G1 phase relative sensitivity
    radiation_s_factor: float = 0.5        # S phase (most resistant)
    radiation_g2_factor: float = 1.5       # G2 phase (more sensitive)
    radiation_m_factor: float = 2.0        # M phase (most sensitive)
    radiation_q_factor: float = 0.8        # Quiescent cells (resistant)
    
    # Chemotherapy parameters
    chemo_sensitivity: float = 0.2         # Base sensitivity to chemotherapy
    drug_diffusion: float = 0.5           # Drug diffusion coefficient
    drug_decay: float = 0.1               # Drug decay rate
    drug_threshold: float = 0.1           # Minimum effective concentration
    
    # Cell cycle-specific chemotherapy sensitivity multipliers
    # Based on typical cell cycle specific drug effects
    chemo_g1_factor: float = 1.0          # G1 phase relative sensitivity
    chemo_s_factor: float = 2.0           # S phase (most affected by many drugs)
    chemo_g2_factor: float = 1.2          # G2 phase sensitivity
    chemo_m_factor: float = 1.5           # M phase sensitivity
    chemo_q_factor: float = 0.3           # Quiescent cells (very resistant)
    
    # Immunotherapy parameters
    immune_boost: float = 1.5             # Factor to boost immune response
    checkpoint_inhibition: float = 0.3    # Reduction in immune suppression
    
    def validate(self):
        """Validate parameter values."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")

class TreatmentModule:
    """
    Implements various treatment modalities with cell cycle-specific effects:
    - Radiation therapy
    - Chemotherapy
    - Immunotherapy
    """
    
    def __init__(self, params: Optional[TreatmentParameters] = None):
        self.params = params or TreatmentParameters()
        self.params.validate()
        
        # Treatment state variables
        self.drug_concentration = None
        self.radiation_dose_map = None
        self.cumulative_dose = None
        
    def initialize(self, shape: Tuple[int, ...]):
        """Initialize treatment-related variables."""
        self.drug_concentration = np.zeros(shape)
        self.radiation_dose_map = np.zeros(shape)
        self.cumulative_dose = np.zeros(shape)
        
    def apply_treatment(self,
                       treatment_type: str,
                       cell_populations: Dict[str, np.ndarray],
                       oxygen_concentration: Optional[np.ndarray] = None,
                       immune_density: Optional[np.ndarray] = None,
                       **kwargs) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Apply specified treatment and calculate its effects.
        
        Args:
            treatment_type: Type of treatment ("radiation", "chemo", or "immunotherapy")
            cell_populations: Dictionary of cell populations by phase
            oxygen_concentration: Current oxygen levels (for radiation)
            immune_density: Current immune cell density (for immunotherapy)
            **kwargs: Additional treatment-specific parameters
            
        Returns:
            Tuple of (treatment effects by population, treatment metrics)
        """
        if self.drug_concentration is None:
            self.initialize(next(iter(cell_populations.values())).shape)
            
        if treatment_type == "radiation":
            effects, metrics = self._apply_radiation(
                cell_populations, oxygen_concentration, **kwargs
            )
        elif treatment_type == "chemo":
            effects, metrics = self._apply_chemotherapy(
                cell_populations, **kwargs
            )
        elif treatment_type == "immunotherapy":
            effects, metrics = self._apply_immunotherapy(
                cell_populations, immune_density, **kwargs
            )
        else:
            raise ValueError(f"Unknown treatment type: {treatment_type}")
            
        return effects, metrics
        
    def _apply_radiation(self,
                        cell_populations: Dict[str, np.ndarray],
                        oxygen_concentration: np.ndarray,
                        dose: float = 2.0,  # Gy
                        beam_angles: Optional[List[float]] = None) -> Tuple[Dict[str, np.ndarray], dict]:
        """Apply radiation therapy using linear-quadratic model with cell cycle effects."""
        if beam_angles is None:
            beam_angles = [0.0]  # Default to single anterior beam
            
        # Calculate dose distribution
        dose_map = self._calculate_dose_distribution(dose, beam_angles)
        self.radiation_dose_map = dose_map
        self.cumulative_dose += dose_map
        
        # Apply oxygen enhancement ratio
        effective_dose = dose_map * (
            1.0 + (self.params.oxygen_enhancement - 1.0) * 
            oxygen_concentration
        )
        
        # Linear-quadratic cell survival model with phase-specific sensitivities
        effects = {}
        total_killed = 0
        
        phase_factors = {
            'G1': self.params.radiation_g1_factor,
            'S': self.params.radiation_s_factor,
            'G2': self.params.radiation_g2_factor,
            'M': self.params.radiation_m_factor,
            'Q': self.params.radiation_q_factor
        }
        
        for phase, population in cell_populations.items():
            if phase == 'N':  # Skip necrotic cells
                effects[phase] = np.zeros_like(population)
                continue
                
            sensitivity = self.params.radiation_sensitivity * phase_factors.get(phase, 1.0)
            alpha = self.params.fractionation_alpha * sensitivity
            beta = self.params.fractionation_beta * sensitivity
            
            survival_fraction = np.exp(
                -alpha * effective_dose - 
                beta * effective_dose**2
            )
            
            effect = -(1.0 - survival_fraction) * population
            effects[phase] = effect
            total_killed -= np.sum(effect)
            
        metrics = {
            'mean_dose': float(np.mean(dose_map)),
            'max_dose': float(np.max(dose_map)),
            'cumulative_dose': float(np.mean(self.cumulative_dose)),
            'total_cells_killed': float(total_killed)
        }
        
        return effects, metrics
        
    def _apply_chemotherapy(self,
                           cell_populations: Dict[str, np.ndarray],
                           drug_amount: float = 1.0,
                           duration: float = 1.0) -> Tuple[Dict[str, np.ndarray], dict]:
        """Apply chemotherapy with cell cycle-specific drug effects."""
        # Update drug concentration
        self._update_drug_concentration(drug_amount, duration)
        
        # Calculate phase-specific effects
        effects = {}
        total_killed = 0
        
        phase_factors = {
            'G1': self.params.chemo_g1_factor,
            'S': self.params.chemo_s_factor,
            'G2': self.params.chemo_g2_factor,
            'M': self.params.chemo_m_factor,
            'Q': self.params.chemo_q_factor
        }
        
        above_threshold = self.drug_concentration > self.params.drug_threshold
        
        for phase, population in cell_populations.items():
            if phase == 'N':  # Skip necrotic cells
                effects[phase] = np.zeros_like(population)
                continue
                
            sensitivity = self.params.chemo_sensitivity * phase_factors.get(phase, 1.0)
            drug_effect = sensitivity * self.drug_concentration * above_threshold
            effect = -drug_effect * population
            effects[phase] = effect
            total_killed -= np.sum(effect)
            
        metrics = {
            'mean_drug_conc': float(np.mean(self.drug_concentration)),
            'effective_coverage': float(np.mean(above_threshold)),
            'total_cells_killed': float(total_killed)
        }
        
        return effects, metrics
        
    def _apply_immunotherapy(self,
                           cell_populations: Dict[str, np.ndarray],
                           immune_density: np.ndarray,
                           boost_factor: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], dict]:
        """Apply immunotherapy with enhanced immune response."""
        if boost_factor is None:
            boost_factor = self.params.immune_boost
            
        # Enhanced immune cell activity
        boosted_immune = immune_density * boost_factor
        
        # Reduced tumor immune suppression
        suppression_factor = 1.0 - self.params.checkpoint_inhibition
        
        # Calculate effects on each population
        effects = {}
        total_killed = 0
        
        # Proliferating cells are more susceptible to immune attack
        phase_factors = {
            'G1': 1.0,
            'S': 1.2,
            'G2': 1.2,
            'M': 1.5,
            'Q': 0.7,
            'N': 0.0  # No effect on necrotic cells
        }
        
        for phase, population in cell_populations.items():
            sensitivity = phase_factors.get(phase, 1.0)
            immune_effect = -(boosted_immune * population * 
                            suppression_factor * sensitivity)
            effects[phase] = immune_effect
            total_killed -= np.sum(immune_effect)
            
        metrics = {
            'immune_boost': float(boost_factor),
            'mean_immune_effect': float(-total_killed),
            'total_cells_killed': float(total_killed)
        }
        
        return effects, metrics
        
    def _calculate_dose_distribution(self,
                                   dose: float,
                                   beam_angles: List[float]) -> np.ndarray:
        """Calculate radiation dose distribution from beam angles."""
        shape = self.radiation_dose_map.shape
        center = np.array(shape) / 2
        y, x = np.ogrid[:shape[0], :shape[1]]
        dose_map = np.zeros(shape)
        
        for angle in beam_angles:
            # Rotate coordinates
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            x_rot = cos_a * (x - center[1]) - sin_a * (y - center[0])
            
            # Simple exponential depth dose
            depth = x_rot + max(shape)  # Shift to positive values
            beam_dose = dose * np.exp(-0.1 * depth)  # Example attenuation
            dose_map += beam_dose
            
        # Normalize
        dose_map *= dose / np.max(dose_map)
        return dose_map
        
    def _update_drug_concentration(self,
                                 drug_amount: float,
                                 duration: float):
        """Update drug concentration considering diffusion and decay."""
        # Add new drug
        self.drug_concentration += drug_amount
        
        # Diffusion
        diffusion = self._compute_diffusion(self.drug_concentration)
        self.drug_concentration += duration * (
            self.params.drug_diffusion * diffusion -
            self.params.drug_decay * self.drug_concentration
        )
        
        # Ensure non-negativity
        np.clip(self.drug_concentration, 0, None, out=self.drug_concentration)
        
    def _compute_diffusion(self, field: np.ndarray) -> np.ndarray:
        """Compute diffusion term using finite differences."""
        return (np.roll(field, 1, axis=0) + 
                np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) + 
                np.roll(field, -1, axis=1) - 
                4 * field)
                
    def get_metrics(self) -> dict:
        """Calculate treatment-related metrics."""
        return {
            'cumulative_radiation': float(np.sum(self.cumulative_dose)),
            'mean_drug_concentration': float(np.mean(self.drug_concentration)),
            'max_drug_concentration': float(np.max(self.drug_concentration)),
            'treatment_coverage': float(np.mean(
                self.drug_concentration > self.params.drug_threshold
            ))
        }
        
    def get_treatment_schedule(self, 
                             treatment_type: str,
                             total_time: float) -> List[Dict]:
        """
        Generate a standard treatment schedule with cell cycle considerations.
        
        Args:
            treatment_type: Type of treatment
            total_time: Total treatment duration in days
            
        Returns:
            List of treatment events with timing
        """
        schedule = []
        
        if treatment_type == "radiation":
            # Standard fractionation: 2 Gy per day, 5 days per week
            # Timing optimized for cell cycle effects (early morning delivery)
            fraction_dose = 2.0  # Gy
            days_per_week = 5
            current_day = 0
            
            while current_day < total_time:
                # Skip weekends
                if (current_day % 7) < days_per_week:
                    schedule.append({
                        'time': current_day,
                        'type': 'radiation',
                        'dose': fraction_dose,
                        'beam_angles': [0, 90, 180, 270],  # 4-field box
                        'delivery_hour': 8  # Morning delivery when more cells in G2/M
                    })
                current_day += 1
                
        elif treatment_type == "chemo":
            # Weekly chemotherapy with cell cycle-specific timing
            cycle_length = 7  # days
            current_day = 0
            
            while current_day < total_time:
                # Different drugs might target different phases
                schedule.append({
                    'time': current_day,
                    'type': 'chemo',
                    'drug_amount': 1.0,
                    'duration': 1.0,
                    'target_phase': 'S',  # Example: S-phase specific drug
                    'delivery_hour': 14  # Afternoon delivery for S-phase peak
                })
                current_day += cycle_length
                
        elif treatment_type == "immunotherapy":
            # Immunotherapy every 2 weeks
            # Less dependent on cell cycle, more on immune system dynamics
            cycle_length = 14  # days
            current_day = 0
            
            while current_day < total_time:
                schedule.append({
                    'time': current_day,
                    'type': 'immunotherapy',
                    'boost_factor': self.params.immune_boost,
                    'delivery_hour': 10  # Morning delivery for immune system activity
                })
                current_day += cycle_length
                
        return schedule