#!/usr/bin/env python3
"""
treatments.py

Models various treatment modalities for tumor therapy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class TreatmentParameters:
    """Parameters for different treatment modalities."""
    # Radiation therapy parameters
    radiation_sensitivity: float = 0.3      # Base sensitivity to radiation
    oxygen_enhancement: float = 2.0         # Oxygen enhancement ratio
    fractionation_alpha: float = 0.15      # Linear term in LQ model
    fractionation_beta: float = 0.05       # Quadratic term in LQ model
    
    # Chemotherapy parameters
    chemo_sensitivity: float = 0.2         # Base sensitivity to chemotherapy
    drug_diffusion: float = 0.5           # Drug diffusion coefficient
    drug_decay: float = 0.1               # Drug decay rate
    drug_threshold: float = 0.1           # Minimum effective concentration
    
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
    Implements various treatment modalities:
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
        
    def initialize(self, shape: Tuple[int, int]):
        """Initialize treatment-related variables."""
        self.drug_concentration = np.zeros(shape)
        self.radiation_dose_map = np.zeros(shape)
        self.cumulative_dose = np.zeros(shape)
        
    def apply_treatment(self,
                       treatment_type: str,
                       tumor_density: np.ndarray,
                       oxygen_concentration: Optional[np.ndarray] = None,
                       immune_density: Optional[np.ndarray] = None,
                       **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Apply specified treatment and calculate its effects.
        
        Args:
            treatment_type: Type of treatment ("radiation", "chemo", or "immunotherapy")
            tumor_density: Current tumor density
            oxygen_concentration: Current oxygen levels (for radiation)
            immune_density: Current immune cell density (for immunotherapy)
            **kwargs: Additional treatment-specific parameters
            
        Returns:
            Tuple of (treatment effect on tumor, treatment metrics)
        """
        if self.drug_concentration is None:
            self.initialize(tumor_density.shape)
            
        if treatment_type == "radiation":
            effect, metrics = self._apply_radiation(
                tumor_density, oxygen_concentration, **kwargs
            )
        elif treatment_type == "chemo":
            effect, metrics = self._apply_chemotherapy(
                tumor_density, **kwargs
            )
        elif treatment_type == "immunotherapy":
            effect, metrics = self._apply_immunotherapy(
                tumor_density, immune_density, **kwargs
            )
        else:
            raise ValueError(f"Unknown treatment type: {treatment_type}")
            
        return effect, metrics
        
    def _apply_radiation(self,
                        tumor_density: np.ndarray,
                        oxygen_concentration: np.ndarray,
                        dose: float = 2.0,  # Gy
                        beam_angles: Optional[List[float]] = None) -> Tuple[np.ndarray, dict]:
        """Apply radiation therapy using linear-quadratic model."""
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
        
        # Linear-quadratic cell survival model
        alpha = self.params.fractionation_alpha
        beta = self.params.fractionation_beta
        
        survival_fraction = np.exp(
            -alpha * effective_dose - 
            beta * effective_dose**2
        )
        
        effect = -(1.0 - survival_fraction) * tumor_density
        
        metrics = {
            'mean_dose': float(np.mean(dose_map)),
            'max_dose': float(np.max(dose_map)),
            'cumulative_dose': float(np.mean(self.cumulative_dose))
        }
        
        return effect, metrics
        
    def _apply_chemotherapy(self,
                           tumor_density: np.ndarray,
                           drug_amount: float = 1.0,
                           duration: float = 1.0) -> Tuple[np.ndarray, dict]:
        """Apply chemotherapy with drug diffusion."""
        # Update drug concentration
        self._update_drug_concentration(drug_amount, duration)
        
        # Calculate drug effect
        above_threshold = self.drug_concentration > self.params.drug_threshold
        drug_effect = (self.params.chemo_sensitivity * 
                      self.drug_concentration * 
                      above_threshold)
        
        effect = -drug_effect * tumor_density
        
        metrics = {
            'mean_drug_conc': float(np.mean(self.drug_concentration)),
            'effective_coverage': float(np.mean(above_threshold))
        }
        
        return effect, metrics
        
    def _apply_immunotherapy(self,
                           tumor_density: np.ndarray,
                           immune_density: np.ndarray,
                           boost_factor: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """Apply immunotherapy to enhance immune response."""
        if boost_factor is None:
            boost_factor = self.params.immune_boost
            
        # Enhanced immune cell activity
        boosted_immune = immune_density * boost_factor
        
        # Reduced tumor immune suppression
        suppression_factor = 1.0 - self.params.checkpoint_inhibition
        
        # Calculate enhanced immune effect
        immune_effect = (boosted_immune * tumor_density * 
                        suppression_factor)
        
        effect = -immune_effect
        
        metrics = {
            'immune_boost': float(boost_factor),
            'mean_immune_effect': float(np.mean(immune_effect))
        }
        
        return effect, metrics
        
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
                
    def get_treatment_schedule(self, 
                             treatment_type: str,
                             total_time: float) -> List[Dict]:
        """
        Generate a standard treatment schedule.
        
        Args:
            treatment_type: Type of treatment
            total_time: Total treatment duration in days
            
        Returns:
            List of treatment events with timing
        """
        schedule = []
        
        if treatment_type == "radiation":
            # Standard fractionation: 2 Gy per day, 5 days per week
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
                        'beam_angles': [0, 90, 180, 270]  # 4-field box
                    })
                current_day += 1
                
        elif treatment_type == "chemo":
            # Example: Weekly chemotherapy
            cycle_length = 7  # days
            current_day = 0
            
            while current_day < total_time:
                schedule.append({
                    'time': current_day,
                    'type': 'chemo',
                    'drug_amount': 1.0,
                    'duration': 1.0
                })
                current_day += cycle_length
                
        elif treatment_type == "immunotherapy":
            # Example: Immunotherapy every 2 weeks
            cycle_length = 14  # days
            current_day = 0
            
            while current_day < total_time:
                schedule.append({
                    'time': current_day,
                    'type': 'immunotherapy',
                    'boost_factor': self.params.immune_boost
                })
                current_day += cycle_length
                
        return schedule
        
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