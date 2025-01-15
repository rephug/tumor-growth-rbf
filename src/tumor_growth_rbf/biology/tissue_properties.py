"""
tissue_properties.py

Models tissue-specific effects on tumor growth and treatment response.
This module handles the interaction between different tissue types and tumor behavior,
particularly important for brain tumors where white and gray matter exhibit distinct
growth patterns.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import logging
from enum import Enum
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)

class TissueType(Enum):
    """Enumeration of tissue types with their characteristic properties."""
    WHITE_MATTER = "white_matter"
    GRAY_MATTER = "gray_matter"
    CSF = "csf"  # Cerebrospinal fluid
    VESSEL = "vessel"
    NECROTIC = "necrotic"

@dataclass
class TissueParameters:
    """Parameters defining tissue-specific properties affecting tumor growth."""
    
    # Diffusion coefficients (mm²/day)
    # Based on published glioma growth studies
    diffusion_coefficients: Dict[TissueType, float] = None
    
    # Growth rate modifiers
    # Relative rates based on experimental observations
    growth_modifiers: Dict[TissueType, float] = None
    
    # Oxygen perfusion rates
    # Based on tissue vascularity
    oxygen_perfusion: Dict[TissueType, float] = None
    
    # Treatment sensitivity modifiers
    # How different tissues affect treatment efficacy
    radiation_modifiers: Dict[TissueType, float] = None
    drug_penetration: Dict[TissueType, float] = None
    
    def __post_init__(self):
        """Initialize default values if none provided."""
        if self.diffusion_coefficients is None:
            self.diffusion_coefficients = {
                TissueType.WHITE_MATTER: 0.1,    # Higher diffusion in white matter
                TissueType.GRAY_MATTER: 0.01,    # Lower diffusion in gray matter
                TissueType.CSF: 0.5,             # High diffusion in CSF
                TissueType.VESSEL: 0.0,          # No diffusion through vessels
                TissueType.NECROTIC: 0.05        # Moderate diffusion in necrotic tissue
            }
            
        if self.growth_modifiers is None:
            self.growth_modifiers = {
                TissueType.WHITE_MATTER: 1.0,    # Reference growth rate
                TissueType.GRAY_MATTER: 0.7,     # Slower growth in gray matter
                TissueType.CSF: 0.1,             # Limited growth in CSF
                TissueType.VESSEL: 0.0,          # No growth in vessels
                TissueType.NECROTIC: 0.3         # Reduced growth in necrotic regions
            }
            
        if self.oxygen_perfusion is None:
            self.oxygen_perfusion = {
                TissueType.WHITE_MATTER: 1.0,    # Normal perfusion
                TissueType.GRAY_MATTER: 1.2,     # Higher perfusion in gray matter
                TissueType.CSF: 0.5,             # Limited oxygen in CSF
                TissueType.VESSEL: 2.0,          # High oxygen near vessels
                TissueType.NECROTIC: 0.2         # Poor perfusion in necrotic tissue
            }
            
        if self.radiation_modifiers is None:
            self.radiation_modifiers = {
                TissueType.WHITE_MATTER: 1.0,    # Reference sensitivity
                TissueType.GRAY_MATTER: 1.0,     # Similar sensitivity
                TissueType.CSF: 1.2,             # Higher radiation effect in CSF
                TissueType.VESSEL: 0.8,          # Some protection near vessels
                TissueType.NECROTIC: 1.1         # Slightly higher sensitivity
            }
            
        if self.drug_penetration is None:
            self.drug_penetration = {
                TissueType.WHITE_MATTER: 1.0,    # Reference penetration
                TissueType.GRAY_MATTER: 0.8,     # Slightly lower penetration
                TissueType.CSF: 1.5,             # Better drug distribution in CSF
                TissueType.VESSEL: 2.0,          # High concentration near vessels
                TissueType.NECROTIC: 0.4         # Poor drug penetration
            }

    def validate(self):
        """Validate parameter values and tissue type coverage."""
        for param_dict in [self.diffusion_coefficients, self.growth_modifiers,
                          self.oxygen_perfusion, self.radiation_modifiers,
                          self.drug_penetration]:
            if not all(isinstance(v, (int, float)) and v >= 0 for v in param_dict.values()):
                raise ValueError("All tissue parameters must be non-negative numbers")

class TissueModel:
    """
    Handles tissue-specific effects on tumor growth and treatment.
    Supports loading tissue maps from imaging data and computing
    local tissue effects.
    """
    
    def __init__(self, params: Optional[TissueParameters] = None):
        self.params = params or TissueParameters()
        self.params.validate()
        
        # Tissue type map
        self.tissue_map = None
        self.vessel_map = None
        
        # Cached property maps for efficiency
        self._diffusion_map = None
        self._growth_map = None
        self._oxygen_map = None
        
    def initialize_from_image(self,
                            tissue_image: np.ndarray,
                            tissue_labels: Dict[int, TissueType],
                            vessel_image: Optional[np.ndarray] = None):
        """
        Initialize tissue maps from medical imaging data.
        
        Args:
            tissue_image: Labeled image with tissue types
            tissue_labels: Mapping from image values to TissueType
            vessel_image: Optional binary vessel mask
        """
        shape = tissue_image.shape
        self.tissue_map = np.zeros(shape, dtype=object)
        
        # Convert image labels to TissueTypes
        for label, tissue_type in tissue_labels.items():
            self.tissue_map[tissue_image == label] = tissue_type
            
        # Initialize vessel map
        if vessel_image is not None:
            self.vessel_map = vessel_image.astype(bool)
            self.tissue_map[self.vessel_map] = TissueType.VESSEL
        else:
            self.vessel_map = np.zeros(shape, dtype=bool)
            
        # Clear cached maps
        self._clear_cache()
        
    def update_tissue_state(self, necrotic_regions: np.ndarray):
        """
        Update tissue state based on tumor evolution.
        
        Args:
            necrotic_regions: Boolean array marking necrotic areas
        """
        # Update tissue map with new necrotic regions
        self.tissue_map[necrotic_regions & 
                       (self.tissue_map != TissueType.VESSEL)] = TissueType.NECROTIC
        
        # Clear cached maps
        self._clear_cache()
        
    def get_diffusion_coefficient_map(self) -> np.ndarray:
        """Get spatially-varying diffusion coefficients."""
        if self._diffusion_map is None:
            self._diffusion_map = self._create_parameter_map(
                self.params.diffusion_coefficients
            )
        return self._diffusion_map
        
    def get_growth_modifier_map(self) -> np.ndarray:
        """Get growth rate modifiers for each position."""
        if self._growth_map is None:
            self._growth_map = self._create_parameter_map(
                self.params.growth_modifiers
            )
        return self._growth_map
        
    def get_oxygen_perfusion_map(self) -> np.ndarray:
        """Get oxygen perfusion rates for each position."""
        if self._oxygen_map is None:
            self._oxygen_map = self._create_parameter_map(
                self.params.oxygen_perfusion
            )
        return self._oxygen_map
        
    def get_treatment_modifier_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get radiation and drug penetration modifier maps."""
        radiation_map = self._create_parameter_map(
            self.params.radiation_modifiers
        )
        drug_map = self._create_parameter_map(
            self.params.drug_penetration
        )
        return radiation_map, drug_map
        
    def _create_parameter_map(self, parameter_dict: Dict[TissueType, float]) -> np.ndarray:
        """Create spatial map of tissue parameters."""
        param_map = np.zeros_like(self.tissue_map, dtype=float)
        
        for tissue_type in TissueType:
            mask = (self.tissue_map == tissue_type)
            param_map[mask] = parameter_dict[tissue_type]
            
        return param_map
        
    def _clear_cache(self):
        """Clear cached parameter maps."""
        self._diffusion_map = None
        self._growth_map = None
        self._oxygen_map = None
        
    def get_metrics(self) -> Dict:
        """Calculate tissue-related metrics."""
        if self.tissue_map is None:
            return {}
            
        metrics = {}
        total_pixels = np.size(self.tissue_map)
        
        # Calculate tissue type fractions
        for tissue_type in TissueType:
            fraction = np.sum(self.tissue_map == tissue_type) / total_pixels
            metrics[f'{tissue_type.value}_fraction'] = float(fraction)
            
        # Calculate vessel density
        if self.vessel_map is not None:
            metrics['vessel_density'] = float(np.mean(self.vessel_map))
            
            # Calculate mean distance to nearest vessel
            if np.any(self.vessel_map):
                distance_map = ndimage.distance_transform_edt(~self.vessel_map)
                metrics['mean_vessel_distance'] = float(np.mean(distance_map))
                metrics['max_vessel_distance'] = float(np.max(distance_map))
                
        return metrics