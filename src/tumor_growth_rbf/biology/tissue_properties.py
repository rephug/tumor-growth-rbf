"""
tissue_properties.py

Models tissue-specific effects on tumor growth and treatment.

BIOLOGY BACKGROUND
==================
Brain tumors (gliomas) grow differently depending on tissue type:

WHITE MATTER: Myelinated axon bundles
- Tumor cells migrate ALONG fiber tracts → faster diffusion
- Good vascularization → adequate oxygen
- This is why gliomas often spread along white matter tracts

GRAY MATTER: Neuronal cell bodies
- More compact structure → slower tumor diffusion
- Higher metabolic rate → more blood supply
- Tumor growth is slower but more compact

CSF: Cerebrospinal fluid
- Tumor cells can spread through CSF spaces
- High diffusion but poor growth support
- Important for meningeal metastasis

NECROTIC TISSUE: Dead tissue in tumor core
- Poor drug penetration (no blood supply)
- Creates acidic microenvironment
- Barrier to treatment delivery

This module maps tissue types to physical parameters that the
PDE solver uses for spatially-varying coefficients.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TissueType(Enum):
    """Tissue types relevant for brain tumor modeling."""
    WHITE_MATTER = "white_matter"
    GRAY_MATTER = "gray_matter"
    CSF = "csf"
    VESSEL = "vessel"
    NECROTIC = "necrotic"


@dataclass
class TissueParameters:
    """
    Tissue-specific parameters affecting tumor growth and treatment.

    Each parameter is a dictionary mapping TissueType → value.
    Values are relative to white matter (reference tissue).
    """
    diffusion_coefficients: Optional[Dict[TissueType, float]] = None
    growth_modifiers: Optional[Dict[TissueType, float]] = None
    oxygen_perfusion: Optional[Dict[TissueType, float]] = None
    radiation_modifiers: Optional[Dict[TissueType, float]] = None
    drug_penetration: Optional[Dict[TissueType, float]] = None

    def __post_init__(self):
        """Set biologically-motivated default values."""
        if self.diffusion_coefficients is None:
            self.diffusion_coefficients = {
                TissueType.WHITE_MATTER: 0.1,   # mm²/day — reference
                TissueType.GRAY_MATTER: 0.01,   # 10x slower in gray matter
                TissueType.CSF: 0.5,            # Fast diffusion in fluid
                TissueType.VESSEL: 0.0,         # No tumor diffusion through vessels
                TissueType.NECROTIC: 0.05,      # Moderate in necrotic tissue
            }

        if self.growth_modifiers is None:
            self.growth_modifiers = {
                TissueType.WHITE_MATTER: 1.0,   # Reference growth rate
                TissueType.GRAY_MATTER: 0.7,    # Slower in gray matter
                TissueType.CSF: 0.1,            # Minimal growth in CSF
                TissueType.VESSEL: 0.0,         # No growth in vessels
                TissueType.NECROTIC: 0.3,       # Reduced in necrotic regions
            }

        if self.oxygen_perfusion is None:
            self.oxygen_perfusion = {
                TissueType.WHITE_MATTER: 1.0,
                TissueType.GRAY_MATTER: 1.2,    # Higher blood supply
                TissueType.CSF: 0.5,
                TissueType.VESSEL: 2.0,          # Direct blood supply
                TissueType.NECROTIC: 0.2,        # Poor perfusion
            }

        if self.radiation_modifiers is None:
            self.radiation_modifiers = {
                TissueType.WHITE_MATTER: 1.0,
                TissueType.GRAY_MATTER: 1.0,
                TissueType.CSF: 1.2,
                TissueType.VESSEL: 0.8,
                TissueType.NECROTIC: 1.1,
            }

        if self.drug_penetration is None:
            self.drug_penetration = {
                TissueType.WHITE_MATTER: 1.0,
                TissueType.GRAY_MATTER: 0.8,
                TissueType.CSF: 1.5,
                TissueType.VESSEL: 2.0,
                TissueType.NECROTIC: 0.4,        # Poor penetration
            }

    def validate(self):
        for param_dict in [self.diffusion_coefficients, self.growth_modifiers,
                           self.oxygen_perfusion, self.radiation_modifiers,
                           self.drug_penetration]:
            for v in param_dict.values():
                if not isinstance(v, (int, float)) or v < 0:
                    raise ValueError("All tissue parameters must be non-negative")


class TissueModel:
    """
    Maps tissue types to spatially-varying parameters.

    Workflow:
    1. Load tissue map from medical imaging (or create synthetic)
    2. Convert tissue labels → TissueType enum values
    3. Generate parameter maps (diffusion, growth, etc.)
    4. These maps are used by TumorModel as coefficients in PDEs
    """

    def __init__(self, params: Optional[TissueParameters] = None):
        self.params = params or TissueParameters()
        self.params.validate()

        self.tissue_map: Optional[np.ndarray] = None
        self.vessel_map: Optional[np.ndarray] = None

        # Cached maps (cleared when tissue state changes)
        self._diffusion_map = None
        self._growth_map = None
        self._oxygen_map = None

    def initialize_from_image(self,
                              tissue_image: np.ndarray,
                              tissue_labels: Dict[int, TissueType],
                              vessel_image: Optional[np.ndarray] = None):
        """
        Initialize from medical imaging data.

        Args:
            tissue_image: Integer-labeled image (e.g., from segmentation)
            tissue_labels: Mapping from integer labels to TissueType
            vessel_image: Optional binary vessel mask
        """
        shape = tissue_image.shape
        self.tissue_map = np.empty(shape, dtype=object)

        for label, tissue_type in tissue_labels.items():
            self.tissue_map[tissue_image == label] = tissue_type

        if vessel_image is not None:
            self.vessel_map = vessel_image.astype(bool)
            self.tissue_map[self.vessel_map] = TissueType.VESSEL
        else:
            self.vessel_map = np.zeros(shape, dtype=bool)

        self._clear_cache()

    def initialize_uniform(self,
                           n_points: int,
                           tissue_type: TissueType = TissueType.WHITE_MATTER):
        """
        Initialize uniform tissue (no imaging data).
        Useful for simple simulations or testing.
        """
        self.tissue_map = np.full(n_points, tissue_type, dtype=object)
        self.vessel_map = np.zeros(n_points, dtype=bool)
        self._clear_cache()

    def update_tissue_state(self, necrotic_regions: np.ndarray):
        """Mark regions as necrotic (called by tumor model)."""
        mask = necrotic_regions & (self.tissue_map != TissueType.VESSEL)
        self.tissue_map[mask] = TissueType.NECROTIC
        self._clear_cache()

    def get_diffusion_coefficient_map(self) -> np.ndarray:
        if self._diffusion_map is None:
            self._diffusion_map = self._create_parameter_map(
                self.params.diffusion_coefficients
            )
        return self._diffusion_map

    def get_growth_modifier_map(self) -> np.ndarray:
        if self._growth_map is None:
            self._growth_map = self._create_parameter_map(
                self.params.growth_modifiers
            )
        return self._growth_map

    def get_oxygen_perfusion_map(self) -> np.ndarray:
        if self._oxygen_map is None:
            self._oxygen_map = self._create_parameter_map(
                self.params.oxygen_perfusion
            )
        return self._oxygen_map

    def get_treatment_modifier_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        radiation = self._create_parameter_map(self.params.radiation_modifiers)
        drug = self._create_parameter_map(self.params.drug_penetration)
        return radiation, drug

    def _create_parameter_map(self, param_dict: Dict) -> np.ndarray:
        """Convert tissue map + parameter dict → spatial parameter array."""
        param_map = np.ones_like(self.tissue_map, dtype=float)
        for tissue_type in TissueType:
            mask = (self.tissue_map == tissue_type)
            if np.any(mask):
                param_map[mask] = param_dict[tissue_type]
        return param_map

    def _clear_cache(self):
        self._diffusion_map = None
        self._growth_map = None
        self._oxygen_map = None

    def get_metrics(self) -> Dict:
        if self.tissue_map is None:
            return {}
        total = self.tissue_map.size
        metrics = {}
        for tt in TissueType:
            metrics[f'{tt.value}_fraction'] = float(
                np.sum(self.tissue_map == tt) / total
            )
        if self.vessel_map is not None:
            metrics['vessel_density'] = float(np.mean(self.vessel_map))
        return metrics
