"""
immune_response.py

Models the immune system's response to tumor growth.

BIOLOGY BACKGROUND
==================
The immune system can recognize and kill tumor cells, but tumors develop
evasion strategies. The key dynamics are:

1. DETECTION: Tumor cells release "danger signals" (chemokines/cytokines)
2. RECRUITMENT: Immune cells (T-cells, NK cells) are attracted to the tumor
3. KILLING: Immune cells destroy tumor cells on contact
4. EXHAUSTION: Prolonged exposure leads to immune cell dysfunction
5. EVASION: Tumors create immunosuppressive microenvironments

Clinical relevance:
- Immunotherapy (checkpoint inhibitors) works by blocking evasion (#5)
- CAR-T therapy enhances killing (#3)
- Understanding immune infiltration patterns guides treatment planning

MODEL DESIGN:
This module handles LOCAL immune dynamics (recruitment, killing rates).
Spatial effects (immune cell migration via chemokine gradients) are
handled by the tumor model using RBF-FD operators, NOT by np.roll.
This was a critical bug in the original code.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImmuneParameters:
    """
    Parameters for immune response model.

    LEARNING NOTE: These parameters capture the key rates of immune dynamics.
    Real values vary enormously between patients and tumor types — this is
    one of the biggest challenges in computational immunology.
    """
    # Immune cell recruitment
    recruitment_rate: float = 0.1       # Rate immune cells arrive (day⁻¹)
    saturation_constant: float = 0.5    # Michaelis-Menten saturation for recruitment
    activation_threshold: float = 0.2   # Min tumor signal for immune activation

    # Immune cell killing
    killing_rate: float = 0.2          # Rate of tumor cell killing (day⁻¹)

    # Immune cell dynamics
    immune_death_rate: float = 0.1     # Natural turnover of immune cells (day⁻¹)

    # Chemokine dynamics
    chemokine_production: float = 1.0  # Chemokine production by tumor cells
    chemokine_decay: float = 0.1       # Chemokine natural decay rate (day⁻¹)
    chemokine_diffusion: float = 0.5   # Chemokine diffusion rate (mm²/day)

    # Immune cell migration
    chemotaxis_strength: float = 0.5   # Strength of directed migration

    def validate(self):
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")


class ImmuneResponse:
    """
    Models immune system response to tumor growth.

    KEY DESIGN PRINCIPLE: This class computes LOCAL reaction terms.
    It does NOT handle spatial derivatives (diffusion, advection).
    Those are computed by the TumorModel using the RBF-FD framework.

    State variables (arrays with one value per spatial point):
    - immune_density: Concentration of immune cells
    - chemokine_concentration: Concentration of tumor-secreted signals
    """

    def __init__(self, params: Optional[ImmuneParameters] = None):
        self.params = params or ImmuneParameters()
        self.params.validate()

        self.immune_density: Optional[np.ndarray] = None
        self.chemokine_concentration: Optional[np.ndarray] = None

    def initialize(self, shape: Tuple[int, ...]):
        """Initialize immune system state variables."""
        self.immune_density = np.zeros(shape)
        self.chemokine_concentration = np.zeros(shape)

    def update(self,
               dt: float,
               tumor_density: np.ndarray,
               oxygen_concentration: np.ndarray,
               laplacian_chemokine: Optional[np.ndarray] = None,
               chemokine_gradient: Optional[Tuple[np.ndarray, ...]] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update immune response for one time step.

        CRITICAL CHANGE from original code:
        Spatial derivatives (laplacian, gradient) are now PASSED IN
        from the tumor model, computed via RBF-FD. The old code used
        np.roll() which only works on regular grids.

        Args:
            dt: Time step in days
            tumor_density: Current tumor density field
            oxygen_concentration: Current oxygen levels
            laplacian_chemokine: ∇²(chemokine) computed externally via RBF-FD
                If None, chemokine diffusion is skipped.
            chemokine_gradient: (∂c/∂x, ∂c/∂y) computed externally
                If None, directed migration is skipped.

        Returns:
            (immune_effect_on_tumor, updated_immune_density)
        """
        if self.immune_density is None:
            self.initialize(tumor_density.shape)

        # Step 1: Update chemokine concentration
        self._update_chemokines(dt, tumor_density, laplacian_chemokine)

        # Step 2: Update immune cell density
        self._update_immune_cells(
            dt, tumor_density, oxygen_concentration, chemokine_gradient
        )

        # Step 3: Calculate effect on tumor
        immune_effect = self._calculate_immune_effect(tumor_density)

        return immune_effect, self.immune_density

    def _update_chemokines(self,
                           dt: float,
                           tumor_density: np.ndarray,
                           laplacian_chemokine: Optional[np.ndarray]):
        """
        Update chemokine concentration.

        BIOLOGY: Tumor cells secrete chemokines (signaling molecules)
        that diffuse through tissue and attract immune cells.
        The concentration follows a reaction-diffusion equation:
            ∂c/∂t = D∇²c + production - decay
        """
        # Production: proportional to tumor density
        production = self.params.chemokine_production * tumor_density

        # Diffusion (computed externally via RBF-FD)
        diffusion = 0.0
        if laplacian_chemokine is not None:
            diffusion = self.params.chemokine_diffusion * laplacian_chemokine

        # Natural decay
        decay = self.params.chemokine_decay * self.chemokine_concentration

        # Forward Euler update
        self.chemokine_concentration += dt * (production + diffusion - decay)
        np.clip(self.chemokine_concentration, 0, None,
                out=self.chemokine_concentration)

    def _update_immune_cells(self,
                             dt: float,
                             tumor_density: np.ndarray,
                             oxygen_concentration: np.ndarray,
                             chemokine_gradient: Optional[Tuple]):
        """
        Update immune cell density.

        BIOLOGY: Immune cell dynamics include:
        1. Recruitment: Drawn to tumor by chemokine signals
           (Michaelis-Menten kinetics — saturates at high signal)
        2. Migration: Directed movement up chemokine gradients (chemotaxis)
        3. Death: Natural turnover, modulated by oxygen

        MATH: Michaelis-Menten recruitment:
            R = R_max * [chemokine] / (K + [chemokine])
        This saturates: even very high chemokine levels only recruit
        so many immune cells. K is the half-maximal concentration.
        """
        # Recruitment (Michaelis-Menten kinetics)
        recruitment = (self.params.recruitment_rate *
                       self.chemokine_concentration /
                       (self.params.saturation_constant +
                        self.chemokine_concentration + 1e-10))

        # Only activate above threshold tumor density
        recruitment *= (tumor_density > self.params.activation_threshold)

        # Migration via chemotaxis (if gradient available)
        migration = np.zeros_like(self.immune_density)
        if chemokine_gradient is not None:
            # Chemotaxis: immune cells move up the chemokine gradient
            # Works for any dimensionality (2D or 3D)
            grad_magnitude = np.sqrt(
                sum(g ** 2 for g in chemokine_gradient) + 1e-10
            )
            migration = self.params.chemotaxis_strength * grad_magnitude

        # Death rate (increases in hypoxic regions — immune cells also need O₂)
        death_rate = self.params.immune_death_rate * (
            1.0 + 0.5 * (1.0 - oxygen_concentration)
        )

        # Update
        self.immune_density += dt * (
            recruitment + migration - death_rate * self.immune_density
        )
        np.clip(self.immune_density, 0, None, out=self.immune_density)

    def _calculate_immune_effect(self, tumor_density: np.ndarray) -> np.ndarray:
        """
        Calculate immune system's effect on tumor cells.

        BIOLOGY: Immune killing follows Michaelis-Menten kinetics:
            kill_rate = k * I * T / (K + T)

        This saturates at high tumor density — immune cells can only
        kill so many tumor cells per unit time. This is why immune
        response alone often can't eliminate large tumors.
        """
        killing = (self.params.killing_rate *
                   self.immune_density *
                   tumor_density /
                   (self.params.saturation_constant + tumor_density + 1e-10))

        return -killing  # Negative = reduces tumor

    def get_metrics(self) -> Dict:
        """Calculate immune response metrics."""
        if self.immune_density is None:
            return {
                'total_immune_cells': 0.0,
                'max_immune_density': 0.0,
                'mean_chemokine_conc': 0.0,
                'immune_coverage': 0.0
            }

        return {
            'total_immune_cells': float(np.sum(self.immune_density)),
            'max_immune_density': float(np.max(self.immune_density)),
            'mean_chemokine_conc': float(
                np.mean(self.chemokine_concentration)),
            'immune_coverage': float(
                np.mean(self.immune_density > 0.1))
        }
