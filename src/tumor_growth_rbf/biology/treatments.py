"""
treatments.py

Models treatment modalities: radiation, chemotherapy, and immunotherapy.

BIOLOGY BACKGROUND
==================

RADIATION THERAPY:
Uses ionizing radiation to damage tumor cell DNA.
Modeled via the Linear-Quadratic (LQ) model:
    Survival = exp(-α*D - β*D²)
where D = dose, α = linear damage, β = quadratic damage.

Key concepts:
- α/β ratio: Determines fractionation sensitivity (~10 Gy for tumors)
- Oxygen Enhancement Ratio (OER): Hypoxic cells need 2-3x more dose
- Cell cycle sensitivity: M/G2 most sensitive, S most resistant

CHEMOTHERAPY:
Cytotoxic drugs that kill dividing cells.
- Many drugs are S-phase specific (target DNA replication)
- Drug must reach therapeutic concentration at tumor site
- Modeled as drug transport + local cell kill

IMMUNOTHERAPY:
Enhances the immune system's anti-tumor response.
- Checkpoint inhibitors (anti-PD1, anti-CTLA4)
- Modeled as boost to immune recruitment and killing
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreatmentParameters:
    """Parameters for treatment modalities with cell cycle specificity."""

    # --- Radiation therapy ---
    radiation_sensitivity: float = 0.3      # Base sensitivity
    oxygen_enhancement: float = 2.0         # OER
    fractionation_alpha: float = 0.15       # α in LQ model (Gy⁻¹)
    fractionation_beta: float = 0.05        # β in LQ model (Gy⁻²)

    # Phase-specific radiation sensitivity multipliers
    # Based on experimental radiosensitivity measurements
    radiation_g1_factor: float = 1.0   # G1: moderate sensitivity
    radiation_s_factor: float = 0.5    # S: most resistant (active DNA repair)
    radiation_g2_factor: float = 1.5   # G2: sensitive (4N DNA content)
    radiation_m_factor: float = 2.0    # M: most sensitive (condensed chromatin)
    radiation_q_factor: float = 0.8    # Q: resistant (not dividing)

    # --- Chemotherapy ---
    chemo_sensitivity: float = 0.2     # Base drug sensitivity
    drug_decay: float = 0.1            # Drug metabolism/clearance rate (day⁻¹)
    drug_threshold: float = 0.1        # Min effective concentration

    # Phase-specific chemo sensitivity
    chemo_g1_factor: float = 1.0   # G1: moderate
    chemo_s_factor: float = 2.0    # S: most affected (many drugs target DNA synth)
    chemo_g2_factor: float = 1.2   # G2: somewhat sensitive
    chemo_m_factor: float = 1.5    # M: sensitive (e.g., taxanes target mitosis)
    chemo_q_factor: float = 0.3    # Q: very resistant (not dividing)

    # --- Immunotherapy ---
    immune_boost: float = 1.5          # Immune activity boost factor
    checkpoint_inhibition: float = 0.3 # Reduction in immune suppression

    def validate(self):
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")


class TreatmentModule:
    """
    Implements treatment modalities with cell cycle-specific effects.

    DESIGN: Like the immune module, this handles LOCAL treatment effects.
    Drug transport (diffusion) is handled by the tumor model with RBF-FD.

    State variables:
    - drug_concentration: Current drug levels at each point
    - cumulative_dose: Total radiation dose received at each point
    """

    def __init__(self, params: Optional[TreatmentParameters] = None):
        self.params = params or TreatmentParameters()
        self.params.validate()

        self.drug_concentration: Optional[np.ndarray] = None
        self.cumulative_dose: Optional[np.ndarray] = None

    def initialize(self, shape: Tuple[int, ...]):
        """Initialize treatment state variables."""
        self.drug_concentration = np.zeros(shape)
        self.cumulative_dose = np.zeros(shape)

    def apply_treatment(self,
                        treatment_type: str,
                        cell_populations: Dict[str, np.ndarray],
                        oxygen_concentration: Optional[np.ndarray] = None,
                        immune_density: Optional[np.ndarray] = None,
                        **kwargs
                        ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Apply a treatment and return its effects.

        Args:
            treatment_type: "radiation", "chemo", or "immunotherapy"
            cell_populations: Dict of phase → density arrays
            oxygen_concentration: Current O₂ levels (needed for radiation)
            immune_density: Current immune cell density (for immunotherapy)
            **kwargs: Treatment-specific parameters (dose, drug_amount, etc.)

        Returns:
            (effects_by_phase, metrics) — effects are CHANGES to apply
        """
        if self.drug_concentration is None:
            shape = next(iter(cell_populations.values())).shape
            self.initialize(shape)

        if treatment_type == "radiation":
            return self._apply_radiation(
                cell_populations, oxygen_concentration, **kwargs
            )
        elif treatment_type in ("chemo", "chemotherapy"):
            return self._apply_chemotherapy(cell_populations, **kwargs)
        elif treatment_type == "immunotherapy":
            return self._apply_immunotherapy(
                cell_populations, immune_density, **kwargs
            )
        else:
            raise ValueError(f"Unknown treatment type: {treatment_type}")

    def _apply_radiation(self,
                         cell_populations: Dict[str, np.ndarray],
                         oxygen_concentration: np.ndarray,
                         dose: float = 2.0,
                         radiation_modifier: Optional[np.ndarray] = None,
                         **kwargs
                         ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Apply radiation therapy using the Linear-Quadratic model.

        THE LQ MODEL — most important equation in radiation biology:
            Survival_fraction = exp(-α*D - β*D²)

        where:
            D = delivered dose (Gy)
            α = probability of lethal single-track damage (Gy⁻¹)
            β = probability of lethal double-track damage (Gy⁻²)

        For a 2 Gy fraction with typical tumor α/β = 10 Gy:
            Survival ≈ exp(-0.15*2 - 0.05*4) = exp(-0.50) ≈ 0.61
        So ~39% of cells are killed per fraction.

        OXYGEN EFFECT: Hypoxic cells are 2-3x more resistant.
        We model this with the Oxygen Enhancement Ratio (OER):
            effective_dose = dose * (1 + (OER-1) * O₂)
        """
        # Track cumulative dose
        self.cumulative_dose += dose

        # Compute effective dose with oxygen enhancement
        if oxygen_concentration is not None:
            effective_dose = dose * (
                1.0 + (self.params.oxygen_enhancement - 1.0) *
                oxygen_concentration
            )
        else:
            effective_dose = dose * np.ones_like(self.cumulative_dose)

        # Apply tissue-specific modifier if available
        if radiation_modifier is not None:
            effective_dose *= radiation_modifier

        # Phase-specific sensitivity factors
        phase_factors = {
            'G1': self.params.radiation_g1_factor,
            'S': self.params.radiation_s_factor,
            'G2': self.params.radiation_g2_factor,
            'M': self.params.radiation_m_factor,
            'Q': self.params.radiation_q_factor,
        }

        effects = {}
        total_killed = 0.0

        for phase, population in cell_populations.items():
            if phase == 'N':  # Can't kill dead cells
                effects[phase] = np.zeros_like(population)
                continue

            # Phase-specific LQ parameters
            sensitivity = phase_factors.get(phase, 1.0)
            alpha = self.params.fractionation_alpha * sensitivity
            beta = self.params.fractionation_beta * sensitivity

            # LQ survival fraction
            survival = np.exp(-alpha * effective_dose -
                              beta * effective_dose ** 2)

            # Effect = killed cells (negative change)
            effect = -(1.0 - survival) * population
            effects[phase] = effect
            total_killed -= float(np.sum(effect))

        metrics = {
            'dose_delivered': float(dose),
            'cumulative_dose': float(np.mean(self.cumulative_dose)),
            'total_cells_killed': total_killed,
            'mean_survival_fraction': float(np.mean(
                np.exp(-self.params.fractionation_alpha * effective_dose)
            )),
        }

        return effects, metrics

    def _apply_chemotherapy(self,
                            cell_populations: Dict[str, np.ndarray],
                            drug_amount: float = 1.0,
                            duration: float = 1.0,
                            drug_modifier: Optional[np.ndarray] = None,
                            **kwargs
                            ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Apply chemotherapy with cell cycle-specific effects.

        MODEL: Drug is delivered, then kills cells proportionally to
        concentration and cell-phase sensitivity:
            kill = sensitivity * [drug] * population

        Drug decays over time (metabolism/clearance).
        """
        # Add new drug dose
        self.drug_concentration += drug_amount

        # Apply tissue modifier
        if drug_modifier is not None:
            self.drug_concentration *= drug_modifier

        # Phase-specific sensitivity
        phase_factors = {
            'G1': self.params.chemo_g1_factor,
            'S': self.params.chemo_s_factor,
            'G2': self.params.chemo_g2_factor,
            'M': self.params.chemo_m_factor,
            'Q': self.params.chemo_q_factor,
        }

        above_threshold = self.drug_concentration > self.params.drug_threshold

        effects = {}
        total_killed = 0.0

        for phase, population in cell_populations.items():
            if phase == 'N':
                effects[phase] = np.zeros_like(population)
                continue

            sensitivity = (self.params.chemo_sensitivity *
                           phase_factors.get(phase, 1.0))
            drug_effect = sensitivity * self.drug_concentration * above_threshold

            effect = -drug_effect * population
            effects[phase] = effect
            total_killed -= float(np.sum(effect))

        # Drug decay after treatment window
        self.drug_concentration *= np.exp(-self.params.drug_decay * duration)
        np.clip(self.drug_concentration, 0, None,
                out=self.drug_concentration)

        metrics = {
            'mean_drug_conc': float(np.mean(self.drug_concentration)),
            'effective_coverage': float(np.mean(above_threshold)),
            'total_cells_killed': total_killed,
        }

        return effects, metrics

    def _apply_immunotherapy(self,
                             cell_populations: Dict[str, np.ndarray],
                             immune_density: Optional[np.ndarray],
                             boost_factor: Optional[float] = None,
                             **kwargs
                             ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Apply immunotherapy (checkpoint inhibitor model).

        BIOLOGY: Checkpoint inhibitors block PD-1/PD-L1 or CTLA-4,
        which tumors use to suppress immune attack. This:
        1. Boosts immune cell activity
        2. Reduces tumor immune suppression

        Unlike radiation/chemo, immunotherapy is less cell-cycle dependent
        but more dependent on the existing immune infiltration.
        """
        if boost_factor is None:
            boost_factor = self.params.immune_boost

        if immune_density is None:
            immune_density = np.zeros_like(
                next(iter(cell_populations.values()))
            )

        boosted_immune = immune_density * boost_factor
        suppression_factor = 1.0 - self.params.checkpoint_inhibition

        # Immune killing varies slightly by phase (proliferating cells
        # present more antigens, are more visible to immune system)
        phase_factors = {
            'G1': 1.0,
            'S': 1.2,
            'G2': 1.2,
            'M': 1.5,  # Dividing cells most visible
            'Q': 0.7,  # Quiescent less visible
            'N': 0.0,  # No effect on dead cells
        }

        effects = {}
        total_killed = 0.0

        for phase, population in cell_populations.items():
            sensitivity = phase_factors.get(phase, 1.0)
            effect = -(boosted_immune * population *
                       suppression_factor * sensitivity)
            effects[phase] = effect
            total_killed -= float(np.sum(effect))

        metrics = {
            'immune_boost': float(boost_factor),
            'total_cells_killed': total_killed,
        }

        return effects, metrics

    def get_metrics(self) -> Dict:
        """Calculate treatment-related metrics."""
        if self.drug_concentration is None:
            return {
                'cumulative_radiation': 0.0,
                'mean_drug_concentration': 0.0,
                'max_drug_concentration': 0.0,
                'treatment_coverage': 0.0,
            }

        return {
            'cumulative_radiation': float(np.sum(self.cumulative_dose)),
            'mean_drug_concentration': float(
                np.mean(self.drug_concentration)),
            'max_drug_concentration': float(
                np.max(self.drug_concentration)),
            'treatment_coverage': float(np.mean(
                self.drug_concentration > self.params.drug_threshold
            )),
        }
