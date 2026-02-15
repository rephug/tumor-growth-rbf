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
- Oxygen Enhancement Ratio (OER): Hypoxic cells are 2-3x more radioresistant.
  Modeled via Alper-Howard-Flanders: OER = (m*K + pO2) / (K + pO2)
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


def oxygen_enhancement_ratio(pO2, m=3.0, K=3.0):
    """
    Alper-Howard-Flanders OER model.

    Computes the Oxygen Enhancement Ratio as a function of partial
    oxygen pressure. The OER describes how much more resistant
    hypoxic cells are to radiation compared to well-oxygenated cells.

    Model:  OER(pO2) = (m * K + pO2) / (K + pO2)

    Behavior:
        - pO2 -> 0:   OER -> m   (maximum radioresistance)
        - pO2 -> inf:  OER -> 1.0 (fully oxygenated, full sensitivity)
        - pO2 = K:     OER = (m + 1) / 2  (half-way between m and 1)

    Args:
        pO2: Partial oxygen pressure in mmHg. Can be scalar or array.
        m: Maximum OER at complete anoxia (default: 3.0).
        K: Half-effect oxygen tension in mmHg (default: 3.0).

    Returns:
        OER values, same shape as pO2. Always >= 1.0.

    Reference:
        Carlson DJ et al. Phys Med Biol 49:4477-4491, 2004.
    """
    return (m * K + pO2) / (K + pO2)


@dataclass
class TreatmentParameters:
    """Parameters for treatment modalities with cell cycle specificity."""

    # --- Radiation therapy ---
    radiation_sensitivity: float = 0.3      # Base sensitivity
    oer_max: float = 3.0                    # Maximum OER at anoxia (Alper-Howard-Flanders m)
    oer_half_effect: float = 3.0            # Half-effect pO₂ in mmHg (Alper-Howard-Flanders K)
    fractionation_alpha: float = 0.035      # α in LQ model (Gy⁻¹), GBM range: 0.01-0.10
    fractionation_beta: float = 0.003       # β in LQ model (Gy⁻²), gives α/β ≈ 11.7 Gy

    # Phase-specific radiation sensitivity multipliers
    # Based on experimental radiosensitivity measurements
    radiation_g1_factor: float = 1.0   # G1: moderate sensitivity
    radiation_s_factor: float = 0.5    # S: most resistant (active DNA repair)
    radiation_g2_factor: float = 1.5   # G2: sensitive (4N DNA content)
    radiation_m_factor: float = 2.0    # M: most sensitive (condensed chromatin)
    radiation_q_factor: float = 0.4    # Q: resistant (not dividing, reduced α by 0.3-0.5×)

    # Treatment-resistant subpopulation
    # Represents cells with inherent resistance (e.g., glioma stem cells,
    # cells with enhanced DNA repair). These survive full treatment courses
    # and seed regrowth. Published estimates: 5-15% for GBM.
    resistant_fraction: float = 0.10          # Fraction of cells inherently resistant
    resistant_radiation_factor: float = 0.05  # Residual radiation sensitivity of resistant cells
    resistant_chemo_factor: float = 0.10      # Residual chemo sensitivity of resistant cells

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

        For a 2 Gy fraction with GBM parameters (α=0.035, α/β≈11.7 Gy):
            At normoxic pO₂=40 mmHg, OER≈1.14:
            α_eff = 0.035/1.14 = 0.031, β_eff = 0.003/1.30 = 0.0023
            Survival ≈ exp(-0.031*2 - 0.0023*4) = exp(-0.071) ≈ 0.93
        At hypoxic pO₂=1 mmHg, OER≈2.5:
            α_eff = 0.035/2.5 = 0.014, β_eff = 0.003/6.25 = 0.00048
            Survival ≈ exp(-0.014*2 - 0.00048*4) = exp(-0.030) ≈ 0.97

        TREATMENT-RESISTANT FRACTION: A subpopulation (default 10%) of
        cells with inherent resistance (e.g., glioma stem cells, enhanced
        DNA repair). These cells have heavily attenuated LQ sensitivity
        and survive full treatment courses to seed regrowth.

        OXYGEN EFFECT (Alper-Howard-Flanders model):
        Hypoxic cells are up to 3x more radioresistant due to reduced
        free-radical fixation in the absence of oxygen. OER modifies
        the LQ parameters, not the delivered dose:
            OER = (m*K + pO₂) / (K + pO₂)
            α_eff = α / OER,  β_eff = β / OER²
        where pO₂ = oxygen * 40 mmHg (normoxic mapping).
        Ref: Carlson DJ et al. Phys Med Biol 49:4477, 2004.
        """
        # Track cumulative dose
        self.cumulative_dose += dose

        # Compute physical dose at each point (tissue-specific modifier)
        physical_dose = dose * np.ones_like(self.cumulative_dose)
        if radiation_modifier is not None:
            physical_dose *= radiation_modifier

        # Compute OER from local oxygen concentration
        # Map normalized oxygen [0,1] to pO₂ [0,40] mmHg
        # (oxygen=1.0 corresponds to ~40 mmHg normoxic tissue pO₂)
        if oxygen_concentration is not None:
            pO2 = oxygen_concentration * 40.0
            oer = oxygen_enhancement_ratio(
                pO2,
                m=self.params.oer_max,
                K=self.params.oer_half_effect,
            )
        else:
            oer = np.ones_like(self.cumulative_dose)

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

            # Apply OER: reduces radiosensitivity under hypoxia
            # OER is a physical (oxygen chemistry) effect that applies
            # equally to all cells regardless of intrinsic resistance
            alpha_eff = alpha / oer
            beta_eff = beta / (oer ** 2)

            # LQ survival for treatable cells
            survival = np.exp(-alpha_eff * physical_dose -
                              beta_eff * physical_dose ** 2)

            # LQ survival for resistant cells (attenuated α/β, also OER-modified)
            alpha_r = alpha * self.params.resistant_radiation_factor
            beta_r = beta * self.params.resistant_radiation_factor
            alpha_r_eff = alpha_r / oer
            beta_r_eff = beta_r / (oer ** 2)
            survival_resistant = np.exp(-alpha_r_eff * physical_dose -
                                        beta_r_eff * physical_dose ** 2)

            # Split population into treatable and resistant subpopulations
            treatable = population * (1.0 - self.params.resistant_fraction)
            resistant = population * self.params.resistant_fraction

            # Combined effect: treatable cells get full LQ kill,
            # resistant cells get heavily attenuated kill
            effect = (-(1.0 - survival) * treatable
                      - (1.0 - survival_resistant) * resistant)
            effects[phase] = effect
            total_killed -= float(np.sum(effect))

        metrics = {
            'dose_delivered': float(dose),
            'cumulative_dose': float(np.mean(self.cumulative_dose)),
            'total_cells_killed': total_killed,
            'mean_survival_fraction': float(np.mean(
                np.exp(-self.params.fractionation_alpha / oer * physical_dose -
                       self.params.fractionation_beta / (oer ** 2) *
                       physical_dose ** 2)
            )),
            'mean_oer': float(np.mean(oer)),
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

            # Split population into treatable and resistant subpopulations
            treatable = population * (1.0 - self.params.resistant_fraction)
            resistant = population * self.params.resistant_fraction

            # Resistant cells have heavily attenuated chemo sensitivity
            drug_effect_r = (drug_effect *
                             self.params.resistant_chemo_factor)

            effect = -drug_effect * treatable - drug_effect_r * resistant
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
