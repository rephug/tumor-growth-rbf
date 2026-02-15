"""
cell_populations.py

Models tumor cell populations through the cell cycle.

BIOLOGY BACKGROUND
==================
Tumor cells, like all dividing cells, go through a cycle:

    G1 (Gap 1) → S (Synthesis) → G2 (Gap 2) → M (Mitosis) → 2x G1
        ↓              ↓              ↓             ↓
      [grow]     [copy DNA]     [prepare]     [divide!]

Additional states:
    Q (Quiescent/G0): Cells that have exited the cycle due to stress
                      (low oxygen, crowding). Can re-enter if conditions improve.
    N (Necrotic): Dead cells. Result of prolonged oxygen deprivation.

KEY CLINICAL RELEVANCE:
- Many chemo drugs target S-phase (DNA synthesis) → S-phase specific
- Radiation is most effective in M and G2 phases
- Quiescent cells are resistant to most treatments → why tumors recur
- The fraction of cells in each phase matters enormously for treatment planning

Typical cell cycle duration for mammalian tumor cells:
    G1: ~10 hours, S: ~8 hours, G2: ~4 hours, M: ~2 hours
    Total: ~24 hours (but highly variable between tumor types)

Transition rates are modeled as exponential processes:
    Rate = 1 / (mean duration)
    Probability of transition in time dt: 1 - exp(-rate * dt)
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CellCycleParameters:
    """
    Parameters for cell cycle transitions and population dynamics.

    All rates are in inverse hours (h⁻¹).
    Thresholds are dimensionless oxygen concentrations (0-1 scale).
    """
    # Cell cycle transition rates
    # Rate = 1 / (mean phase duration in hours)
    g1_to_s_rate: float = 1 / 10.0   # G1 → S: ~10 hours
    s_to_g2_rate: float = 1 / 8.0    # S → G2: ~8 hours
    g2_to_m_rate: float = 1 / 4.0    # G2 → M: ~4 hours
    m_to_g1_rate: float = 1 / 2.0    # M → G1: ~2 hours (produces 2 daughter cells)

    # Oxygen thresholds
    hypoxia_threshold: float = 0.1          # Below this → quiescence
    severe_hypoxia_threshold: float = 0.01  # Below this → necrosis

    # Survival parameters
    necrosis_rate: float = 0.1         # Rate of necrotic cell clearance (h⁻¹)
    quiescent_survival_time: float = 48.0  # Hours cells survive in quiescence

    def validate(self):
        """Validate all parameters are non-negative."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative, got {value}")


class CellPopulationModel:
    """
    Manages cell populations across all cell cycle phases.

    DESIGN NOTE: This model handles only LOCAL (pointwise) dynamics.
    Spatial effects (diffusion, migration) are handled by the tumor model.
    This clean separation means:
    - Cell populations can be tested independently
    - Easy to swap in different cell cycle models
    - No dependence on mesh/spatial discretization
    """

    # All tracked phases
    PHASES = ['G1', 'S', 'G2', 'M', 'Q', 'N']
    PROLIFERATING_PHASES = ['G1', 'S', 'G2', 'M']

    def __init__(self, params: Optional[CellCycleParameters] = None):
        self.params = params or CellCycleParameters()
        self.params.validate()

        self.populations: Dict[str, Optional[np.ndarray]] = {
            phase: None for phase in self.PHASES
        }
        self.quiescence_time: Optional[np.ndarray] = None

    def initialize(self, shape: Tuple[int, ...]):
        """
        Initialize all population arrays to zeros.

        Args:
            shape: Shape of each population array (matches spatial points)
        """
        for phase in self.PHASES:
            self.populations[phase] = np.zeros(shape)
        self.quiescence_time = np.zeros(shape)

    def update(self, dt: float, oxygen: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update cell populations for one time step.

        The update order matters biologically:
        1. First, handle oxygen effects (quiescence, necrosis)
        2. Then, advance the cell cycle for surviving cells
        3. Finally, clear necrotic debris

        Args:
            dt: Time step in DAYS (converted internally to hours)
            oxygen: Oxygen concentration field (0-1 scale)

        Returns:
            Dictionary of updated population arrays
        """
        # Convert dt from days to hours (cell cycle rates are in hours)
        dt_hours = dt * 24.0

        # Step 1: Oxygen-dependent transitions
        self._handle_oxygen_effects(oxygen)

        # Step 2: Cell cycle progression (only for normally oxygenated cells)
        self._update_cell_cycle(dt_hours)

        # Step 3: Quiescent cell survival tracking
        self._update_quiescent_cells(dt_hours)

        # Step 4: Necrotic cell clearance
        self._clear_necrotic_cells(dt_hours)

        return self.populations

    def _handle_oxygen_effects(self, oxygen: np.ndarray):
        """
        Handle cell state transitions based on oxygen levels.

        BIOLOGY: Oxygen is the primary driver of cell fate decisions.
        - Normal O₂ (>0.1): Cells proliferate normally
        - Moderate hypoxia (0.01-0.1): Cells enter quiescence (G0)
          → Mediated by HIF-1α signaling
        - Severe hypoxia (<0.01): Cells die (necrosis)
          → Irreversible membrane damage

        In real tumors, this creates the classic layered structure:
            [proliferating rim] → [quiescent zone] → [necrotic core]
        """
        # Severe hypoxia → immediate necrosis (all phases)
        severe_hypoxia = oxygen < self.params.severe_hypoxia_threshold
        if np.any(severe_hypoxia):
            for phase in self.PROLIFERATING_PHASES + ['Q']:
                self.populations['N'][severe_hypoxia] += \
                    self.populations[phase][severe_hypoxia]
                self.populations[phase][severe_hypoxia] = 0.0

        # Moderate hypoxia → quiescence (proliferating phases only)
        hypoxia = ((oxygen >= self.params.severe_hypoxia_threshold) &
                   (oxygen < self.params.hypoxia_threshold))
        if np.any(hypoxia):
            for phase in self.PROLIFERATING_PHASES:
                self.populations['Q'][hypoxia] += \
                    self.populations[phase][hypoxia]
                self.populations[phase][hypoxia] = 0.0

        # Recovery: normal oxygen → quiescent cells re-enter G1
        normal_oxygen = oxygen >= self.params.hypoxia_threshold
        recovering = normal_oxygen & (self.populations['Q'] > 0)
        if np.any(recovering):
            self.populations['G1'][recovering] += \
                self.populations['Q'][recovering]
            self.populations['Q'][recovering] = 0.0
            self.quiescence_time[recovering] = 0.0

    def _update_cell_cycle(self, dt_hours: float):
        """
        Advance cell cycle transitions.

        MATH NOTE: We model transitions as first-order kinetics:
            dN_phase/dt = -rate * N_phase

        The fraction transitioning in time dt is:
            fraction = 1 - exp(-rate * dt)

        This is exact for exponential decay (no approximation error),
        and ensures the fraction is always between 0 and 1.

        BIOLOGY NOTE: M → G1 produces 2 daughter cells (cell division!).
        This is the ONLY source of new cells. The factor of 2 is what
        drives exponential tumor growth.
        """
        # G1 → S
        g1_to_s = self.populations['G1'] * \
                  (1.0 - np.exp(-self.params.g1_to_s_rate * dt_hours))
        self.populations['G1'] -= g1_to_s
        self.populations['S'] += g1_to_s

        # S → G2
        s_to_g2 = self.populations['S'] * \
                  (1.0 - np.exp(-self.params.s_to_g2_rate * dt_hours))
        self.populations['S'] -= s_to_g2
        self.populations['G2'] += s_to_g2

        # G2 → M
        g2_to_m = self.populations['G2'] * \
                  (1.0 - np.exp(-self.params.g2_to_m_rate * dt_hours))
        self.populations['G2'] -= g2_to_m
        self.populations['M'] += g2_to_m

        # M → G1 (DIVISION: one cell becomes two!)
        m_to_g1 = self.populations['M'] * \
                  (1.0 - np.exp(-self.params.m_to_g1_rate * dt_hours))
        self.populations['M'] -= m_to_g1
        self.populations['G1'] += 2.0 * m_to_g1  # ← the factor of 2!

    def _update_quiescent_cells(self, dt_hours: float):
        """
        Track quiescence duration and trigger necrosis if too long.

        BIOLOGY: Quiescent cells can survive without oxygen for a while
        by switching to anaerobic metabolism, but this is limited.
        After ~48 hours, accumulated metabolic waste and ATP depletion
        lead to cell death.
        """
        quiescent_mask = self.populations['Q'] > 0
        if not np.any(quiescent_mask):
            return

        self.quiescence_time[quiescent_mask] += dt_hours

        # Cells that have been quiescent too long → necrosis
        death_mask = self.quiescence_time > self.params.quiescent_survival_time
        if np.any(death_mask):
            self.populations['N'][death_mask] += \
                self.populations['Q'][death_mask]
            self.populations['Q'][death_mask] = 0.0
            self.quiescence_time[death_mask] = 0.0

    def _clear_necrotic_cells(self, dt_hours: float):
        """
        Model clearance of necrotic debris.

        BIOLOGY: Dead cells are gradually cleared by:
        - Phagocytosis (macrophages eating debris)
        - Diffusion and dissolution
        This is a slow process, which is why necrotic cores persist
        in real tumors.
        """
        clearance = self.populations['N'] * \
                    (1.0 - np.exp(-self.params.necrosis_rate * dt_hours))
        self.populations['N'] -= clearance

    def get_total_density(self) -> np.ndarray:
        """Get total cell density (sum of all populations)."""
        return sum(pop for pop in self.populations.values()
                   if pop is not None)

    def get_viable_density(self) -> np.ndarray:
        """Get density of living (non-necrotic) cells."""
        return sum(self.populations[p] for p in self.PROLIFERATING_PHASES + ['Q'])

    def get_proliferating_density(self) -> np.ndarray:
        """Get density of actively cycling cells."""
        return sum(self.populations[p] for p in self.PROLIFERATING_PHASES)

    def get_metrics(self) -> Dict:
        """
        Calculate population metrics.

        Returns per-phase fractions and aggregate statistics.
        These are the numbers oncologists care about:
        - Proliferating fraction → tumor aggressiveness
        - Quiescent fraction → treatment resistance reservoir
        - Necrotic fraction → tumor maturity/hypoxia severity
        """
        total = self.get_total_density()
        total_sum = float(np.sum(total))

        # Avoid division by zero
        safe_total = total + 1e-10

        metrics = {
            'total_cells': total_sum,
        }

        # Per-phase fractions (FIX: these were missing in original code!)
        for phase in self.PHASES:
            pop = self.populations[phase]
            if pop is not None:
                fraction = float(np.mean(pop / safe_total))
                metrics[f'{phase.lower()}_fraction'] = fraction
            else:
                metrics[f'{phase.lower()}_fraction'] = 0.0

        # Aggregate fractions
        metrics['proliferating_fraction'] = float(np.mean(
            self.get_proliferating_density() / safe_total
        ))
        metrics['quiescent_fraction'] = float(np.mean(
            self.populations['Q'] / safe_total
        ))
        metrics['necrotic_fraction'] = float(np.mean(
            self.populations['N'] / safe_total
        ))

        # Quiescence time
        q_mask = self.populations['Q'] > 0
        if np.any(q_mask):
            metrics['mean_quiescence_time'] = float(
                np.mean(self.quiescence_time[q_mask])
            )
        else:
            metrics['mean_quiescence_time'] = 0.0

        return metrics
