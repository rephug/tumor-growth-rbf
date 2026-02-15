"""
tumor_model.py

Integrated tumor growth model — the main simulation engine.

ARCHITECTURE
============
This model follows a clean separation of concerns:

    TumorModel (this file)
    ├── Spatial operations (RBF-FD): diffusion, gradients
    ├── CellPopulationModel: cell cycle transitions (local/pointwise)
    ├── ImmuneResponse: immune dynamics (local, spatial ops passed in)
    ├── TreatmentModule: treatment effects (local/pointwise)
    ├── TissueModel: spatially-varying coefficients
    └── MeshHandler: point management and adaptivity

The update loop each timestep:
    1. Solve oxygen transport PDE
    2. Update cell cycle transitions (biology)
    3. Compute and apply diffusion (spatial PDE)
    4. Compute growth with carrying capacity (biology + tissue)
    5. Update immune response (biology + spatial)
    6. Apply combined effects to tumor density
    7. Enforce physical constraints (non-negativity, carrying capacity)
    8. Adapt mesh if needed

ALL spatial derivatives are computed here using RBF-FD operators.
Biology modules only handle local (pointwise) reactions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

from ..core.rbf_solver import RBFSolver
from ..core.pde_assembler import PDEAssembler
from ..core.mesh_handler import MeshHandler
from .immune_response import ImmuneResponse, ImmuneParameters
from .treatments import TreatmentModule, TreatmentParameters
from .cell_populations import CellPopulationModel, CellCycleParameters
from .tissue_properties import TissueModel, TissueParameters, TissueType

logger = logging.getLogger(__name__)


@dataclass
class TumorParameters:
    """
    Parameters for tumor growth model.

    LEARNING NOTE: These are the "knobs" you turn to model different
    tumor types. Real parameter fitting uses clinical data (MRI scans
    over time, biopsy results) to calibrate these values.
    """
    # Growth
    growth_rate: float = 0.1            # Logistic growth rate (day⁻¹)
    carrying_capacity: float = 1.0      # Maximum local density (normalized)

    # Diffusion (mm²/day)
    diffusion_white: float = 0.1        # Base diffusion in white matter
    diffusion_grey: float = 0.01        # Base diffusion in grey matter

    # Oxygen dynamics
    oxygen_consumption: float = 0.1     # O₂ consumption rate
    oxygen_diffusion: float = 1.0       # O₂ diffusion coefficient
    hypoxia_threshold: float = 0.1      # O₂ level defining hypoxia

    # Accelerated repopulation
    # After radiation-induced cell death, surviving tumor cells detect
    # reduced local density and proliferate faster. Typically begins
    # 21-28 days after the start of radiotherapy.
    # Ref: Withers HR et al. Acta Oncol 27:131, 1988.
    repopulation_kick_time: float = 28.0  # Days after first treatment before acceleration
    repopulation_factor: float = 1.5      # Growth rate multiplier (1.0 = off, 2.0 = double)

    # Mesh
    min_spacing: float = 0.05           # Min point spacing (mm)
    max_spacing: float = 0.5            # Max point spacing (mm)
    refinement_threshold: float = 0.1   # Gradient threshold for refinement

    def validate(self):
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")


class TumorModel:
    """
    Comprehensive tumor growth model.

    This is the main class you interact with. Example:
        model = TumorModel(domain_size=(10.0, 10.0))
        for step in range(100):
            model.update(dt=0.1)
            if step % 10 == 0:
                print(model.get_metrics())
    """

    def __init__(self,
                 domain_size: Tuple[float, ...],
                 params: Optional[TumorParameters] = None,
                 immune_params: Optional[ImmuneParameters] = None,
                 treatment_params: Optional[TreatmentParameters] = None,
                 cell_cycle_params: Optional[CellCycleParameters] = None,
                 tissue_params: Optional[TissueParameters] = None,
                 n_initial_points: int = 500):
        """
        Args:
            domain_size: Physical domain size (Lx, Ly) or (Lx, Ly, Lz) in mm
            params: Tumor growth parameters
            immune_params: Immune response parameters
            treatment_params: Treatment parameters
            cell_cycle_params: Cell cycle parameters
            tissue_params: Tissue-specific parameters
            n_initial_points: Number of spatial points
        """
        self.domain_size = domain_size
        self.ndim = len(domain_size)
        self.params = params or TumorParameters()
        self.params.validate()

        # --- Core numerical components ---
        self.mesh = MeshHandler(
            domain_size,
            min_spacing=self.params.min_spacing,
            max_spacing=self.params.max_spacing
        )
        self.mesh.initialize_points(n_initial_points, distribution="halton")

        self.rbf_solver = RBFSolver(kernel="phs", phs_order=3, poly_degree=2)
        self.pde_assembler = PDEAssembler(self.rbf_solver)

        # --- Biology components ---
        self.cell_populations = CellPopulationModel(cell_cycle_params)
        self.immune_system = ImmuneResponse(immune_params)
        self.treatment_module = TreatmentModule(treatment_params)
        self.tissue_model = TissueModel(tissue_params)

        # --- State variables (1D arrays, one value per spatial point) ---
        self.tumor_density: Optional[np.ndarray] = None
        self.oxygen: Optional[np.ndarray] = None

        # Tissue property maps (1D arrays)
        self.diffusion_map: Optional[np.ndarray] = None
        self.growth_modifier_map: Optional[np.ndarray] = None
        self.oxygen_perfusion_map: Optional[np.ndarray] = None

        # --- Time tracking ---
        self.current_time: float = 0.0
        self._first_treatment_time: Optional[float] = None

        # --- Cached operators (rebuilt when mesh changes) ---
        self._laplacian_op = None
        self._grad_x_op = None
        self._grad_y_op = None
        self._grad_z_op = None
        self._operators_dirty = True

        # --- Initialize everything ---
        self._initialize_state()

    def _initialize_state(self):
        """Set up initial conditions for all fields."""
        n_points = len(self.mesh.points)

        # Initial tumor: Gaussian blob centered in domain
        center = np.array(self.domain_size) / 2.0
        distances = np.linalg.norm(self.mesh.points - center, axis=1)
        sigma = min(self.domain_size) / 10.0
        initial_density = np.exp(-distances ** 2 / (2 * sigma ** 2))

        # Initialize cell populations with realistic phase distribution
        self.cell_populations.initialize((n_points,))
        self.cell_populations.populations['G1'] = initial_density * 0.60
        self.cell_populations.populations['S'] = initial_density * 0.20
        self.cell_populations.populations['G2'] = initial_density * 0.15
        self.cell_populations.populations['M'] = initial_density * 0.05

        self.tumor_density = self.cell_populations.get_total_density()

        # Oxygen: starts at 1.0 everywhere (well-oxygenated)
        self.oxygen = np.ones(n_points)

        # Default tissue properties (uniform)
        self.diffusion_map = np.ones(n_points)
        self.growth_modifier_map = np.ones(n_points)
        self.oxygen_perfusion_map = np.ones(n_points)

        # Initialize other components
        self.immune_system.initialize((n_points,))
        self.treatment_module.initialize((n_points,))

        # Mark operators as needing rebuild
        self._operators_dirty = True

    def load_tissue_data(self,
                         tissue_image: np.ndarray,
                         tissue_labels: Dict[int, TissueType],
                         vessel_image: Optional[np.ndarray] = None):
        """Load tissue type information from medical imaging data."""
        self.tissue_model.initialize_from_image(
            tissue_image, tissue_labels, vessel_image
        )
        self._update_tissue_properties()

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """
        Advance the simulation by one time step.

        Args:
            dt: Time step size in days
        """
        # Advance simulation clock
        self.current_time += dt

        # Rebuild operators if mesh changed
        if self._operators_dirty:
            self._rebuild_operators()

        # Store previous state for adaptivity
        prev_density = self.tumor_density.copy()

        # 1. Update oxygen distribution
        self._update_oxygen(dt)

        # 2. Update cell cycle transitions (pointwise biology)
        self.cell_populations.update(dt, self.oxygen)

        # 3. Get updated total density
        self.tumor_density = self.cell_populations.get_total_density()

        # 4. Compute spatial effects: diffusion
        diffusion = self._compute_diffusion()

        # 5. Compute growth with tissue modifiers
        growth = self._compute_growth()

        # 6. Update immune response (with spatial operators)
        immune_effect = self._update_immune(dt)

        # 7. Apply combined effects
        self.tumor_density += dt * (growth + diffusion + immune_effect)

        # 8. Enforce physical constraints
        np.clip(self.tumor_density, 0, self.params.carrying_capacity,
                out=self.tumor_density)

        # 9. Redistribute density across cell populations
        self._redistribute_populations()

        # 10. Adapt mesh (optional, can skip for learning)
        # self._adapt_mesh(prev_density)

    def _rebuild_operators(self):
        """Rebuild RBF-FD operators (expensive — only when mesh changes)."""
        logger.debug("Rebuilding RBF-FD operators...")

        self._laplacian_op = self.pde_assembler.build_operator(
            self.mesh.points, self.mesh.neighbor_lists, "laplacian"
        )
        self._grad_x_op = self.pde_assembler.build_operator(
            self.mesh.points, self.mesh.neighbor_lists, "gradient_x"
        )
        self._grad_y_op = self.pde_assembler.build_operator(
            self.mesh.points, self.mesh.neighbor_lists, "gradient_y"
        )
        if self.ndim == 3:
            self._grad_z_op = self.pde_assembler.build_operator(
                self.mesh.points, self.mesh.neighbor_lists, "gradient_z"
            )

        self._operators_dirty = False
        logger.debug(f"Operators built for {len(self.mesh.points)} points")

    def _update_oxygen(self, dt: float):
        """
        Update oxygen distribution.

        PDE: ∂O/∂t = D_O ∇²O - consumption + production

        where:
        - consumption = rate * tumor_density * O  (cells consume O₂)
        - production = perfusion * (1 - tumor_density)  (from blood vessels)

        Proliferating cells (S, G2, M) consume more oxygen than
        quiescent cells — this drives the hypoxic core formation.
        """
        # Weighted oxygen consumption by cell cycle phase
        high_consumption = sum(
            self.cell_populations.populations[p] for p in ['S', 'G2', 'M']
        )
        low_consumption = 0.5 * self.cell_populations.populations['G1']
        proliferating_density = high_consumption + low_consumption

        consumption = (self.params.oxygen_consumption *
                       proliferating_density * self.oxygen)

        # Production from perfusion (decreases where tumor is dense)
        production = self.oxygen_perfusion_map * (1.0 - self.tumor_density)
        production = np.clip(production, 0, None)

        # Diffusion of oxygen
        oxygen_diffusion = self.params.oxygen_diffusion * (
            self._laplacian_op @ self.oxygen
        )

        # Forward Euler update
        self.oxygen += dt * (oxygen_diffusion - consumption + production)
        np.clip(self.oxygen, 0, 1, out=self.oxygen)

    def _compute_diffusion(self) -> np.ndarray:
        """
        Compute tumor diffusion term: D(x) * ∇²u

        LEARNING NOTE: This is where tissue heterogeneity matters most.
        D(x) varies by tissue type:
        - White matter: D = 0.1 mm²/day (tumor spreads along fibers)
        - Grey matter: D = 0.01 mm²/day (10x slower)

        This creates the "butterfly" patterns seen in real gliomas
        that follow white matter tracts.
        """
        D = self.params.diffusion_white * self.diffusion_map
        return D * (self._laplacian_op @ self.tumor_density)

    def _compute_growth(self) -> np.ndarray:
        """
        Compute tumor growth term with logistic saturation.

        PDE: growth = ρ_eff * u * (1 - u/K) * tissue_modifier

        This is LOGISTIC GROWTH:
        - When u << K: growth ≈ ρ*u (exponential)
        - When u → K: growth → 0 (carrying capacity)

        ACCELERATED REPOPULATION: After treatment begins, surviving
        tumor cells detect reduced local density and upregulate
        proliferation. This kicks in after repopulation_kick_time days
        and ramps up over 14 days to avoid a discontinuous jump.
        Ref: Withers HR et al. Acta Oncol 27:131, 1988.

        Oxygen dependence: hypoxic regions grow 10x slower
        (cells are quiescent, but those still cycling grow slowly).
        """
        # Compute effective growth rate with accelerated repopulation
        effective_growth_rate = self.params.growth_rate
        if self._first_treatment_time is not None:
            time_since_treatment = (self.current_time -
                                    self._first_treatment_time)
            kick = self.params.repopulation_kick_time
            if time_since_treatment > kick:
                # Ramp up over 14 days to prevent discontinuous jump
                ramp = min(1.0, (time_since_treatment - kick) / 14.0)
                accel = 1.0 + (self.params.repopulation_factor - 1.0) * ramp
                effective_growth_rate = self.params.growth_rate * accel

        growth = (effective_growth_rate *
                  self.tumor_density *
                  (1.0 - self.tumor_density / self.params.carrying_capacity))

        # Tissue-specific growth modifier
        growth *= self.growth_modifier_map

        # Hypoxic reduction
        hypoxic = self.oxygen < self.params.hypoxia_threshold
        growth[hypoxic] *= 0.1

        return growth

    def _update_immune(self, dt: float) -> np.ndarray:
        """
        Update immune response, providing spatial operators.

        This is where we bridge the local immune model with
        global spatial operators computed by RBF-FD.
        """
        # Compute spatial derivatives of chemokine field for the immune module
        laplacian_chemokine = None
        chemokine_gradient = None

        if self.immune_system.chemokine_concentration is not None:
            chemokine = self.immune_system.chemokine_concentration
            laplacian_chemokine = self._laplacian_op @ chemokine
            grad_x = self._grad_x_op @ chemokine
            grad_y = self._grad_y_op @ chemokine
            if self.ndim == 3:
                grad_z = self._grad_z_op @ chemokine
                chemokine_gradient = (grad_x, grad_y, grad_z)
            else:
                chemokine_gradient = (grad_x, grad_y)

        immune_effect, _ = self.immune_system.update(
            dt, self.tumor_density, self.oxygen,
            laplacian_chemokine=laplacian_chemokine,
            chemokine_gradient=chemokine_gradient
        )

        return immune_effect

    def _redistribute_populations(self):
        """
        After spatial transport, redistribute total density across phases.

        The spatial PDE modifies total density, but doesn't know about
        cell cycle phases. We redistribute proportionally:
        each phase keeps its current FRACTION of the total.
        """
        total_before = self.cell_populations.get_total_density()
        safe_total = total_before + 1e-10

        for phase in self.cell_populations.PHASES:
            fraction = self.cell_populations.populations[phase] / safe_total
            self.cell_populations.populations[phase] = fraction * self.tumor_density

    def _update_tissue_properties(self):
        """Update spatial property maps from tissue model."""
        self.diffusion_map = self.tissue_model.get_diffusion_coefficient_map()
        self.growth_modifier_map = self.tissue_model.get_growth_modifier_map()
        self.oxygen_perfusion_map = self.tissue_model.get_oxygen_perfusion_map()

    def set_initial_density(self, density: np.ndarray):
        """
        Replace tumor density with a patient-specific initial condition.

        Used by the parameter fitter to initialize the model with a
        patient's tumor contour from imaging data (at time t₁) instead
        of the default Gaussian blob.

        Args:
            density: 1D array of tumor density values, one per mesh point.
                     Must have length == len(self.mesh.points).
        """
        if len(density) != len(self.mesh.points):
            raise ValueError(
                f"Density array length ({len(density)}) does not match "
                f"number of mesh points ({len(self.mesh.points)})"
            )
        self.tumor_density = density.copy()

        # Redistribute across cell cycle phases with realistic fractions
        self.cell_populations.populations['G1'] = density * 0.60
        self.cell_populations.populations['S'] = density * 0.20
        self.cell_populations.populations['G2'] = density * 0.15
        self.cell_populations.populations['M'] = density * 0.05

    # ------------------------------------------------------------------
    # Treatment interface
    # ------------------------------------------------------------------

    def apply_treatment(self,
                        treatment_type: str,
                        **treatment_params) -> Dict:
        """
        Apply a treatment to the tumor.

        Args:
            treatment_type: "radiation", "chemo", or "immunotherapy"
            **treatment_params: Treatment-specific parameters

        Returns:
            Treatment metrics dictionary
        """
        # Record first treatment time for accelerated repopulation
        if self._first_treatment_time is None:
            self._first_treatment_time = self.current_time

        # Get tissue-specific modifiers
        if self.tissue_model.tissue_map is not None:
            rad_mod, drug_mod = self.tissue_model.get_treatment_modifier_maps()
            if treatment_type == "radiation":
                treatment_params.setdefault("radiation_modifier", rad_mod)
            elif treatment_type in ("chemo", "chemotherapy"):
                treatment_params.setdefault("drug_modifier", drug_mod)

        # Apply treatment
        effects, metrics = self.treatment_module.apply_treatment(
            treatment_type,
            self.cell_populations.populations,
            self.oxygen,
            self.immune_system.immune_density,
            **treatment_params
        )

        # Apply effects to each population
        for phase, effect in effects.items():
            self.cell_populations.populations[phase] += effect
            np.clip(self.cell_populations.populations[phase], 0, None,
                    out=self.cell_populations.populations[phase])

        # Update total density
        self.tumor_density = self.cell_populations.get_total_density()

        return metrics

    # ------------------------------------------------------------------
    # Metrics and output
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        """Get comprehensive model metrics."""
        return {
            "tumor": {
                "total_mass": float(np.sum(self.tumor_density)),
                "max_density": float(np.max(self.tumor_density)),
                "mean_density": float(np.mean(self.tumor_density)),
                "hypoxic_fraction": float(
                    np.mean(self.oxygen < self.params.hypoxia_threshold)
                ),
            },
            "cell_populations": self.cell_populations.get_metrics(),
            "mesh": self.mesh.get_metrics(),
            "immune": self.immune_system.get_metrics(),
            "treatment": self.treatment_module.get_metrics(),
        }

    def _compute_gradient_magnitude(self) -> np.ndarray:
        """Compute |∇u| for refinement indicators."""
        dx = self._grad_x_op @ self.tumor_density
        dy = self._grad_y_op @ self.tumor_density
        mag_sq = dx ** 2 + dy ** 2
        if self.ndim == 3:
            dz = self._grad_z_op @ self.tumor_density
            mag_sq += dz ** 2
        return np.sqrt(mag_sq)


if __name__ == "__main__":
    # Quick demo
    model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=200)

    dt = 0.1
    for step in range(100):
        model.update(dt)
        if step % 20 == 0:
            m = model.get_metrics()
            day = step * dt
            print(f"Day {day:.1f}: mass={m['tumor']['total_mass']:.3f}, "
                  f"max={m['tumor']['max_density']:.3f}, "
                  f"hypoxic={m['tumor']['hypoxic_fraction']:.2%}")

        # Radiation at day 5
        if step == 50:
            result = model.apply_treatment("radiation", dose=2.0)
            print(f"  → Radiation: killed {result['total_cells_killed']:.1f} cells")
