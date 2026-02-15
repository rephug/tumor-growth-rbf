"""
Test suite for patient-specific parameter fitting (Phase 1B).
Run with: pytest tests/test_parameter_fitting.py -v
"""

import numpy as np
import pytest

from tumor_growth_rbf import (
    TumorModel, TumorParameters,
    ParameterFitter, FittingResult,
    MeshHandler,
)


# =============================================================
# Dice Coefficient
# =============================================================

class TestDiceCoefficient:
    def test_identical_arrays(self):
        """Identical fields should give Dice = 1.0."""
        a = np.array([0.0, 0.5, 0.8, 1.0, 0.3])
        dice = ParameterFitter.dice_coefficient(a, a, threshold=0.1)
        assert dice == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping fields should give Dice = 0.0."""
        a = np.array([0.0, 0.0, 0.5, 0.8])
        b = np.array([0.5, 0.8, 0.0, 0.0])
        dice = ParameterFitter.dice_coefficient(a, b, threshold=0.1)
        assert dice == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap should give 0 < Dice < 1."""
        a = np.array([0.0, 0.5, 0.5, 0.0])
        b = np.array([0.0, 0.0, 0.5, 0.5])
        dice = ParameterFitter.dice_coefficient(a, b, threshold=0.1)
        # A = {1, 2}, B = {2, 3}, intersection = {2}
        # Dice = 2*1 / (2+2) = 0.5
        assert dice == pytest.approx(0.5)

    def test_empty_arrays(self):
        """Both-empty fields should return 0.0."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        dice = ParameterFitter.dice_coefficient(a, b, threshold=0.1)
        assert dice == pytest.approx(0.0)


# =============================================================
# FittingResult Dataclass
# =============================================================

class TestFittingResult:
    def test_dataclass_creation(self):
        """FittingResult should store all fields correctly."""
        result = FittingResult(
            best_rho=0.05,
            best_D=0.1,
            best_dice=0.85,
            rho_grid=np.linspace(0.01, 0.1, 10),
            D_grid=np.linspace(0.01, 1.0, 10),
            dice_surface=np.random.rand(10, 10),
            n_simulations=100,
            refinement_used=False,
        )
        assert result.best_rho == 0.05
        assert result.best_D == 0.1
        assert result.best_dice == 0.85
        assert result.n_simulations == 100
        assert result.refinement_used is False

    def test_dice_surface_shape(self):
        """Dice surface should match grid dimensions."""
        n_rho, n_D = 8, 12
        result = FittingResult(
            best_rho=0.05,
            best_D=0.1,
            best_dice=0.85,
            rho_grid=np.linspace(0.01, 0.1, n_rho),
            D_grid=np.linspace(0.01, 1.0, n_D),
            dice_surface=np.random.rand(n_rho, n_D),
            n_simulations=n_rho * n_D,
            refinement_used=False,
        )
        assert result.dice_surface.shape == (n_rho, n_D)
        assert len(result.rho_grid) == n_rho
        assert len(result.D_grid) == n_D


# =============================================================
# set_initial_density() on TumorModel
# =============================================================

class TestSetInitialDensity:
    def test_density_replaced(self):
        """set_initial_density should replace the tumor density."""
        model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=100)
        n = len(model.mesh.points)
        custom = np.random.rand(n) * 0.5
        model.set_initial_density(custom)
        np.testing.assert_array_almost_equal(model.tumor_density, custom)

    def test_phase_distribution(self):
        """Phases should be 60/20/15/5 of density."""
        model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=100)
        n = len(model.mesh.points)
        custom = np.ones(n) * 0.8
        model.set_initial_density(custom)

        np.testing.assert_array_almost_equal(
            model.cell_populations.populations['G1'], 0.8 * 0.60)
        np.testing.assert_array_almost_equal(
            model.cell_populations.populations['S'], 0.8 * 0.20)
        np.testing.assert_array_almost_equal(
            model.cell_populations.populations['G2'], 0.8 * 0.15)
        np.testing.assert_array_almost_equal(
            model.cell_populations.populations['M'], 0.8 * 0.05)

    def test_zero_density(self):
        """Zero density should zero out all phases."""
        model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=100)
        n = len(model.mesh.points)
        model.set_initial_density(np.zeros(n))
        assert np.sum(model.tumor_density) == 0.0
        for phase in ['G1', 'S', 'G2', 'M']:
            assert np.sum(model.cell_populations.populations[phase]) == 0.0

    def test_wrong_length_raises(self):
        """Wrong density length should raise ValueError."""
        model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=100)
        with pytest.raises(ValueError, match="does not match"):
            model.set_initial_density(np.zeros(5))


# =============================================================
# ParameterFitter
# =============================================================

class TestParameterFitter:
    def test_initialization(self):
        """Fitter should store configuration."""
        fitter = ParameterFitter(
            domain_size=(10.0, 10.0), n_points=200, dt=0.2
        )
        assert fitter.domain_size == (10.0, 10.0)
        assert fitter.n_points == 200
        assert fitter.dt == 0.2

    def test_single_evaluation(self):
        """A single (ρ, D) evaluation should return Dice in [0, 1]."""
        fitter = ParameterFitter(
            domain_size=(10.0, 10.0), n_points=100, dt=0.5
        )
        # Create a simple initial condition
        model = TumorModel(
            domain_size=(10.0, 10.0), n_initial_points=100
        )
        initial = model.tumor_density.copy()

        # Simulate a bit to get "observed"
        for _ in range(5):
            model.update(0.5)
        observed = model.tumor_density.copy()

        dice = fitter._evaluate(0.05, 0.1, initial, observed, delta_t=2.5)
        assert 0.0 <= dice <= 1.0

    def test_grid_search_shape(self):
        """Grid search should produce correctly shaped dice surface."""
        fitter = ParameterFitter(
            domain_size=(10.0, 10.0), n_points=100, dt=0.5
        )
        model = TumorModel(
            domain_size=(10.0, 10.0), n_initial_points=100
        )
        initial = model.tumor_density.copy()
        for _ in range(3):
            model.update(0.5)
        observed = model.tumor_density.copy()

        n_grid = 3
        result = fitter.fit(
            initial_density=initial,
            observed_density=observed,
            delta_t=1.5,
            rho_range=(0.01, 0.1),
            D_range=(0.01, 0.5),
            n_grid=n_grid,
        )
        assert result.dice_surface.shape == (n_grid, n_grid)
        assert result.n_simulations == n_grid * n_grid
        assert 0.0 <= result.best_dice <= 1.0

    def test_known_parameters_recovery(self):
        """
        KEY TEST: Generate synthetic data with known (ρ*, D*),
        then verify the fitter recovers them.

        We use a short simulation (3 days) so the tumor grows enough
        to see differences but not so much that secondary effects
        (immune response, oxygen depletion) dominate. The fitter
        creates fresh models each time, so only the primary PDE
        dynamics (diffusion + logistic growth) matter for matching.
        """
        # Ground truth parameters
        rho_true = 0.05
        D_true = 0.05

        # Generate synthetic "patient" data using identical setup as fitter
        params = TumorParameters(
            growth_rate=rho_true,
            diffusion_white=D_true,
        )
        model = TumorModel(
            domain_size=(10.0, 10.0),
            params=params,
            n_initial_points=200,
        )
        initial = model.tumor_density.copy()

        # Short simulation: 3 days at dt=0.5
        delta_t = 3.0
        dt = 0.5
        for _ in range(int(delta_t / dt)):
            model.update(dt)
        observed = model.tumor_density.copy()

        # Run fitter with a focused grid
        fitter = ParameterFitter(
            domain_size=(10.0, 10.0), n_points=200, dt=dt
        )
        result = fitter.fit(
            initial_density=initial,
            observed_density=observed,
            delta_t=delta_t,
            rho_range=(0.02, 0.08),
            D_range=(0.02, 0.08),
            n_grid=5,
        )

        # The best-fit parameters should be close to ground truth
        # Grid spacing is 0.015, so within one grid cell is acceptable
        assert abs(result.best_rho - rho_true) <= 0.016, \
            f"rho recovery: got {result.best_rho}, expected ~{rho_true}"
        assert abs(result.best_D - D_true) <= 0.016, \
            f"D recovery: got {result.best_D}, expected ~{D_true}"

        # Dice should be very high at the optimum
        assert result.best_dice > 0.9, \
            f"Best Dice too low: {result.best_dice}"

    def test_refinement(self):
        """Nelder-Mead refinement should maintain or improve Dice."""
        params = TumorParameters(growth_rate=0.05, diffusion_white=0.05)
        model = TumorModel(
            domain_size=(10.0, 10.0),
            params=params,
            n_initial_points=150,
        )
        initial = model.tumor_density.copy()

        delta_t = 3.0
        dt = 0.5
        for _ in range(int(delta_t / dt)):
            model.update(dt)
        observed = model.tumor_density.copy()

        fitter = ParameterFitter(
            domain_size=(10.0, 10.0), n_points=150, dt=dt
        )

        # Grid search only
        result_grid = fitter.fit(
            initial_density=initial,
            observed_density=observed,
            delta_t=delta_t,
            rho_range=(0.01, 0.1),
            D_range=(0.01, 0.1),
            n_grid=3,
            refine=False,
        )

        # Grid search + refinement
        result_refined = fitter.fit(
            initial_density=initial,
            observed_density=observed,
            delta_t=delta_t,
            rho_range=(0.01, 0.1),
            D_range=(0.01, 0.1),
            n_grid=3,
            refine=True,
        )

        # Refined should be at least as good
        assert result_refined.best_dice >= result_grid.best_dice - 0.01


# =============================================================
# Parallel Fitting
# =============================================================

class TestParallelFitting:
    def test_parallel_matches_sequential(self):
        """Parallel and sequential grid search should give same results."""
        model = TumorModel(
            domain_size=(10.0, 10.0), n_initial_points=100
        )
        initial = model.tumor_density.copy()
        for _ in range(3):
            model.update(0.5)
        observed = model.tumor_density.copy()

        fitter = ParameterFitter(
            domain_size=(10.0, 10.0), n_points=100, dt=0.5
        )

        result_seq = fitter.fit(
            initial_density=initial,
            observed_density=observed,
            delta_t=1.5,
            rho_range=(0.01, 0.1),
            D_range=(0.01, 0.5),
            n_grid=3,
            parallel=False,
        )

        result_par = fitter.fit(
            initial_density=initial,
            observed_density=observed,
            delta_t=1.5,
            rho_range=(0.01, 0.1),
            D_range=(0.01, 0.5),
            n_grid=3,
            parallel=True,
        )

        # Same best parameters and Dice
        assert result_seq.best_rho == pytest.approx(result_par.best_rho, abs=1e-6)
        assert result_seq.best_D == pytest.approx(result_par.best_D, abs=1e-6)
        assert result_seq.best_dice == pytest.approx(result_par.best_dice, abs=0.01)
