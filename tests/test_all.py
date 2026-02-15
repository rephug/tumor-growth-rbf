"""
Test suite for tumor growth simulator.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from tumor_growth_rbf import (
    TumorModel, TumorParameters,
    CellPopulationModel, CellCycleParameters,
    TreatmentModule, TreatmentParameters,
    ImmuneResponse, ImmuneParameters,
    MeshHandler, RBFSolver, PDEAssembler,
    TissueModel, TissueParameters, TissueType,
)


# =============================================================
# Core: Mesh Handler
# =============================================================

class TestMeshHandler:
    def test_initialization(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0))
        mesh.initialize_points(100, distribution="halton")
        assert len(mesh.points) == 100
        assert mesh.points.shape[1] == 2

    def test_points_in_domain(self):
        mesh = MeshHandler(domain_size=(5.0, 8.0))
        mesh.initialize_points(200, distribution="halton")
        assert np.all(mesh.points[:, 0] >= 0)
        assert np.all(mesh.points[:, 0] <= 5.0)
        assert np.all(mesh.points[:, 1] >= 0)
        assert np.all(mesh.points[:, 1] <= 8.0)

    def test_neighbor_lists_populated(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0))
        mesh.initialize_points(50, distribution="halton")
        assert mesh.neighbor_lists is not None
        assert len(mesh.neighbor_lists) == 50
        # Each point should have at least itself as neighbor
        for nbrs in mesh.neighbor_lists:
            assert len(nbrs) >= 1

    def test_distributions(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0))
        for dist in ["halton", "random", "grid"]:
            mesh.initialize_points(50, distribution=dist)
            assert len(mesh.points) >= 40  # Grid may round differently

    def test_refine_points(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0),
                           min_spacing=0.1, max_spacing=1.0)
        mesh.initialize_points(50, distribution="halton")
        initial_count = len(mesh.points)

        # Refine everywhere
        indicator = np.ones(initial_count)
        n_added = mesh.refine_points(indicator, threshold=0.5)
        assert len(mesh.points) >= initial_count

    def test_metrics(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0))
        mesh.initialize_points(100, distribution="halton")
        metrics = mesh.get_metrics()
        assert metrics["n_points"] == 100
        assert metrics["min_spacing"] > 0
        assert metrics["mean_spacing"] > 0


# =============================================================
# Core: RBF Solver
# =============================================================

class TestRBFSolver:
    def test_laplacian_of_quadratic(self):
        """∇²(x² + y²) = 4 everywhere — exact for poly_degree=2."""
        rbf = RBFSolver(epsilon=1.0, poly_degree=2)
        mesh = MeshHandler(domain_size=(10.0, 10.0))
        mesh.initialize_points(300, distribution="halton")

        f = mesh.points[:, 0] ** 2 + mesh.points[:, 1] ** 2
        L = rbf.assemble_global_operator(
            mesh.points, mesh.neighbor_lists, "laplacian"
        )
        result = L @ f

        # Check interior points (boundary points may be inaccurate)
        interior = ((mesh.points[:, 0] > 1) & (mesh.points[:, 0] < 9) &
                    (mesh.points[:, 1] > 1) & (mesh.points[:, 1] < 9))
        assert np.allclose(result[interior], 4.0, atol=1e-6)

    def test_gradient_x_of_linear(self):
        """∂/∂x(3x + 2y) = 3 everywhere."""
        rbf = RBFSolver(epsilon=1.0, poly_degree=2)
        mesh = MeshHandler(domain_size=(10.0, 10.0))
        mesh.initialize_points(300, distribution="halton")

        f = 3 * mesh.points[:, 0] + 2 * mesh.points[:, 1]
        Gx = rbf.assemble_global_operator(
            mesh.points, mesh.neighbor_lists, "gradient_x"
        )
        result = Gx @ f

        interior = ((mesh.points[:, 0] > 1) & (mesh.points[:, 0] < 9) &
                    (mesh.points[:, 1] > 1) & (mesh.points[:, 1] < 9))
        assert np.allclose(result[interior], 3.0, atol=1e-4)


# =============================================================
# Biology: Cell Populations
# =============================================================

class TestCellPopulations:
    @pytest.fixture
    def pop_model(self):
        model = CellPopulationModel()
        model.initialize((100,))
        model.populations['G1'][:] = 0.6
        model.populations['S'][:] = 0.2
        model.populations['G2'][:] = 0.15
        model.populations['M'][:] = 0.05
        return model

    def test_initial_fractions(self, pop_model):
        metrics = pop_model.get_metrics()
        assert 0.55 <= metrics['g1_fraction'] <= 0.65
        assert 0.15 <= metrics['s_fraction'] <= 0.25
        assert 0.10 <= metrics['g2_fraction'] <= 0.20
        assert 0.03 <= metrics['m_fraction'] <= 0.07

    def test_cell_cycle_produces_growth(self, pop_model):
        """Under normal oxygen, cells should multiply."""
        initial_total = float(np.sum(pop_model.get_total_density()))
        oxygen = np.ones(100)
        for _ in range(100):
            pop_model.update(0.1, oxygen)
        final_total = float(np.sum(pop_model.get_total_density()))
        assert final_total > initial_total * 1.5

    def test_hypoxia_causes_quiescence(self, pop_model):
        """Low oxygen should push cells into quiescence."""
        hypoxic_oxygen = np.full(100, 0.05)
        pop_model.update(0.1, hypoxic_oxygen)
        metrics = pop_model.get_metrics()
        assert metrics['quiescent_fraction'] > 0.3

    def test_severe_hypoxia_causes_necrosis(self, pop_model):
        """Very low oxygen should cause necrosis."""
        severe_oxygen = np.full(100, 0.005)
        pop_model.update(0.1, severe_oxygen)
        metrics = pop_model.get_metrics()
        assert metrics['necrotic_fraction'] > 0.5

    def test_positivity(self, pop_model):
        """All populations should remain non-negative."""
        oxygen = np.ones(100)
        for _ in range(50):
            pop_model.update(0.1, oxygen)
            for phase in pop_model.PHASES:
                assert np.all(pop_model.populations[phase] >= 0)

    def test_metrics_keys(self, pop_model):
        """Check that metrics include per-phase fractions."""
        metrics = pop_model.get_metrics()
        for phase in ['g1', 's', 'g2', 'm', 'q', 'n']:
            assert f'{phase}_fraction' in metrics


# =============================================================
# Biology: Treatment Effects
# =============================================================

class TestTreatments:
    @pytest.fixture
    def populations(self):
        n = 50
        return {
            'G1': np.full(n, 0.6),
            'S': np.full(n, 0.2),
            'G2': np.full(n, 0.15),
            'M': np.full(n, 0.05),
            'Q': np.zeros(n),
            'N': np.zeros(n),
        }

    def test_radiation_kills_cells(self, populations):
        module = TreatmentModule()
        oxygen = np.ones(50)
        effects, metrics = module.apply_treatment(
            "radiation", populations, oxygen, dose=2.0
        )
        # All effects should be negative (cell killing)
        for phase in ['G1', 'S', 'G2', 'M']:
            assert np.all(effects[phase] <= 0)
        assert metrics['total_cells_killed'] > 0

    def test_radiation_phase_sensitivity(self, populations):
        """M phase should be more affected than S phase."""
        module = TreatmentModule()
        oxygen = np.ones(50)
        effects, _ = module.apply_treatment(
            "radiation", populations, oxygen, dose=2.0
        )
        # M kill fraction should be higher than S kill fraction
        m_kill_frac = np.mean(-effects['M'] / (populations['M'] + 1e-10))
        s_kill_frac = np.mean(-effects['S'] / (populations['S'] + 1e-10))
        assert m_kill_frac > s_kill_frac

    def test_chemo_kills_cells(self, populations):
        module = TreatmentModule()
        effects, metrics = module.apply_treatment(
            "chemo", populations, drug_amount=1.0
        )
        assert metrics['total_cells_killed'] > 0

    def test_immunotherapy(self, populations):
        module = TreatmentModule()
        immune = np.full(50, 0.5)
        effects, metrics = module.apply_treatment(
            "immunotherapy", populations, immune_density=immune
        )
        assert metrics['total_cells_killed'] > 0


# =============================================================
# Integration: Full TumorModel
# =============================================================

class TestTumorModel:
    @pytest.fixture
    def model(self):
        return TumorModel(
            domain_size=(10.0, 10.0),
            n_initial_points=100
        )

    def test_initialization(self, model):
        assert model.tumor_density is not None
        assert len(model.tumor_density) == len(model.mesh.points)
        assert np.all(model.tumor_density >= 0)

    def test_initial_cell_fractions(self, model):
        metrics = model.get_metrics()
        cp = metrics['cell_populations']
        assert 0.55 <= cp['g1_fraction'] <= 0.65
        assert 0.15 <= cp['s_fraction'] <= 0.25

    def test_update_runs(self, model):
        """Model update should run without errors."""
        model.update(0.1)
        assert np.all(model.tumor_density >= 0)
        assert np.all(model.oxygen >= 0)

    def test_carrying_capacity(self, model):
        """Density should never exceed carrying capacity."""
        for _ in range(50):
            model.update(0.1)
        assert np.all(
            model.tumor_density <= model.params.carrying_capacity + 1e-10
        )

    def test_positivity(self, model):
        """All fields should remain non-negative."""
        for _ in range(20):
            model.update(0.1)
            assert np.all(model.tumor_density >= 0)
            assert np.all(model.oxygen >= 0)
            for phase in model.cell_populations.PHASES:
                assert np.all(
                    model.cell_populations.populations[phase] >= 0
                )

    def test_radiation_reduces_mass(self, model):
        """Radiation should reduce tumor mass."""
        for _ in range(20):
            model.update(0.1)
        pre_mass = model.get_metrics()['tumor']['total_mass']
        model.apply_treatment("radiation", dose=2.0)
        post_mass = model.get_metrics()['tumor']['total_mass']
        assert post_mass < pre_mass

    def test_metrics_structure(self, model):
        """Check metrics dict has expected structure."""
        m = model.get_metrics()
        assert 'tumor' in m
        assert 'cell_populations' in m
        assert 'mesh' in m
        assert 'immune' in m
        assert 'treatment' in m
        assert 'total_mass' in m['tumor']
        assert 'g1_fraction' in m['cell_populations']

    def test_chemo_treatment(self, model):
        for _ in range(20):
            model.update(0.1)
        pre_mass = model.get_metrics()['tumor']['total_mass']
        model.apply_treatment("chemo", drug_amount=1.0)
        post_mass = model.get_metrics()['tumor']['total_mass']
        assert post_mass < pre_mass

    def test_multiple_timesteps(self, model):
        """Run for many timesteps without crashing."""
        for step in range(50):
            model.update(0.1)
        m = model.get_metrics()
        assert m['tumor']['total_mass'] > 0


# =============================================================
# 3D Extension: Mesh Handler
# =============================================================

class TestMeshHandler3D:
    def test_initialization_3d(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0, 10.0))
        mesh.initialize_points(500, distribution="halton")
        assert mesh.points.shape[1] == 3
        assert mesh.ndim == 3
        assert len(mesh.points) == 500

    def test_points_in_domain_3d(self):
        mesh = MeshHandler(domain_size=(5.0, 8.0, 6.0))
        mesh.initialize_points(500, distribution="halton")
        for d, L in enumerate([5.0, 8.0, 6.0]):
            assert np.all(mesh.points[:, d] >= 0)
            assert np.all(mesh.points[:, d] <= L)

    def test_neighbor_count_3d(self):
        """3D should default to more neighbors than 2D."""
        mesh = MeshHandler(domain_size=(10.0, 10.0, 10.0))
        mesh.initialize_points(500, distribution="halton")
        # Default 3D neighbor count should be ~40
        for nbrs in mesh.neighbor_lists:
            assert len(nbrs) >= 20

    def test_3d_distributions(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0, 10.0))
        for dist in ["halton", "random", "grid"]:
            mesh.initialize_points(100, distribution=dist)
            assert mesh.points.shape[1] == 3

    def test_refine_3d(self):
        mesh = MeshHandler(domain_size=(10.0, 10.0, 10.0),
                           min_spacing=0.1, max_spacing=1.0)
        mesh.initialize_points(100, distribution="halton")
        initial_count = len(mesh.points)
        indicator = np.ones(initial_count)
        n_added = mesh.refine_points(indicator, threshold=0.5)
        assert len(mesh.points) >= initial_count
        assert mesh.points.shape[1] == 3  # Still 3D after refinement


# =============================================================
# 3D Extension: RBF Solver
# =============================================================

class TestRBFSolver3D:
    def test_laplacian_of_quadratic_3d(self):
        """∇²(x² + y² + z²) = 6 everywhere — exact for poly_degree=2."""
        rbf = RBFSolver(epsilon=1.0, poly_degree=2)
        mesh = MeshHandler(domain_size=(10.0, 10.0, 10.0))
        mesh.initialize_points(1000, distribution="halton")

        f = (mesh.points[:, 0] ** 2 +
             mesh.points[:, 1] ** 2 +
             mesh.points[:, 2] ** 2)
        L = rbf.assemble_global_operator(
            mesh.points, mesh.neighbor_lists, "laplacian"
        )
        result = L @ f

        interior = (
            (mesh.points[:, 0] > 1) & (mesh.points[:, 0] < 9) &
            (mesh.points[:, 1] > 1) & (mesh.points[:, 1] < 9) &
            (mesh.points[:, 2] > 1) & (mesh.points[:, 2] < 9)
        )
        assert np.allclose(result[interior], 6.0, atol=1e-4)

    def test_gradient_z_of_linear_3d(self):
        """∂/∂z(5z + 1) = 5 everywhere."""
        rbf = RBFSolver(epsilon=1.0, poly_degree=2)
        mesh = MeshHandler(domain_size=(10.0, 10.0, 10.0))
        mesh.initialize_points(1000, distribution="halton")

        f = 5 * mesh.points[:, 2] + 1
        Gz = rbf.assemble_global_operator(
            mesh.points, mesh.neighbor_lists, "gradient_z"
        )
        result = Gz @ f

        interior = (
            (mesh.points[:, 0] > 1) & (mesh.points[:, 0] < 9) &
            (mesh.points[:, 1] > 1) & (mesh.points[:, 1] < 9) &
            (mesh.points[:, 2] > 1) & (mesh.points[:, 2] < 9)
        )
        assert np.allclose(result[interior], 5.0, atol=1e-3)

    def test_gradient_x_in_3d(self):
        """∂/∂x(3x + 2y + z) = 3 in 3D (cross-check x-gradient works in 3D)."""
        rbf = RBFSolver(epsilon=1.0, poly_degree=2)
        mesh = MeshHandler(domain_size=(10.0, 10.0, 10.0))
        mesh.initialize_points(1000, distribution="halton")

        f = 3 * mesh.points[:, 0] + 2 * mesh.points[:, 1] + mesh.points[:, 2]
        Gx = rbf.assemble_global_operator(
            mesh.points, mesh.neighbor_lists, "gradient_x"
        )
        result = Gx @ f

        interior = (
            (mesh.points[:, 0] > 1) & (mesh.points[:, 0] < 9) &
            (mesh.points[:, 1] > 1) & (mesh.points[:, 1] < 9) &
            (mesh.points[:, 2] > 1) & (mesh.points[:, 2] < 9)
        )
        assert np.allclose(result[interior], 3.0, atol=1e-3)

    def test_poly_matrix_shape_3d(self):
        """3D poly_degree=2 should produce 10-column matrix."""
        rbf = RBFSolver(epsilon=1.0, poly_degree=2)
        points_3d = np.random.rand(20, 3)
        P = rbf._build_poly_matrix(points_3d)
        assert P.shape == (20, 10)

    def test_poly_matrix_shape_2d_unchanged(self):
        """2D poly_degree=2 should still produce 6-column matrix."""
        rbf = RBFSolver(epsilon=1.0, poly_degree=2)
        points_2d = np.random.rand(20, 2)
        P = rbf._build_poly_matrix(points_2d)
        assert P.shape == (20, 6)


# =============================================================
# 3D Extension: Full TumorModel
# =============================================================

class TestTumorModel3D:
    @pytest.fixture
    def model_3d(self):
        return TumorModel(
            domain_size=(10.0, 10.0, 10.0),
            n_initial_points=200
        )

    def test_initialization_3d(self, model_3d):
        assert model_3d.ndim == 3
        assert model_3d.tumor_density is not None
        assert len(model_3d.tumor_density) == len(model_3d.mesh.points)
        assert np.all(model_3d.tumor_density >= 0)

    def test_update_runs_3d(self, model_3d):
        """Model update should run without errors in 3D."""
        model_3d.update(0.1)
        assert np.all(model_3d.tumor_density >= 0)
        assert np.all(model_3d.oxygen >= 0)

    def test_carrying_capacity_3d(self, model_3d):
        """Density should never exceed carrying capacity in 3D."""
        for _ in range(20):
            model_3d.update(0.1)
        assert np.all(
            model_3d.tumor_density <= model_3d.params.carrying_capacity + 1e-10
        )

    def test_spherical_symmetry_3d(self, model_3d):
        """Initial Gaussian should be approximately spherically symmetric."""
        center = np.array(model_3d.domain_size) / 2.0
        distances = np.linalg.norm(model_3d.mesh.points - center, axis=1)
        # Points close to center should have higher density than far points
        close = distances < 1.0
        far = (distances > 3.0) & (distances < 4.0)
        if np.any(close) and np.any(far):
            assert np.mean(model_3d.tumor_density[close]) > np.mean(
                model_3d.tumor_density[far]
            )

    def test_radiation_reduces_mass_3d(self, model_3d):
        """Radiation should reduce tumor mass in 3D."""
        for _ in range(10):
            model_3d.update(0.1)
        pre_mass = model_3d.get_metrics()['tumor']['total_mass']
        model_3d.apply_treatment("radiation", dose=2.0)
        post_mass = model_3d.get_metrics()['tumor']['total_mass']
        assert post_mass < pre_mass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
