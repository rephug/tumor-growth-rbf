"""
test_tumor_model.py

Comprehensive test suite for the enhanced tumor growth model.
Tests cover biological accuracy, numerical stability, and integration
of tissue heterogeneity with cell population dynamics.
"""

import numpy as np
import pytest
from pathlib import Path
import matplotlib.pyplot as plt

from tumor_growth_rbf.biology.tumor_model import TumorModel, TumorParameters
from tumor_growth_rbf.biology.tissue_properties import TissueType, TissueParameters
from tumor_growth_rbf.biology.cell_populations import CellCycleParameters

class TestTumorGrowthBiology:
    """Test biological accuracy of tumor growth patterns."""

    @pytest.fixture
    def base_model(self):
        """Create a basic tumor model for testing."""
        domain_size = (10.0, 10.0)
        return TumorModel(domain_size, n_initial_points=50)

    @pytest.fixture
    def tissue_aware_model(self):
        """Create a model with tissue heterogeneity."""
        domain_size = (10.0, 10.0)
        tissue_params = TissueParameters()
        model = TumorModel(domain_size, tissue_params=tissue_params,
                          n_initial_points=50)
        
        # Create simple tissue map
        tissue_image = np.ones((50, 50))
        tissue_image[:, :25] = 0  # Left half white matter, right half gray matter
        tissue_labels = {
            0: TissueType.WHITE_MATTER,
            1: TissueType.GRAY_MATTER
        }
        model.load_tissue_data(tissue_image, tissue_labels)
        return model

    def test_growth_in_carrying_capacity(self, base_model):
        """Test that tumor growth respects carrying capacity."""
        # Run simulation for several time steps
        dt = 0.1
        for _ in range(100):
            base_model.update(dt)
            
        # Check that density never exceeds carrying capacity
        assert np.all(base_model.tumor_density <= base_model.params.carrying_capacity)
        
        # Check that growth slows as we approach carrying capacity
        high_density = base_model.tumor_density > 0.9 * base_model.params.carrying_capacity
        if np.any(high_density):
            growth = base_model._compute_growth()
            assert np.mean(growth[high_density]) < 0.1 * base_model.params.growth_rate

    def test_cell_cycle_progression(self, base_model):
        """Test realistic cell cycle progression."""
        # Initial cell distribution should match typical tumor profiles
        metrics = base_model.get_metrics()
        pop_metrics = metrics['cell_populations']
        
        # Check initial distribution
        assert 0.55 <= pop_metrics['g1_fraction'] <= 0.65  # ~60% in G1
        assert 0.15 <= pop_metrics['s_fraction'] <= 0.25   # ~20% in S
        assert 0.10 <= pop_metrics['g2_fraction'] <= 0.20  # ~15% in G2
        assert 0.03 <= pop_metrics['m_fraction'] <= 0.07   # ~5% in M
        
        # Run for one approximate cell cycle
        dt = 0.1
        initial_g1 = base_model.cell_populations.populations['G1'].copy()
        
        for _ in range(int(24/dt)):  # ~24 hour cell cycle
            base_model.update(dt)
            
        # Verify cell cycle completion effects
        final_metrics = base_model.get_metrics()
        assert final_metrics['tumor']['total_mass'] > metrics['tumor']['total_mass']

    def test_tissue_dependent_growth(self, tissue_aware_model):
        """Test tissue-specific growth patterns."""
        # Run simulation
        dt = 0.1
        for _ in range(50):
            tissue_aware_model.update(dt)
            
        # Get tumor density in different tissue regions
        tissue_map = tissue_aware_model.tissue_model.tissue_map
        white_matter_mask = tissue_map == TissueType.WHITE_MATTER
        gray_matter_mask = tissue_map == TissueType.GRAY_MATTER
        
        white_matter_density = tissue_aware_model.tumor_density[white_matter_mask]
        gray_matter_density = tissue_aware_model.tumor_density[gray_matter_mask]
        
        # Verify faster growth in white matter
        assert np.mean(white_matter_density) > np.mean(gray_matter_density)

    def test_hypoxia_response(self, base_model):
        """Test cellular response to hypoxic conditions."""
        # Create hypoxic conditions
        base_model.oxygen[:] = 0.05  # Below hypoxia threshold
        
        # Run simulation
        dt = 0.1
        for _ in range(50):
            base_model.update(dt)
            
        # Check for appropriate biological responses
        metrics = base_model.get_metrics()
        pop_metrics = metrics['cell_populations']
        
        # Expect increased quiescence and some necrosis
        assert pop_metrics['quiescent_fraction'] > 0.3
        assert pop_metrics['necrotic_fraction'] > 0

class TestTreatmentResponse:
    """Test treatment effects and tissue-specific responses."""

    @pytest.fixture
    def treated_model(self):
        """Create model for treatment testing."""
        domain_size = (10.0, 10.0)
        return TumorModel(domain_size, n_initial_points=50)

    def test_radiation_response(self, treated_model):
        """Test radiation therapy effects."""
        # Record pre-treatment state
        pre_metrics = treated_model.get_metrics()
        pre_total = pre_metrics['tumor']['total_mass']
        
        # Apply radiation treatment
        treated_model.apply_treatment("radiation", dose=2.0)
        
        # Verify treatment effects
        post_metrics = treated_model.get_metrics()
        post_total = post_metrics['tumor']['total_mass']
        
        # Check for cell kill
        assert post_total < pre_total
        
        # Check phase-specific sensitivity
        g2m_ratio_post = (post_metrics['cell_populations']['g2_fraction'] + 
                         post_metrics['cell_populations']['m_fraction'])
        g2m_ratio_pre = (pre_metrics['cell_populations']['g2_fraction'] + 
                        pre_metrics['cell_populations']['m_fraction'])
        
        # G2/M phases should be more affected
        assert g2m_ratio_post < g2m_ratio_pre

    def test_chemotherapy_response(self, treated_model):
        """Test chemotherapy effects."""
        # Apply chemotherapy
        treated_model.apply_treatment("chemo", drug_amount=1.0, duration=1.0)
        
        # Run for drug effect period
        dt = 0.1
        metrics_history = []
        
        for _ in range(20):
            treated_model.update(dt)
            metrics_history.append(treated_model.get_metrics())
            
        # Verify S-phase specific effects
        initial_s = metrics_history[0]['cell_populations']['s_fraction']
        final_s = metrics_history[-1]['cell_populations']['s_fraction']
        
        # S-phase should be more affected
        assert final_s < initial_s

class TestNumericalProperties:
    """Test numerical stability and conservation properties."""

    @pytest.fixture
    def numerical_model(self):
        """Create model for numerical testing."""
        domain_size = (10.0, 10.0)
        return TumorModel(domain_size, n_initial_points=100)

    def test_mass_conservation(self, numerical_model):
        """Test conservation of mass during transport."""
        initial_mass = np.sum(numerical_model.tumor_density)
        
        # Run diffusion only (no growth)
        old_growth_rate = numerical_model.params.growth_rate
        numerical_model.params.growth_rate = 0.0
        
        dt = 0.1
        numerical_model.update(dt)
        
        # Restore growth rate
        numerical_model.params.growth_rate = old_growth_rate
        
        final_mass = np.sum(numerical_model.tumor_density)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        
        assert relative_error < 1e-10

    def test_positivity_preservation(self, numerical_model):
        """Test that density remains non-negative."""
        dt = 0.1
        for _ in range(100):
            numerical_model.update(dt)
            assert np.all(numerical_model.tumor_density >= 0)
            assert np.all(numerical_model.oxygen >= 0)
            
            for pop_name, population in numerical_model.cell_populations.populations.items():
                assert np.all(population >= 0)

    def test_mesh_adaptation(self, numerical_model):
        """Test mesh refinement behavior."""
        initial_points = len(numerical_model.mesh.points)
        
        # Run simulation with growth
        dt = 0.1
        for _ in range(50):
            numerical_model.update(dt)
            
        # Verify mesh adaptation occurs
        final_points = len(numerical_model.mesh.points)
        assert final_points != initial_points
        
        # Check that refinement occurs near tumor boundary
        gradient = numerical_model._compute_gradient_magnitude()
        high_gradient = gradient > np.mean(gradient)
        
        # Should have more points in high gradient regions
        point_density = np.zeros_like(numerical_model.tumor_density)
        points = numerical_model.mesh.points
        for p in points:
            i, j = int(p[0]), int(p[1])
            point_density[i, j] += 1
            
        assert np.mean(point_density[high_gradient]) > np.mean(point_density[~high_gradient])

def test_visualization(base_model):
    """Test visualization capabilities."""
    from tumor_growth_rbf.utils.visualization import TumorVisualizer
    
    visualizer = TumorVisualizer(base_model)
    
    # Run simulation and collect states
    dt = 0.1
    times = np.arange(0, 1, dt)
    states = []
    metrics = []
    
    for t in times:
        base_model.update(dt)
        states.append({
            'tumor_density': base_model.tumor_density.copy(),
            'oxygen': base_model.oxygen.copy()
        })
        metrics.append(base_model.get_metrics())
    
    # Create and verify all plot types
    state_fig = visualizer.create_state_visualization(times[-1])
    assert isinstance(state_fig, plt.Figure)
    
    population_fig = visualizer.plot_cell_populations()
    assert isinstance(population_fig, plt.Figure)
    
    metrics_fig = visualizer.plot_metrics_history(times, metrics)
    assert isinstance(metrics_fig, plt.Figure)

if __name__ == '__main__':
    pytest.main([__file__])
