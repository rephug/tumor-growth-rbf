"""
tests/test_tumor_cell_populations.py

Comprehensive test suite for tumor cell population dynamics.
Tests both the CellPopulationModel and its integration with TumorModel.
"""

import numpy as np
import pytest
from tumor_growth_rbf.biology.cell_populations import CellPopulationModel, CellCycleParameters
from tumor_growth_rbf.biology.tumor_model import TumorModel, TumorParameters

def test_initial_cell_distribution():
    """Test that initial cell distribution matches expected cell cycle proportions."""
    domain_size = (5.0, 5.0)
    model = TumorModel(domain_size, n_initial_points=100)
    
    # Get initial metrics
    metrics = model.get_metrics()
    cell_metrics = metrics['cell_populations']
    
    # Check proportions are roughly as expected
    assert 0.55 <= cell_metrics['g1_fraction'] <= 0.65  # ~60% in G1
    assert 0.15 <= cell_metrics['s_fraction'] <= 0.25   # ~20% in S
    assert 0.10 <= cell_metrics['g2_fraction'] <= 0.20  # ~15% in G2
    assert 0.03 <= cell_metrics['m_fraction'] <= 0.07   # ~5% in M

def test_oxygen_dependent_transitions():
    """Test cell state transitions under different oxygen conditions."""
    domain_size = (5.0, 5.0)
    model = TumorModel(domain_size, n_initial_points=100)
    
    # Create hypoxic conditions
    model.oxygen[:] = 0.05  # Below hypoxia threshold
    
    # Run for several time steps
    dt = 0.1
    for _ in range(10):
        model.update(dt)
    
    metrics = model.get_metrics()
    
    # Under hypoxia, expect increased quiescence and some necrosis
    assert metrics['cell_populations']['quiescent_fraction'] > 0.3
    assert metrics['cell_populations']['necrotic_fraction'] > 0

def test_cell_cycle_progression():
    """Test normal cell cycle progression under optimal conditions."""
    domain_size = (5.0, 5.0)
    model = TumorModel(domain_size, n_initial_points=100)
    
    # Set optimal conditions
    model.oxygen[:] = 1.0
    
    # Track initial cell numbers
    initial_metrics = model.get_metrics()
    initial_total = initial_metrics['tumor']['total_mass']
    
    # Run for one approximate cell cycle
    dt = 0.1
    for _ in range(int(24/dt)):  # ~24 hour cell cycle
        model.update(dt)
    
    final_metrics = model.get_metrics()
    final_total = final_metrics['tumor']['total_mass']
    
    # Should see population growth
    assert final_total > initial_total * 1.5  # At least 50% increase

def test_treatment_response():
    """Test cell population response to radiation treatment."""
    domain_size = (5.0, 5.0)
    model = TumorModel(domain_size, n_initial_points=100)
    
    # Get pre-treatment metrics
    pre_metrics = model.get_metrics()
    pre_total = pre_metrics['tumor']['total_mass']
    
    # Apply radiation treatment
    model.apply_treatment("radiation", dose=2.0)
    
    # Get post-treatment metrics
    post_metrics = model.get_metrics()
    post_total = post_metrics['tumor']['total_mass']
    
    # Expect significant cell death
    assert post_total < pre_total * 0.8  # At least 20% reduction
    
    # Cells in M and G2 phases should be more affected
    g2m_ratio_post = (post_metrics['cell_populations']['g2_fraction'] + 
                      post_metrics['cell_populations']['m_fraction'])
    g2m_ratio_pre = (pre_metrics['cell_populations']['g2_fraction'] + 
                     pre_metrics['cell_populations']['m_fraction'])
    assert g2m_ratio_post < g2m_ratio_pre  # More sensitive phases affected more

def test_mass_conservation():
    """Test that total cell mass is conserved except for division/death."""
    domain_size = (5.0, 5.0)
    model = TumorModel(domain_size, n_initial_points=100)
    
    def get_total_cells():
        metrics = model.get_metrics()
        return metrics['tumor']['total_mass']
    
    initial_total = get_total_cells()
    
    # Single small timestep (too short for division)
    dt = 0.01
    model.update(dt)
    
    # Mass should be very close to initial (small numerical differences ok)
    assert abs(get_total_cells() - initial_total) < 1e-10

def test_prolonged_hypoxia_necrosis():
    """Test that prolonged hypoxia leads to necrosis."""
    domain_size = (5.0, 5.0)
    model = TumorModel(domain_size, n_initial_points=100)
    
    # Create severe hypoxia
    model.oxygen[:] = 0.01  # Severe hypoxia
    
    # Run for extended period
    dt = 0.1
    for _ in range(100):  # 10 time units
        model.update(dt)
    
    metrics = model.get_metrics()
    
    # Expect significant necrosis
    assert metrics['cell_populations']['necrotic_fraction'] > 0.5

def test_oxygen_consumption():
    """Test that proliferating cells consume more oxygen."""
    domain_size = (5.0, 5.0)
    model = TumorModel(domain_size, n_initial_points=100)
    
    # Record initial oxygen
    initial_oxygen = model.oxygen.copy()
    
    # Run for several time steps
    dt = 0.1
    for _ in range(10):
        model.update(dt)
    
    # Oxygen should be depleted where there are proliferating cells
    assert np.any(model.oxygen < initial_oxygen)
    
    # Areas with more proliferating cells should have lower oxygen
    high_proliferation = (model.cell_populations.populations['S'] + 
                         model.cell_populations.populations['G2'] + 
                         model.cell_populations.populations['M']) > 0.1
    
    if np.any(high_proliferation):
        mean_oxygen_proliferating = np.mean(model.oxygen[high_proliferation])
        mean_oxygen_other = np.mean(model.oxygen[~high_proliferation])
        assert mean_oxygen_proliferating < mean_oxygen_other

if __name__ == "__main__":
    pytest.main([__file__])