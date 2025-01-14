#!/usr/bin/env python3
"""
validation.py

Validation utilities for tumor growth simulations.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Metrics for comparing simulation with experimental data."""
    rmse: float  # Root mean square error
    r_squared: float  # R-squared value
    correlation: float  # Pearson correlation coefficient
    peak_error: float  # Maximum absolute error
    relative_error: float  # Mean relative error
    
class ModelValidator:
    """
    Validates tumor growth simulations against experimental data
    and theoretical predictions.
    """
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_growth_curve(self,
                            sim_times: np.ndarray,
                            sim_masses: np.ndarray,
                            exp_times: np.ndarray,
                            exp_masses: np.ndarray) -> ValidationMetrics:
        """
        Validate simulated growth curve against experimental data.
        
        Args:
            sim_times: Simulation time points
            sim_masses: Simulated tumor masses
            exp_times: Experimental time points
            exp_masses: Experimental tumor masses
            
        Returns:
            ValidationMetrics object with comparison metrics
        """
        # Interpolate simulation data to experimental time points
        sim_interp = np.interp(exp_times, sim_times, sim_masses)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((sim_interp - exp_masses)**2))
        
        # R-squared
        ss_res = np.sum((exp_masses - sim_interp)**2)
        ss_tot = np.sum((exp_masses - np.mean(exp_masses))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Correlation
        correlation = stats.pearsonr(sim_interp, exp_masses)[0]
        
        # Peak error
        peak_error = np.max(np.abs(sim_interp - exp_masses))
        
        # Relative error
        relative_error = np.mean(np.abs(sim_interp - exp_masses) / exp_masses)
        
        metrics = ValidationMetrics(
            rmse=rmse,
            r_squared=r_squared,
            correlation=correlation,
            peak_error=peak_error,
            relative_error=relative_error
        )
        
        return metrics
        
    def validate_theoretical_predictions(self,
                                      model,
                                      predictions: Dict[str, callable],
                                      time_points: np.ndarray) -> Dict[str, ValidationMetrics]:
        """
        Validate simulation against theoretical predictions.
        
        Args:
            model: TumorModel instance
            predictions: Dictionary of theoretical prediction functions
            time_points: Time points for validation
            
        Returns:
            Dictionary of ValidationMetrics for each prediction
        """
        results = {}
        
        for name, pred_func in predictions.items():
            # Get theoretical predictions
            pred_values = pred_func(time_points)
            
            # Run simulation
            sim_values = []
            model.reset()  # Reset model to initial state
            
            for t in time_points:
                model.update(t)
                metrics = model.get_metrics()
                sim_values.append(metrics['total_mass'])
                
            # Validate
            metrics = self.validate_growth_curve(
                time_points, np.array(sim_values),
                time_points, pred_values
            )
            results[name] = metrics
            
        return results
        
    def analyze_parameter_sensitivity(self,
                                   model,
                                   parameter_ranges: Dict[str, Tuple[float, float]],
                                   n_samples: int = 10) -> Dict[str, np.ndarray]:
        """
        Analyze model sensitivity to parameter variations.
        
        Args:
            model: TumorModel instance
            parameter_ranges: Dictionary of parameter ranges to test
            n_samples: Number of samples per parameter
            
        Returns:
            Dictionary of sensitivity coefficients for each parameter
        """
        sensitivities = {}
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            # Create parameter samples
            param_values = np.linspace(min_val, max_val, n_samples)
            responses = []
            
            # Test each parameter value
            original_value = getattr(model.params, param_name)
            
            for value in param_values:
                setattr(model.params, param_name, value)
                model.reset()
                
                # Run simulation
                for _ in range(10):  # Simulate for 10 time steps
                    model.update(0.1)
                    
                metrics = model.get_metrics()
                responses.append(metrics['total_mass'])
                
            # Restore original value
            setattr(model.params, param_name, original_value)
            
            # Calculate sensitivity coefficient
            param_range = max_val - min_val
            response_range = max(responses) - min(responses)
            sensitivity = response_range / param_range
            
            sensitivities[param_name] = sensitivity
            
        return sensitivities
        
    def generate_validation_report(self) -> str:
        """Generate a text report of validation results."""
        report = []
        report.append("Validation Report")
        report.append("=" * 50)
        
        for test_name, metrics in self.validation_results.items():
            report.append(f"\nTest: {test_name}")
            report.append("-" * 30)
            
            if isinstance(metrics, ValidationMetrics):
                report.append(f"RMSE: {metrics.rmse:.4f}")
                report.append(f"R-squared: {metrics.r_squared:.4f}")
                report.append(f"Correlation: {metrics.correlation:.4f}")
                report.append(f"Peak Error: {metrics.peak_error:.4f}")
                report.append(f"Relative Error: {metrics.relative_error:.4f}")
            else:
                report.append(str(metrics))
                
        return "\n".join(report)
    
    class ValidationSuite:
    """
    Comprehensive validation suite for tumor growth model.
    Includes unit tests, integration tests, and parameter validation.
    """
    
    def __init__(self, model):
        self.model = model
        self.validator = ModelValidator()
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all validation tests."""
        results = {}
        
        # Conservation tests
        results['mass_conservation'] = self.test_mass_conservation()
        results['positivity'] = self.test_positivity()
        
        # Parameter tests
        results['parameter_bounds'] = self.test_parameter_bounds()
        results['parameter_sensitivity'] = self.test_parameter_sensitivity()
        
        # Growth tests
        results['growth_bounds'] = self.test_growth_bounds()
        results['oxygen_coupling'] = self.test_oxygen_coupling()
        
        return results
        
    def test_mass_conservation(self) -> bool:
        """Test conservation of mass during diffusion."""
        # Save initial state
        initial_state = self.model.tumor_density.copy()
        initial_mass = np.sum(initial_state)
        
        # Simulate diffusion only
        old_growth_rate = self.model.params.growth_rate
        self.model.params.growth_rate = 0.0
        self.model.update(dt=0.1)
        self.model.params.growth_rate = old_growth_rate
        
        # Check mass conservation
        final_mass = np.sum(self.model.tumor_density)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        
        return relative_error < 1e-10
        
    def test_positivity(self) -> bool:
        """Test that density remains non-negative."""
        self.model.update(dt=0.1)
        return np.all(self.model.tumor_density >= 0)
        
    def test_parameter_bounds(self) -> bool:
        """Test parameter bounds and constraints."""
        try:
            self.model.params.validate()
            return True
        except ValueError:
            return False
            
    def test_parameter_sensitivity(self) -> bool:
        """Test sensitivity to parameter variations."""
        parameter_ranges = {
            'growth_rate': (0.05, 0.2),
            'diffusion_white': (0.05, 0.2),
            'oxygen_consumption': (0.05, 0.2)
        }
        
        sensitivities = self.validator.analyze_parameter_sensitivity(
            self.model, parameter_ranges
        )
        
        # Check that model responds to parameter changes
        return all(s > 0 for s in sensitivities.values())
        
    def test_growth_bounds(self) -> bool:
        """Test that growth remains bounded."""
        initial_max = np.max(self.model.tumor_density)
        
        for _ in range(10):
            self.model.update(dt=0.1)
            
        final_max = np.max(self.model.tumor_density)
        
        return (final_max <= self.model.params.carrying_capacity * 1.001 and
                final_max >= 0)
                
    def test_oxygen_coupling(self) -> bool:
        """Test oxygen-growth coupling."""
        # Create hypoxic region
        self.model.oxygen *= 0.5
        initial_growth = self.model._compute_growth()
        
        # Restore oxygen
        self.model.oxygen *= 2
        final_growth = self.model._compute_growth()
        
        # Growth should be higher with more oxygen
        return np.mean(final_growth) > np.mean(initial_growth)
        
def validate_simulation_setup(model,
                            domain_size: Tuple[float, float],
                            dt: float,
                            n_points: int) -> List[str]:
    """
    Validate simulation setup parameters.
    
    Args:
        model: TumorModel instance
        domain_size: Physical domain size
        dt: Time step size
        n_points: Number of spatial points
        
    Returns:
        List of warning messages, if any
    """
    warnings = []
    
    # Check CFL condition for diffusion
    dx = domain_size[0] / n_points
    D = model.params.diffusion_white  # use larger diffusion coefficient
    cfl = D * dt / (dx * dx)
    
    if cfl > 0.5:
        warnings.append(
            f"CFL condition might be violated: {cfl:.3f} > 0.5. "
            f"Consider reducing dt or increasing n_points."
        )
        
    # Check spatial resolution
    min_feature_size = min(domain_size) / 20  # assume we want to resolve features 1/20th of domain
    if dx > min_feature_size:
        warnings.append(
            f"Spatial resolution dx={dx:.3f} might be too coarse to resolve "
            f"features of size {min_feature_size:.3f}."
        )
        
    # Check if domain is large enough
    tumor_radius = np.sqrt(np.sum(model.tumor_density > 0.01) * dx * dx / np.pi)
    if tumor_radius > 0.25 * min(domain_size):
        warnings.append(
            f"Initial tumor radius {tumor_radius:.3f} might be too large "
            f"relative to domain size {domain_size}."
        )
        
    return warnings