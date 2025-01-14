"""
Tumor Growth RBF Simulator

A high-performance, meshless tumor growth simulation framework using
Radial Basis Function-generated Finite Differences (RBF-FD).
"""

from .biology.tumor_model import TumorModel, TumorParameters
from .core.rbf_solver import RBFSolver
from .core.pde_assembler import PDEAssembler
from .utils.visualization import TumorVisualizer
from .utils.validation import ModelValidator, ValidationSuite
from .utils.optimization import PerformanceOptimizer, AdaptiveTimestepper

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Make key classes available at package level
__all__ = [
    'TumorModel',
    'TumorParameters',
    'RBFSolver',
    'PDEAssembler',
    'TumorVisualizer',
    'ModelValidator',
    'ValidationSuite',
    'PerformanceOptimizer',
    'AdaptiveTimestepper'
]