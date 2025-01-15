#!/usr/bin/env python3
"""
Enhanced mesh handler with feature-based adaptivity and error estimation.
Builds on existing MeshHandler class to provide more sophisticated
mesh refinement for tumor simulations.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AdaptivityParameters:
    """Parameters controlling mesh adaptation."""
    # Feature detection thresholds
    gradient_threshold: float = 0.1    # For tumor boundary detection
    curvature_threshold: float = 0.2   # For sharp features
    oxygen_threshold: float = 0.15     # For hypoxic regions
    
    # Refinement control
    max_refinement_level: int = 5      # Maximum number of refinement levels
    coarsening_threshold: float = 0.01 # When to merge points
    min_point_spacing: float = 0.01    # Minimum allowed distance between points
    
    # Error estimation
    error_tolerance: float = 1e-3      # Maximum allowed error estimate
    
    def validate(self):
        """Validate parameter values."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Parameter {name} must be non-negative")

class EnhancedMeshHandler:
    """
    Enhanced mesh handling with feature detection and error estimation.
    Provides sophisticated point distribution for tumor simulations.
    """
    
    def __init__(self,
                 domain_size: Tuple[float, float],
                 params: Optional[AdaptivityParameters] = None):
        """
        Initialize enhanced mesh handler.
        
        Args:
            domain_size: Physical domain size (Lx, Ly)
            params: Adaptivity control parameters
        """
        self.domain_size = domain_size
        self.params = params or AdaptivityParameters()
        self.params.validate()
        
        # Core mesh data
        self.points = None
        self.neighbor_lists = None
        self.kdtree = None
        
        # Refinement tracking
        self.refinement_levels = None
        self.feature_indicators = None
        
    def detect_features(self,
                       tumor_density: np.ndarray,
                       oxygen_concentration: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect important features requiring mesh refinement.
        
        Args:
            tumor_density: Current tumor density field
            oxygen_concentration: Optional oxygen concentration field
            
        Returns:
            Feature indicator field (0-1 scale, higher means more refinement needed)
        """
        if self.points is None:
            raise ValueError("Points not initialized")
            
        # 1. Compute tumor boundary regions (high gradients)
        gradients = self._compute_field_gradients(tumor_density)
        gradient_indicator = np.clip(
            gradients / self.params.gradient_threshold, 0, 1
        )
        
        # 2. Detect regions of high curvature
        curvature = self._compute_field_curvature(tumor_density)
        curvature_indicator = np.clip(
            curvature / self.params.curvature_threshold, 0, 1
        )
        
        # 3. Include hypoxic regions if oxygen data available
        oxygen_indicator = np.zeros_like(tumor_density)
        if oxygen_concentration is not None:
            oxygen_indicator = np.clip(
                (self.params.oxygen_threshold - oxygen_concentration) /
                self.params.oxygen_threshold,
                0, 1
            )
        
        # Combine indicators (can adjust weights based on importance)
        feature_indicator = np.maximum.reduce([
            gradient_indicator,
            0.7 * curvature_indicator,  # Slightly lower weight
            0.5 * oxygen_indicator      # Lower priority
        ])
        
        self.feature_indicators = feature_indicator
        return feature_indicator
        
    def refine_mesh(self,
                   feature_indicator: np.ndarray,
                   tumor_density: np.ndarray,
                   error_estimates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Refine mesh based on features and error estimates.
        
        Args:
            feature_indicator: Indicator field showing where refinement is needed
            tumor_density: Current tumor density field
            error_estimates: Optional point-wise error estimates
            
        Returns:
            Updated point coordinates
        """
        if self.points is None:
            raise ValueError("Points not initialized")
            
        # Initialize refinement levels if needed
        if self.refinement_levels is None:
            self.refinement_levels = np.zeros(len(self.points))
            
        # 1. Identify regions needing refinement
        refine_mask = feature_indicator > self.params.error_tolerance
        
        # Don't refine if already at max level
        refine_mask &= (self.refinement_levels < self.params.max_refinement_level)
        
        # 2. Generate new points
        new_points = []
        new_levels = []
        
        for i in range(len(self.points)):
            if refine_mask[i]:
                # Generate points based on local feature orientation
                local_points = self._generate_oriented_points(
                    self.points[i],
                    tumor_density,
                    feature_indicator[i]
                )
                
                new_points.extend(local_points)
                new_levels.extend([
                    self.refinement_levels[i] + 1
                ] * len(local_points))
                
        # 3. Filter points to maintain minimum spacing
        if new_points:
            filtered_points = []
            filtered_levels = []
            
            for p, level in zip(new_points, new_levels):
                if self._check_point_spacing(p, filtered_points):
                    filtered_points.append(p)
                    filtered_levels.append(level)
                    
            if filtered_points:
                self.points = np.vstack((
                    self.points,
                    np.array(filtered_points)
                ))
                self.refinement_levels = np.concatenate((
                    self.refinement_levels,
                    np.array(filtered_levels)
                ))
                
        # 4. Update neighbor lists
        self._update_neighbor_lists()
        return self.points
        
    def _generate_oriented_points(self,
                                center: np.ndarray,
                                field: np.ndarray,
                                feature_strength: float) -> List[np.ndarray]:
        """
        Generate new points oriented along feature directions.
        
        Args:
            center: Center point coordinates
            field: Field used to determine orientation
            feature_strength: Local feature indicator value
            
        Returns:
            List of new point coordinates
        """
        # Compute local gradient to determine feature orientation
        grad = self._compute_local_gradient(center, field)
        
        if np.all(grad == 0):
            # If no clear orientation, use radial distribution
            return self._generate_radial_points(center, feature_strength)
            
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
            
        # Generate points along and perpendicular to gradient
        theta = np.arctan2(grad[1], grad[0])
        spacing = self.params.min_point_spacing * (
            1 + (1 - feature_strength)
        )
        
        points = []
        
        # Along gradient direction
        points.append(center + spacing * grad)
        points.append(center - spacing * grad)
        
        # Perpendicular direction
        perp = np.array([-grad[1], grad[0]])
        points.append(center + spacing * perp)
        points.append(center - spacing * perp)
        
        # Filter points within domain
        valid_points = []
        for p in points:
            if (0 <= p[0] <= self.domain_size[0] and
                0 <= p[1] <= self.domain_size[1]):
                valid_points.append(p)
                
        return valid_points
        
    def _generate_radial_points(self,
                              center: np.ndarray,
                              feature_strength: float) -> List[np.ndarray]:
        """Generate points in a radial pattern."""
        spacing = self.params.min_point_spacing * (
            1 + (1 - feature_strength)
        )
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        
        points = []
        for theta in angles:
            offset = spacing * np.array([np.cos(theta), np.sin(theta)])
            point = center + offset
            
            if (0 <= point[0] <= self.domain_size[0] and
                0 <= point[1] <= self.domain_size[1]):
                points.append(point)
                
        return points
        
    def _compute_field_gradients(self, field: np.ndarray) -> np.ndarray:
        """Compute magnitude of field gradients."""
        if self.neighbor_lists is None:
            return np.zeros_like(field)
            
        gradients = np.zeros_like(field)
        
        for i, nbrs in enumerate(self.neighbor_lists):
            if len(nbrs) > 1:
                # Compute local gradients using neighbors
                dx = self.points[nbrs] - self.points[i]
                df = field[nbrs] - field[i]
                
                # Least squares fit for gradient
                try:
                    grad = np.linalg.lstsq(dx, df, rcond=None)[0]
                    gradients[i] = np.linalg.norm(grad)
                except np.linalg.LinAlgError:
                    continue
                    
        return gradients
        
    def _compute_field_curvature(self, field: np.ndarray) -> np.ndarray:
        """Compute field curvature."""
        if self.neighbor_lists is None:
            return np.zeros_like(field)
            
        curvature = np.zeros_like(field)
        
        for i, nbrs in enumerate(self.neighbor_lists):
            if len(nbrs) > 2:
                # Compute local quadratic fit
                dx = self.points[nbrs] - self.points[i]
                df = field[nbrs] - field[i]
                
                # Build quadratic terms
                A = np.column_stack([
                    dx[:,0]**2, dx[:,0]*dx[:,1], dx[:,1]**2,
                    dx[:,0], dx[:,1], np.ones_like(dx[:,0])
                ])
                
                try:
                    # Solve for quadratic coefficients
                    coeffs = np.linalg.lstsq(A, df, rcond=None)[0]
                    # Approximate curvature from quadratic terms
                    curvature[i] = np.abs(coeffs[0]) + np.abs(coeffs[2])
                except np.linalg.LinAlgError:
                    continue
                    
        return curvature
        
    def _check_point_spacing(self,
                           point: np.ndarray,
                           existing_points: List[np.ndarray]) -> bool:
        """Check if point maintains minimum spacing."""
        if not existing_points:
            return True
            
        existing = np.array(existing_points)
        distances = np.linalg.norm(existing - point, axis=1)
        return np.all(distances >= self.params.min_point_spacing)
        
    def _update_neighbor_lists(self, radius_factor: float = 2.5):
        """Update neighbor lists for all points."""
        self.kdtree = cKDTree(self.points)
        radius = radius_factor * self.params.min_point_spacing
        
        self.neighbor_lists = [
            self.kdtree.query_ball_point(p, radius)
            for p in self.points
        ]
        
    def get_refinement_metrics(self) -> Dict:
        """Calculate mesh refinement metrics."""
        if self.points is None:
            raise ValueError("Points not initialized")
            
        return {
            'n_points': len(self.points),
            'mean_refinement_level': float(np.mean(self.refinement_levels)),
            'max_refinement_level': int(np.max(self.refinement_levels)),
            'mean_feature_indicator': float(np.mean(self.feature_indicators)),
            'refined_fraction': float(np.mean(self.refinement_levels > 0))
        }