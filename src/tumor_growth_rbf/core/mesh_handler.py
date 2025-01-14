#!/usr/bin/env python3
"""
mesh_handler.py

Handles point distribution and refinement for meshless RBF-FD method.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MeshHandler:
    """
    Handles point distribution and adaptive refinement for RBF-FD method.
    """
    
    def __init__(self,
                 domain_size: Tuple[float, float],
                 min_spacing: float = 0.01,
                 max_spacing: float = 0.1):
        """
        Initialize mesh handler.
        
        Args:
            domain_size: Physical domain size (Lx, Ly)
            min_spacing: Minimum allowed point spacing
            max_spacing: Maximum allowed point spacing
        """
        self.domain_size = domain_size
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        
        self.points = None
        self.neighbor_lists = None
        self.kdtree = None
        
    def initialize_points(self,
                        n_points: int,
                        distribution: str = "uniform") -> np.ndarray:
        """
        Initialize point distribution.
        
        Args:
            n_points: Number of points to generate
            distribution: Type of distribution ("uniform", "random", or "halton")
            
        Returns:
            Array of point coordinates
        """
        if distribution == "uniform":
            points = self._create_uniform_points(n_points)
        elif distribution == "random":
            points = self._create_random_points(n_points)
        elif distribution == "halton":
            points = self._create_halton_points(n_points)
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")
            
        self.points = points
        self._update_neighbor_lists()
        
        return self.points
        
    def _create_uniform_points(self, n_points: int) -> np.ndarray:
        """Create uniform grid points."""
        nx = ny = int(np.sqrt(n_points))
        x = np.linspace(0, self.domain_size[0], nx)
        y = np.linspace(0, self.domain_size[1], ny)
        X, Y = np.meshgrid(x, y)
        return np.column_stack((X.ravel(), Y.ravel()))
        
    def _create_random_points(self, n_points: int) -> np.ndarray:
        """Create random points with minimum spacing."""
        points = []
        while len(points) < n_points:
            point = np.random.rand(2) * self.domain_size
            
            if not points or self._check_spacing(point, points):
                points.append(point)
                
        return np.array(points)
        
    def _create_halton_points(self, n_points: int) -> np.ndarray:
        """Create points using Halton sequence for better uniformity."""
        def halton(index: int, base: int) -> float:
            """Generate Halton sequence value."""
            f = 1
            result = 0
            while index > 0:
                f = f / base
                result = result + f * (index % base)
                index = index // base
            return result
            
        points = np.zeros((n_points, 2))
        for i in range(n_points):
            points[i] = [
                halton(i, 2) * self.domain_size[0],
                halton(i, 3) * self.domain_size[1]
            ]
            
        return points
        
    def _check_spacing(self, 
                      point: np.ndarray,
                      existing_points: List[np.ndarray]) -> bool:
        """Check if point maintains minimum spacing with existing points."""
        if not existing_points:
            return True
            
        existing = np.array(existing_points)
        distances = np.linalg.norm(existing - point, axis=1)
        return np.all(distances >= self.min_spacing)
        
    def _update_neighbor_lists(self, radius_factor: float = 2.5):
        """Update neighbor lists for all points."""
        self.kdtree = cKDTree(self.points)
        
        # Use maximum spacing to determine neighbor search radius
        radius = radius_factor * self.max_spacing
        
        self.neighbor_lists = [
            self.kdtree.query_ball_point(p, radius)
            for p in self.points
        ]
        
    def refine_points(self,
                     refinement_indicator: np.ndarray,
                     threshold: float = 0.1) -> np.ndarray:
        """
        Refine point distribution based on error indicator.
        
        Args:
            refinement_indicator: Error or refinement indicator field
            threshold: Refinement threshold
            
        Returns:
            Updated point coordinates
        """
        if self.points is None:
            raise ValueError("Points not initialized")
            
        # Identify regions needing refinement
        refine_mask = refinement_indicator > threshold
        
        # Add points in refinement regions
        new_points = []
        for i in range(len(self.points)):
            if refine_mask[i]:
                # Add points around high error point
                new_points.extend(
                    self._generate_refinement_points(self.points[i])
                )
                
        if new_points:
            # Add new points while maintaining minimum spacing
            filtered_points = []
            for p in new_points:
                if self._check_spacing(p, list(self.points) + filtered_points):
                    filtered_points.append(p)
                    
            if filtered_points:
                self.points = np.vstack((self.points, np.array(filtered_points)))
                self._update_neighbor_lists()
                
        return self.points
        
    def _generate_refinement_points(self,
                                  center: np.ndarray,
                                  n_points: int = 4) -> List[np.ndarray]:
        """Generate new points around a center point."""
        new_points = []
        spacing = self.min_spacing
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        for theta in angles:
            point = center + spacing * np.array([np.cos(theta), np.sin(theta)])
            
            # Check domain bounds
            if (0 <= point[0] <= self.domain_size[0] and
                0 <= point[1] <= self.domain_size[1]):
                new_points.append(point)
                
        return new_points
        
    def coarsen_points(self,
                      coarsening_indicator: np.ndarray,
                      threshold: float = 0.01) -> np.ndarray:
        """
        Coarsen point distribution in regions of low error.
        
        Args:
            coarsening_indicator: Error or coarsening indicator field
            threshold: Coarsening threshold
            
        Returns:
            Updated point coordinates
        """
        if self.points is None:
            raise ValueError("Points not initialized")
            
        # Identify points that can be removed
        coarsen_mask = coarsening_indicator < threshold
        
        if np.any(coarsen_mask):
            # Keep points with low error density
            self.points = self.points[~coarsen_mask]
            self._update_neighbor_lists()
            
        return self.points
        
    def get_boundary_points(self) -> np.ndarray:
        """Identify boundary points."""
        if self.points is None:
            raise ValueError("Points not initialized")
            
        tol = 1e-10  # Tolerance for boundary detection
        
        is_boundary = (
            (np.abs(self.points[:,0]) < tol) |
            (np.abs(self.points[:,0] - self.domain_size[0]) < tol) |
            (np.abs(self.points[:,1]) < tol) |
            (np.abs(self.points[:,1] - self.domain_size[1]) < tol)
        )
        
        return self.points[is_boundary]
        
    def get_metrics(self) -> dict:
        """Calculate mesh quality metrics."""
        if self.points is None:
            raise ValueError("Points not initialized")
            
        # Calculate nearest neighbor distances
        distances, _ = self.kdtree.query(self.points, k=2)
        min_distances = distances[:,1]  # Exclude self-distance
        
        return {
            'n_points': len(self.points),
            'min_spacing': float(np.min(min_distances)),
            'max_spacing': float(np.max(min_distances)),
            'mean_spacing': float(np.mean(min_distances)),
            'spacing_std': float(np.std(min_distances)),
            'mean_neighbors': float(np.mean([len(nbrs) for nbrs in self.neighbor_lists]))
        }