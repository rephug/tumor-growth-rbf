#!/usr/bin/env python3
"""
pde_assembler.py

Assembles PDE operators using RBF-FD methods.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PDEAssembler:
    """
    Assembles PDE operators for tumor growth simulation.
    """
    
    def __init__(self, rbf_solver):
        self.rbf_solver = rbf_solver
        
    def build_operator(self,
                      points: np.ndarray,
                      neighbor_lists: List[List[int]],
                      operator_type: str = "laplacian",
                      coefficients: Optional[np.ndarray] = None) -> csr_matrix:
        """
        Build sparse operator matrix.
        
        Args:
            points: Node coordinates
            neighbor_lists: Lists of neighbor indices
            operator_type: Type of operator to build
            coefficients: Optional coefficient field
            
        Returns:
            Sparse operator matrix
        """
        if operator_type not in ["laplacian", "gradient"]:
            raise ValueError(f"Unknown operator type: {operator_type}")
            
        n_points = len(points)
        rows = []
        cols = []
        data = []
        
        for i in range(n_points):
            nbrs = neighbor_lists[i]
            center = points[i]
            neighbor_coords = points[nbrs]
            
            # Compute weights for operator
            weights = self.rbf_solver.compute_weights(
                center, neighbor_coords, operator_type
            )
            
            # Apply coefficient if provided
            if coefficients is not None:
                weights *= coefficients[i]
                
            rows.extend([i] * len(nbrs))
            cols.extend(nbrs)
            data.extend(weights)
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        
    def build_reaction_diffusion(self,
                               points: np.ndarray,
                               neighbor_lists: List[List[int]],
                               diffusion_coeff: np.ndarray,
                               reaction_coeff: np.ndarray) -> csr_matrix:
        """
        Build reaction-diffusion operator.
        
        Args:
            points: Node coordinates
            neighbor_lists: Lists of neighbor indices
            diffusion_coeff: Diffusion coefficient field
            reaction_coeff: Reaction coefficient field
            
        Returns:
            Reaction-diffusion operator matrix
        """
        # Build diffusion operator
        diffusion = self.build_operator(
            points, neighbor_lists, "laplacian", diffusion_coeff
        )
        
        # Add reaction term to diagonal
        n_points = len(points)
        reaction = csr_matrix(
            (reaction_coeff, (range(n_points), range(n_points))),
            shape=(n_points, n_points)
        )
        
        return diffusion + reaction
        
    def build_conservation_constraint(self,
                                    points: np.ndarray,
                                    neighbor_lists: List[List[int]]) -> np.ndarray:
        """
        Build constraint vector for mass conservation.
        
        Args:
            points: Node coordinates
            neighbor_lists: Lists of neighbor indices
            
        Returns:
            Constraint vector
        """
        n_points = len(points)
        constraint = np.zeros(n_points)
        
        # Compute volumes/areas associated with each node
        volumes = self._compute_node_volumes(points, neighbor_lists)
        
        for i in range(n_points):
            constraint[i] = volumes[i]
            
        return constraint
        
    def _compute_node_volumes(self,
                            points: np.ndarray,
                            neighbor_lists: List[List[int]]) -> np.ndarray:
        """
        Compute approximate volumes/areas associated with nodes.
        
        Args:
            points: Node coordinates
            neighbor_lists: Lists of neighbor indices
            
        Returns:
            Array of node volumes/areas
        """
        n_points = len(points)
        volumes = np.zeros(n_points)
        
        for i in range(n_points):
            # Use neighbor distances to estimate local volume
            nbrs = neighbor_lists[i]
            if len(nbrs) > 1:
                dists = np.linalg.norm(points[nbrs] - points[i], axis=1)
                volumes[i] = np.mean(dists)**points.shape[1]
            else:
                # Fallback for isolated points
                volumes[i] = 1.0
                
        # Normalize
        volumes /= np.sum(volumes)
        return volumes