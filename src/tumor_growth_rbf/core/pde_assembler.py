"""
pde_assembler.py

Assembles PDE operators using RBF-FD methods.

KEY CONCEPT: Separation of Concerns
====================================
The RBF solver knows how to compute weights for individual stencils.
The PDE assembler uses those weights to build GLOBAL operators that
apply across the entire domain.

Think of it like this:
- RBFSolver = "how to compute a derivative at one point"
- PDEAssembler = "how to compute derivatives everywhere, and combine
  them into PDE operators like reaction-diffusion"

This separation makes it easy to swap in different RBF kernels or
add new PDE types without rewriting everything.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PDEAssembler:
    """
    Assembles PDE operators for tumor growth simulation.

    Supported operators:
    - "laplacian": ∇²u (for diffusion)
    - "gradient_x": ∂u/∂x
    - "gradient_y": ∂u/∂y

    Can also build composite operators like reaction-diffusion:
        ∂u/∂t = D∇²u + R(u)
    """

    def __init__(self, rbf_solver):
        """
        Args:
            rbf_solver: RBFSolver instance for computing stencil weights
        """
        self.rbf_solver = rbf_solver
        # Cache assembled operators for reuse
        self._operator_cache = {}

    def build_operator(self,
                       points: np.ndarray,
                       neighbor_lists: List[List[int]],
                       operator: str = "laplacian",
                       coefficients: Optional[np.ndarray] = None
                       ) -> csr_matrix:
        """
        Build a sparse operator matrix.

        LEARNING NOTE: The result is a matrix L such that:
            L @ u = [operator(u) at point 0, operator(u) at point 1, ...]

        For example, if operator="laplacian":
            (L @ u)[i] ≈ ∇²u(x_i)

        If coefficients are provided (e.g., diffusion coefficient D(x)),
        the operator becomes D(x)*L, which handles spatially-varying
        material properties (like different diffusion in white vs gray matter).

        Args:
            points: Node coordinates, shape (N, 2)
            neighbor_lists: Neighbor indices for each point
            operator: "laplacian", "gradient_x", or "gradient_y"
            coefficients: Optional per-point coefficient field

        Returns:
            Sparse operator matrix, shape (N, N)
        """
        valid_operators = ["laplacian", "gradient_x", "gradient_y", "gradient_z"]
        if operator not in valid_operators:
            raise ValueError(
                f"Unknown operator '{operator}'. "
                f"Valid operators: {valid_operators}"
            )

        n_points = len(points)
        rows = []
        cols = []
        data = []

        for i in range(n_points):
            nbrs = neighbor_lists[i]
            center = points[i]
            neighbor_coords = points[nbrs]

            # Get stencil weights from RBF solver
            weights = self.rbf_solver.compute_weights(
                center, neighbor_coords, operator
            )

            # Apply spatially-varying coefficient if provided
            if coefficients is not None:
                weights *= coefficients[i]

            rows.extend([i] * len(nbrs))
            cols.extend(nbrs)
            data.extend(weights.tolist())

        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    def build_reaction_diffusion(self,
                                 points: np.ndarray,
                                 neighbor_lists: List[List[int]],
                                 diffusion_coeff: np.ndarray,
                                 reaction_coeff: np.ndarray
                                 ) -> csr_matrix:
        """
        Build combined reaction-diffusion operator: D∇² + R

        LEARNING NOTE: Many biological PDEs have the form:
            ∂u/∂t = D(x)∇²u + R(x)*u

        The first term (diffusion) spreads things out spatially.
        The second term (reaction) handles local growth/decay.

        For tumor growth: D is the diffusion coefficient (tissue-dependent),
        and R captures the net growth rate.

        Args:
            points: Node coordinates
            neighbor_lists: Neighbor indices
            diffusion_coeff: Per-point diffusion coefficient D(x)
            reaction_coeff: Per-point reaction coefficient R(x)

        Returns:
            Combined operator matrix
        """
        # Diffusion: D(x) * ∇²
        diffusion_op = self.build_operator(
            points, neighbor_lists, "laplacian", diffusion_coeff
        )

        # Reaction: diagonal matrix with R(x) on diagonal
        n_points = len(points)
        reaction_op = diags(reaction_coeff, 0, shape=(n_points, n_points),
                            format='csr')

        return diffusion_op + reaction_op

    def build_advection_operator(self,
                                 points: np.ndarray,
                                 neighbor_lists: List[List[int]],
                                 velocity_x: np.ndarray,
                                 velocity_y: np.ndarray,
                                 velocity_z: Optional[np.ndarray] = None
                                 ) -> csr_matrix:
        """
        Build advection operator: v·∇u = vx*∂u/∂x + vy*∂u/∂y [+ vz*∂u/∂z]

        Useful for modeling directed cell migration or drug transport.

        Args:
            points: Node coordinates, shape (N, d) where d is 2 or 3
            neighbor_lists: Neighbor indices
            velocity_x: x-component of velocity field
            velocity_y: y-component of velocity field
            velocity_z: z-component of velocity field (3D only)

        Returns:
            Advection operator matrix
        """
        grad_x = self.build_operator(
            points, neighbor_lists, "gradient_x", velocity_x
        )
        grad_y = self.build_operator(
            points, neighbor_lists, "gradient_y", velocity_y
        )
        result = grad_x + grad_y
        if velocity_z is not None:
            grad_z = self.build_operator(
                points, neighbor_lists, "gradient_z", velocity_z
            )
            result = result + grad_z
        return result
