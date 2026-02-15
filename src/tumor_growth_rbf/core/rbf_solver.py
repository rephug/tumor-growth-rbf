"""
rbf_solver.py

RBF-FD (Radial Basis Function - Finite Differences) solver.

KEY CONCEPT: How RBF-FD works
================================
Traditional finite differences approximate derivatives on a regular grid:
    du/dx ≈ (u[i+1] - u[i-1]) / (2*dx)

RBF-FD does the same thing but on SCATTERED points:
1. Pick a center point and its neighbors
2. Fit an RBF interpolant through those neighbor values
3. Analytically differentiate the interpolant
4. This gives you "weights" — multiply neighbor values by weights to get the derivative

The magic: these weights work for ANY point arrangement, not just grids.

MATH BACKGROUND:
- Default kernel: Polyharmonic Spline (PHS): φ(r) = r^k (default k=3)
  - NO shape parameter — unlike Gaussian, no tuning required
  - Better conditioned interpolation matrices
  - Recommended for modern RBF-FD (Flyer et al. 2016, Bayona et al. 2017)
- Legacy kernel: Gaussian: φ(r) = exp(-(ε*r)²) — still available as fallback
  - ε (epsilon) controls how flat/peaked the basis functions are
  - Small ε: flat, smooth, but ill-conditioned
  - Large ε: peaked, localized, but less accurate
- Both kernels are augmented with polynomials for consistency
  (so constant/linear/quadratic fields are reproduced exactly)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class RBFSolver:
    """
    RBF-FD solver for spatial differential operators.

    This class computes the "stencil weights" that let us approximate
    differential operators (Laplacian, gradient) at scattered points.
    """

    def __init__(self,
                 epsilon: float = 1.0,
                 poly_degree: int = 2,
                 kernel: str = "phs",
                 phs_order: int = 3,
                 tol: float = 1e-10):
        """
        Args:
            epsilon: RBF shape parameter (only used for Gaussian kernel)
            poly_degree: Degree of polynomial augmentation (0, 1, or 2)
                - 0: constant reproduction
                - 1: linear reproduction (recommended minimum)
                - 2: quadratic reproduction (best accuracy)
            kernel: RBF kernel type — "phs" (default) or "gaussian" (legacy)
            phs_order: PHS order k (only used for PHS kernel, default 3)
                - Odd k: φ(r) = r^k  (r^3, r^5, r^7)
                - Even k: φ(r) = r^k * log(r)
            tol: Tolerance for numerical checks
        """
        self.epsilon = epsilon
        self.poly_degree = poly_degree
        self.kernel = kernel
        self.phs_order = phs_order
        self.tol = tol
        self._validate_parameters()

    def _validate_parameters(self):
        if self.kernel not in ("phs", "gaussian"):
            raise ValueError(f"Unknown kernel: {self.kernel}")
        if self.kernel == "gaussian" and self.epsilon <= 0:
            raise ValueError("Epsilon must be positive for Gaussian kernel")
        if self.kernel == "phs" and self.phs_order < 1:
            raise ValueError("PHS order must be >= 1")
        if self.poly_degree < 0:
            raise ValueError("Polynomial degree must be non-negative")

    # ----------------------------------------------------------------
    # Core: Building local interpolation systems
    # ----------------------------------------------------------------

    def build_local_matrices(self,
                             center: np.ndarray,
                             neighbors: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the local RBF interpolation system for one stencil.

        LEARNING NOTE: The interpolation problem is:
            Given values f_j at neighbor points x_j,
            find coefficients λ_j such that:
                s(x) = Σ λ_j * φ(||x - x_j||) + polynomial terms
            interpolates the data: s(x_j) = f_j

        This leads to a linear system [A P; P^T 0] [λ; c] = [f; 0]
        where A_ij = φ(||x_i - x_j||) is the RBF matrix
        and P is the polynomial basis matrix.

        Args:
            center: The point where we want to evaluate the operator
            neighbors: Coordinates of neighbor points, shape (n, 2)

        Returns:
            (A, P) — the RBF matrix and polynomial matrix
        """
        n = len(neighbors)

        # RBF matrix: A_ij = φ(||x_i - x_j||)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                r = np.linalg.norm(neighbors[i] - neighbors[j])
                A[i, j] = self._phi(r)

        # Polynomial matrix
        P = self._build_poly_matrix(neighbors)

        return A, P

    def compute_weights(self,
                        center: np.ndarray,
                        neighbors: np.ndarray,
                        operator: str = "laplacian") -> np.ndarray:
        """
        Compute RBF-FD weights for a differential operator at one point.

        LEARNING NOTE: To get derivative weights, we:
        1. Build the interpolation system [A P; P^T 0]
        2. Apply the differential operator to the RBF/polynomial basis
        3. Evaluate the result at the center point
        4. Solve for the weights

        The weights w satisfy: L[u](center) ≈ Σ w_j * u(x_j)
        where L is the differential operator (e.g., Laplacian).

        Args:
            center: Point where operator is evaluated
            neighbors: Neighbor coordinates, shape (n, d) where d is 2 or 3
            operator: One of "laplacian", "gradient_x", "gradient_y", "gradient_z"

        Returns:
            Weight vector, shape (n,)
        """
        A, P = self.build_local_matrices(center, neighbors)

        n_points = len(neighbors)
        n_poly = P.shape[1]
        ndim = neighbors.shape[1]

        # Build augmented system: [A P; P^T 0]
        M = np.zeros((n_points + n_poly, n_points + n_poly))
        M[:n_points, :n_points] = A
        M[:n_points, n_points:] = P
        M[n_points:, :n_points] = P.T

        # Right-hand side: operator applied to basis functions at center
        rhs = np.zeros(n_points + n_poly)

        # RBF part: L[φ(||x - x_j||)] evaluated at center
        if operator == "laplacian":
            if self.kernel == "phs":
                lap_fn = (self._laplacian_phs_2d if ndim == 2
                          else self._laplacian_phs_3d)
            else:
                lap_fn = (self._laplacian_gaussian_rbf_2d if ndim == 2
                          else self._laplacian_gaussian_rbf_3d)
            for j in range(n_points):
                r = np.linalg.norm(center - neighbors[j])
                rhs[j] = lap_fn(r)

        elif operator in ("gradient_x", "gradient_y", "gradient_z"):
            comp = {"gradient_x": 0, "gradient_y": 1, "gradient_z": 2}[operator]
            for j in range(n_points):
                diff = center - neighbors[j]
                r = np.linalg.norm(diff)
                if self.kernel == "phs":
                    rhs[j] = self._gradient_phs(r, diff, comp)
                else:
                    grad_fns = {
                        "gradient_x": self._gradient_x_gaussian_rbf,
                        "gradient_y": self._gradient_y_gaussian_rbf,
                        "gradient_z": self._gradient_z_gaussian_rbf,
                    }
                    rhs[j] = grad_fns[operator](r, diff)

        else:
            raise ValueError(f"Unknown operator: {operator}")

        # Polynomial part: L[polynomial basis] evaluated at center
        rhs[n_points:] = self._operator_on_polynomials(center, operator)

        # Solve the system
        try:
            weights_full = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            logger.warning("Singular system, using least-squares fallback")
            weights_full = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return weights_full[:n_points]

    def assemble_global_operator(self,
                                 points: np.ndarray,
                                 neighbor_lists: List[List[int]],
                                 operator: str = "laplacian") -> csr_matrix:
        """
        Assemble a global sparse operator matrix from local stencils.

        LEARNING NOTE: This is where local RBF-FD becomes a global method.
        Each row i of the matrix contains the weights for computing the
        operator at point i using its neighbors. The result is a sparse
        matrix L such that:
            L @ u ≈ [L[u](x_0), L[u](x_1), ..., L[u](x_N)]

        This is analogous to assembling a finite element stiffness matrix,
        but much simpler since each stencil is independent.

        Args:
            points: All point coordinates, shape (N, 2)
            neighbor_lists: Neighbor indices for each point
            operator: Differential operator type

        Returns:
            Sparse operator matrix, shape (N, N)
        """
        n_points = len(points)
        rows = []
        cols = []
        data = []

        for i in range(n_points):
            nbrs = neighbor_lists[i]
            center = points[i]
            neighbor_coords = points[nbrs]

            weights = self.compute_weights(center, neighbor_coords, operator)

            rows.extend([i] * len(nbrs))
            cols.extend(nbrs)
            data.extend(weights.tolist())

        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    def interpolate(self,
                    old_points: np.ndarray,
                    new_points: np.ndarray,
                    values: np.ndarray) -> np.ndarray:
        """
        Interpolate field values from old points to new points.
        Uses RBF interpolation with the same kernel.

        Args:
            old_points: Source point coordinates
            new_points: Target point coordinates
            values: Values at source points

        Returns:
            Interpolated values at target points
        """
        # For efficiency, use inverse-distance weighting for large problems
        old_tree = cKDTree(old_points)
        k = min(10, len(old_points))
        distances, indices = old_tree.query(new_points, k=k)

        # Inverse distance weighting (robust fallback)
        weights = np.where(distances > 1e-15, 1.0 / distances, 1e15)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        interpolated = np.sum(weights * values[indices], axis=1) / \
                       weight_sums.ravel()

        return interpolated

    # ----------------------------------------------------------------
    # RBF kernel and its derivatives
    # ----------------------------------------------------------------

    def _phi(self, r: float) -> float:
        """Evaluate the RBF kernel at distance r (dispatches to active kernel)."""
        if self.kernel == "phs":
            return self._phs_rbf(r)
        else:
            return self._gaussian_rbf(r)

    def _gaussian_rbf(self, r: float) -> float:
        """Gaussian RBF: φ(r) = exp(-(ε*r)²)"""
        return np.exp(-(self.epsilon * r) ** 2)

    def _phs_rbf(self, r) -> float:
        """
        Polyharmonic Spline RBF.

        Odd order k: φ(r) = r^k   (r^3, r^5, r^7, ...)
        Even order k: φ(r) = r^k * log(r)   (r^2*log(r), r^4*log(r), ...)
        """
        k = self.phs_order
        if k % 2 == 1:
            return r ** k
        else:
            if isinstance(r, np.ndarray):
                return np.where(r > 0, r ** k * np.log(r), 0.0)
            else:
                return r ** k * np.log(r) if r > 0 else 0.0

    def _laplacian_gaussian_rbf_2d(self, r: float) -> float:
        """
        Laplacian of Gaussian RBF in 2D.

        LEARNING NOTE — CRITICAL FIX from original code!
        The original had the 3D formula. In 2D:

        φ(r) = exp(-(εr)²)
        ∇²φ = d²φ/dr² + (1/r)(dφ/dr)   [2D Laplacian in polar coords]

        Working it out:
            dφ/dr = -2ε²r * exp(-(εr)²)
            d²φ/dr² = (4ε⁴r² - 2ε²) * exp(-(εr)²)
            (1/r)(dφ/dr) = -2ε² * exp(-(εr)²)

        So: ∇²φ = 2ε² * (2ε²r² - 2) * exp(-(εr)²)

        Compare 3D:
            ∇²φ = 2ε² * (2ε²r² - 3) * exp(-(εr)²)
                                   ^ THIS 3 should be 2 in 2D
        """
        e = self.epsilon
        er2 = (e * r) ** 2
        return 2.0 * e ** 2 * (2.0 * er2 - 2.0) * np.exp(-er2)

    def _laplacian_gaussian_rbf_3d(self, r: float) -> float:
        """
        Laplacian of Gaussian RBF in 3D.

        In 3D (spherical coords):
            ∇²φ = d²φ/dr² + (2/r)(dφ/dr)

        The (d-1)/r prefactor gives d-1=2 in 3D vs d-1=1 in 2D,
        changing the constant from -2 to -3:

            ∇²φ = 2ε² * (2ε²r² - 3) * exp(-(εr)²)
        """
        e = self.epsilon
        er2 = (e * r) ** 2
        return 2.0 * e ** 2 * (2.0 * er2 - 3.0) * np.exp(-er2)

    def _gradient_x_gaussian_rbf(self, r: float, diff: np.ndarray) -> float:
        """
        x-derivative of Gaussian RBF.

        dφ/dx = dφ/dr * dr/dx = dφ/dr * (x - x_j)/r
        dφ/dr = -2ε²r * exp(-(εr)²)
        So: dφ/dx = -2ε²(x - x_j) * exp(-(εr)²)
        """
        if r < 1e-15:
            return 0.0
        e = self.epsilon
        return -2.0 * e ** 2 * diff[0] * np.exp(-(e * r) ** 2)

    def _gradient_y_gaussian_rbf(self, r: float, diff: np.ndarray) -> float:
        """y-derivative of Gaussian RBF (same form as x, using y-component)."""
        if r < 1e-15:
            return 0.0
        e = self.epsilon
        return -2.0 * e ** 2 * diff[1] * np.exp(-(e * r) ** 2)

    def _gradient_z_gaussian_rbf(self, r: float, diff: np.ndarray) -> float:
        """z-derivative of Gaussian RBF (same form as x/y, using z-component)."""
        if r < 1e-15:
            return 0.0
        e = self.epsilon
        return -2.0 * e ** 2 * diff[2] * np.exp(-(e * r) ** 2)

    # ----------------------------------------------------------------
    # PHS kernel derivatives
    # ----------------------------------------------------------------

    def _laplacian_phs_2d(self, r: float) -> float:
        """
        Laplacian of PHS in 2D.

        For odd k:  φ(r) = r^k
            dφ/dr = k * r^(k-1)
            d²φ/dr² = k*(k-1) * r^(k-2)
            ∇²φ = d²φ/dr² + (1/r)*dφ/dr = k² * r^(k-2)

        For even k: φ(r) = r^k * log(r)
            ∇²φ = r^(k-2) * (k² * log(r) + 2k)
        """
        k = self.phs_order
        if r < 1e-15:
            return 0.0
        if k % 2 == 1:
            return k * k * r ** (k - 2)
        else:
            return r ** (k - 2) * (k * k * np.log(r) + 2 * k)

    def _laplacian_phs_3d(self, r: float) -> float:
        """
        Laplacian of PHS in 3D.

        For odd k:  φ(r) = r^k
            ∇²φ = d²φ/dr² + (2/r)*dφ/dr = k*(k+1) * r^(k-2)

        For even k: φ(r) = r^k * log(r)
            ∇²φ = r^(k-2) * (k*(k+1) * log(r) + 2k + 1)
        """
        k = self.phs_order
        if r < 1e-15:
            return 0.0
        if k % 2 == 1:
            return k * (k + 1) * r ** (k - 2)
        else:
            return r ** (k - 2) * (k * (k + 1) * np.log(r) + 2 * k + 1)

    def _gradient_phs(self, r: float, diff: np.ndarray,
                      component: int) -> float:
        """
        Gradient component of PHS.

        For odd k:  dφ/dx_i = k * r^(k-2) * (x_i - x_j_i)
        For even k: dφ/dx_i = r^(k-2) * (k*log(r) + 1) * (x_i - x_j_i)
        """
        k = self.phs_order
        if r < 1e-15:
            return 0.0
        if k % 2 == 1:
            return k * r ** (k - 2) * diff[component]
        else:
            return r ** (k - 2) * (k * np.log(r) + 1) * diff[component]

    # ----------------------------------------------------------------
    # Polynomial basis and its derivatives
    # ----------------------------------------------------------------

    def _build_poly_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Build polynomial basis matrix.

        LEARNING NOTE: Polynomial augmentation ensures that the RBF-FD
        method exactly reproduces polynomials up to a given degree.
        This is important for accuracy: if your solution is locally
        well-approximated by a polynomial, the method will be exact.

        2D:
            Degree 0: [1]                           → 1 term
            Degree 1: [1, x, y]                     → 3 terms
            Degree 2: [1, x, y, x², xy, y²]         → 6 terms

        3D:
            Degree 0: [1]                            → 1 term
            Degree 1: [1, x, y, z]                   → 4 terms
            Degree 2: [1, x, y, z, x², xy, xz, y², yz, z²] → 10 terms
        """
        n = len(points)
        ndim = points.shape[1]
        x = points[:, 0]
        y = points[:, 1]

        if ndim == 2:
            if self.poly_degree == 0:
                return np.ones((n, 1))
            elif self.poly_degree == 1:
                return np.column_stack([np.ones(n), x, y])
            elif self.poly_degree == 2:
                return np.column_stack([
                    np.ones(n), x, y, x ** 2, x * y, y ** 2
                ])
            else:
                raise ValueError(f"Polynomial degree {self.poly_degree} not supported")

        elif ndim == 3:
            z = points[:, 2]
            if self.poly_degree == 0:
                return np.ones((n, 1))
            elif self.poly_degree == 1:
                return np.column_stack([np.ones(n), x, y, z])
            elif self.poly_degree == 2:
                return np.column_stack([
                    np.ones(n), x, y, z,
                    x ** 2, x * y, x * z, y ** 2, y * z, z ** 2
                ])
            else:
                raise ValueError(f"Polynomial degree {self.poly_degree} not supported")

        else:
            raise ValueError(f"Unsupported dimension: {ndim}")

    def _operator_on_polynomials(self,
                                 center: np.ndarray,
                                 operator: str) -> np.ndarray:
        """
        Apply differential operator to polynomial basis at center point.

        2D basis [1, x, y, x², xy, y²]:
            ∇²:   [0, 0, 0, 2, 0, 2]
            d/dx: [0, 1, 0, 2x, y, 0]
            d/dy: [0, 0, 1, 0, x, 2y]

        3D basis [1, x, y, z, x², xy, xz, y², yz, z²]:
            ∇²:   [0, 0, 0, 0, 2, 0, 0, 2, 0, 2]
            d/dx: [0, 1, 0, 0, 2x, y, z, 0, 0, 0]
            d/dy: [0, 0, 1, 0, 0, x, 0, 2y, z, 0]
            d/dz: [0, 0, 0, 1, 0, 0, x, 0, y, 2z]
        """
        ndim = len(center)

        if ndim == 2:
            x, y = center

            if operator == "laplacian":
                if self.poly_degree == 0:
                    return np.array([0.0])
                elif self.poly_degree == 1:
                    return np.array([0.0, 0.0, 0.0])
                else:  # degree 2
                    return np.array([0.0, 0.0, 0.0, 2.0, 0.0, 2.0])

            elif operator == "gradient_x":
                if self.poly_degree == 0:
                    return np.array([0.0])
                elif self.poly_degree == 1:
                    return np.array([0.0, 1.0, 0.0])
                else:  # degree 2
                    return np.array([0.0, 1.0, 0.0, 2 * x, y, 0.0])

            elif operator == "gradient_y":
                if self.poly_degree == 0:
                    return np.array([0.0])
                elif self.poly_degree == 1:
                    return np.array([0.0, 0.0, 1.0])
                else:  # degree 2
                    return np.array([0.0, 0.0, 1.0, 0.0, x, 2 * y])

            else:
                raise ValueError(f"Unknown operator: {operator}")

        elif ndim == 3:
            x, y, z = center

            if operator == "laplacian":
                if self.poly_degree == 0:
                    return np.array([0.0])
                elif self.poly_degree == 1:
                    return np.array([0.0, 0.0, 0.0, 0.0])
                else:  # degree 2
                    return np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0])

            elif operator == "gradient_x":
                if self.poly_degree == 0:
                    return np.array([0.0])
                elif self.poly_degree == 1:
                    return np.array([0.0, 1.0, 0.0, 0.0])
                else:  # degree 2
                    return np.array([0.0, 1.0, 0.0, 0.0, 2 * x, y, z, 0.0, 0.0, 0.0])

            elif operator == "gradient_y":
                if self.poly_degree == 0:
                    return np.array([0.0])
                elif self.poly_degree == 1:
                    return np.array([0.0, 0.0, 1.0, 0.0])
                else:  # degree 2
                    return np.array([0.0, 0.0, 1.0, 0.0, 0.0, x, 0.0, 2 * y, z, 0.0])

            elif operator == "gradient_z":
                if self.poly_degree == 0:
                    return np.array([0.0])
                elif self.poly_degree == 1:
                    return np.array([0.0, 0.0, 0.0, 1.0])
                else:  # degree 2
                    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, x, 0.0, y, 2 * z])

            else:
                raise ValueError(f"Unknown operator: {operator}")

        else:
            raise ValueError(f"Unsupported dimension: {ndim}")
