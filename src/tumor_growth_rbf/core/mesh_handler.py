"""
mesh_handler.py

Basic mesh handler for scattered point distributions.
This is the foundation everything else builds on.

KEY CONCEPT: RBF-FD is a "meshless" method. Instead of a structured grid
(like pixels in an image), we scatter points in the domain and define
spatial relationships through neighbor lists. Each point knows who its
neighbors are, and we compute derivatives using those local neighborhoods.

Why meshless?
- Easy to add/remove points (adaptive refinement)
- No mesh generation headaches for complex geometries
- Natural handling of irregular domains (like brain anatomy)
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MeshHandler:
    """
    Manages scattered point distributions for RBF-FD simulations.

    Think of this like managing a collection of weather stations:
    - Each station (point) has a position
    - Each station knows its nearby stations (neighbors)
    - We use neighbor relationships to estimate spatial derivatives

    Attributes:
        domain_size: Physical size of the simulation domain (Lx, Ly) in mm
        points: Array of point coordinates, shape (N, 2)
        neighbor_lists: For each point, indices of its neighbors
        kdtree: Spatial index for fast neighbor queries
    """

    def __init__(self,
                 domain_size: Tuple[float, ...],
                 min_spacing: float = 0.01,
                 max_spacing: float = 0.1):
        """
        Args:
            domain_size: Physical domain size (Lx, Ly) or (Lx, Ly, Lz) in mm
            min_spacing: Minimum allowed distance between points
            max_spacing: Maximum spacing (controls base resolution)
        """
        if len(domain_size) not in (2, 3):
            raise ValueError(f"domain_size must have 2 or 3 elements, got {len(domain_size)}")
        self.domain_size = domain_size
        self.ndim = len(domain_size)
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing

        # These get populated by initialize_points()
        self.points: Optional[np.ndarray] = None
        self.neighbor_lists: Optional[List[List[int]]] = None
        self.kdtree: Optional[cKDTree] = None

        # Track how many times each point has been refined
        self.refinement_levels: Optional[np.ndarray] = None

    def initialize_points(self,
                          n_points: int,
                          distribution: str = "halton") -> np.ndarray:
        """
        Create initial point distribution in the domain.

        Args:
            n_points: Approximate number of points to generate
            distribution: Point generation strategy:
                - "halton": Quasi-random, good space-filling (recommended)
                - "random": Purely random (fast but clumpy)
                - "grid": Regular grid (simple but inflexible)

        Returns:
            Array of point coordinates, shape (N, 2)

        LEARNING NOTE: The choice of initial point distribution matters!
        - Random points leave gaps and clusters → bad accuracy
        - Grid points are uniform but can't adapt
        - Halton sequences fill space quasi-randomly → good balance
        """
        if distribution == "halton":
            self.points = self._generate_halton_points(n_points)
        elif distribution == "random":
            self.points = self._generate_random_points(n_points)
        elif distribution == "grid":
            self.points = self._generate_grid_points(n_points)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        self.refinement_levels = np.zeros(len(self.points), dtype=int)
        self._build_neighbor_lists()

        logger.info(f"Initialized {len(self.points)} points "
                    f"({distribution} distribution)")
        return self.points

    def _generate_halton_points(self, n_points: int) -> np.ndarray:
        """
        Generate quasi-random Halton sequence points.

        LEARNING NOTE: Halton sequences use prime-base representations
        to fill space more uniformly than random sampling. Primes 2, 3
        (and 5 for 3D) are used for the spatial dimensions.

        Example for base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
        These fill [0,1] more evenly than random numbers would.
        """
        bases = [2, 3, 5][:self.ndim]

        def halton_sequence(index, base):
            """Generate one value in a Halton sequence."""
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f /= base
            return result

        points = np.zeros((n_points, self.ndim))
        for i in range(n_points):
            for d in range(self.ndim):
                points[i, d] = halton_sequence(i + 1, bases[d]) * self.domain_size[d]

        return points

    def _generate_random_points(self, n_points: int) -> np.ndarray:
        """Generate uniformly random points in the domain."""
        points = np.random.rand(n_points, self.ndim)
        for d in range(self.ndim):
            points[:, d] *= self.domain_size[d]
        return points

    def _generate_grid_points(self, n_points: int) -> np.ndarray:
        """Generate points on a regular grid."""
        if self.ndim == 2:
            # Calculate grid dimensions to approximately match n_points
            aspect = self.domain_size[0] / self.domain_size[1]
            ny = int(np.sqrt(n_points / aspect))
            nx = int(n_points / ny)

            x = np.linspace(0, self.domain_size[0], nx)
            y = np.linspace(0, self.domain_size[1], ny)
            xx, yy = np.meshgrid(x, y)

            return np.column_stack([xx.ravel(), yy.ravel()])
        else:  # 3D
            Lx, Ly, Lz = self.domain_size
            # nx/ny = Lx/Ly, nx/nz = Lx/Lz, nx*ny*nz ~ n_points
            nx = max(int(round((n_points * Lx ** 2 / (Ly * Lz)) ** (1.0 / 3.0))), 2)
            ny = max(int(round(nx * Ly / Lx)), 2)
            nz = max(int(round(nx * Lz / Lx)), 2)

            x = np.linspace(0, Lx, nx)
            y = np.linspace(0, Ly, ny)
            z = np.linspace(0, Lz, nz)
            xx, yy, zz = np.meshgrid(x, y, z)

            return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def _build_neighbor_lists(self, n_neighbors: Optional[int] = None):
        """
        Build neighbor lists using k-d tree spatial indexing.

        LEARNING NOTE: The number of neighbors per point is a key parameter.
        - Too few neighbors → inaccurate derivative approximations
        - Too many neighbors → slow computation, ill-conditioned systems
        - 10-20 neighbors is typical for 2D RBF-FD
        - 30-50 neighbors is recommended for 3D RBF-FD

        We use scipy's cKDTree for O(N log N) neighbor finding,
        much faster than brute-force O(N²).
        """
        if self.points is None:
            raise ValueError("Points not initialized")

        if n_neighbors is None:
            n_neighbors = 15 if self.ndim == 2 else 40

        self.kdtree = cKDTree(self.points)

        # Find k nearest neighbors for each point
        # +1 because query includes the point itself
        k = min(n_neighbors + 1, len(self.points))
        distances, indices = self.kdtree.query(self.points, k=k)

        # Convert to list of lists (each point's neighbor indices)
        self.neighbor_lists = [list(idx) for idx in indices]

    def refine_points(self,
                      indicator: np.ndarray,
                      threshold: float = 0.5,
                      max_level: int = 5) -> int:
        """
        Add points in regions where the indicator is high.

        LEARNING NOTE: Adaptive refinement is crucial for tumor simulations.
        The tumor boundary (where density changes rapidly) needs fine resolution,
        but the far-field can be coarse. This saves massive computation.

        Args:
            indicator: Per-point refinement indicator (0 = coarse OK, 1 = need fine)
            threshold: Minimum indicator value to trigger refinement
            max_level: Maximum refinement depth

        Returns:
            Number of points added
        """
        if self.points is None:
            raise ValueError("Points not initialized")

        # Find points that need refinement
        refine_mask = (indicator > threshold) & \
                      (self.refinement_levels < max_level)
        refine_indices = np.where(refine_mask)[0]

        if len(refine_indices) == 0:
            return 0

        new_points = []
        new_levels = []

        for idx in refine_indices:
            center = self.points[idx]
            level = self.refinement_levels[idx]

            # Spacing decreases with refinement level
            spacing = self.max_spacing / (2 ** (level + 1))
            spacing = max(spacing, self.min_spacing)

            # Add points around the center
            if self.ndim == 2:
                # Cross pattern (4 offsets)
                offsets = spacing * np.array([
                    [1, 0], [-1, 0], [0, 1], [0, -1]
                ])
            else:
                # Octahedral pattern (6 offsets) for 3D
                offsets = spacing * np.array([
                    [1, 0, 0], [-1, 0, 0],
                    [0, 1, 0], [0, -1, 0],
                    [0, 0, 1], [0, 0, -1]
                ])

            for offset in offsets:
                new_pt = center + offset
                # Check domain bounds (works for any dimension)
                in_bounds = all(
                    0 <= new_pt[d] <= self.domain_size[d]
                    for d in range(self.ndim)
                )
                if in_bounds:
                    # Check minimum spacing against existing + new points
                    if self._check_min_spacing(new_pt, new_points):
                        new_points.append(new_pt)
                        new_levels.append(level + 1)

        if new_points:
            self.points = np.vstack([self.points, np.array(new_points)])
            self.refinement_levels = np.concatenate([
                self.refinement_levels, np.array(new_levels, dtype=int)
            ])
            self._build_neighbor_lists()

        n_added = len(new_points)
        if n_added > 0:
            logger.debug(f"Refined mesh: added {n_added} points "
                         f"(total: {len(self.points)})")
        return n_added

    def coarsen_points(self,
                       indicator: np.ndarray,
                       threshold: float = 0.05) -> int:
        """
        Remove points in regions where the indicator is low.

        Only removes points that were added by refinement (level > 0),
        never the original base points.

        Args:
            indicator: Per-point refinement indicator
            threshold: Points with indicator below this may be removed

        Returns:
            Number of points removed
        """
        if self.points is None:
            return 0

        # Only coarsen refined points (not base mesh)
        keep_mask = ~((indicator < threshold) &
                      (self.refinement_levels > 0))

        n_removed = np.sum(~keep_mask)

        if n_removed > 0:
            self.points = self.points[keep_mask]
            self.refinement_levels = self.refinement_levels[keep_mask]
            self._build_neighbor_lists()
            logger.debug(f"Coarsened mesh: removed {n_removed} points "
                         f"(total: {len(self.points)})")

        return n_removed

    def _check_min_spacing(self,
                           point: np.ndarray,
                           additional_points: List[np.ndarray]) -> bool:
        """Check if a new point maintains minimum spacing."""
        # Check against existing points
        if self.kdtree is not None:
            dist, _ = self.kdtree.query(point)
            if dist < self.min_spacing:
                return False

        # Check against other new points being added in this batch
        for p in additional_points:
            if np.linalg.norm(point - p) < self.min_spacing:
                return False

        return True

    def interpolate_to_new_points(self,
                                  old_points: np.ndarray,
                                  old_values: np.ndarray,
                                  new_points: Optional[np.ndarray] = None
                                  ) -> np.ndarray:
        """
        Interpolate field values from old points to new points.
        Uses inverse-distance weighting (simple but robust).

        Args:
            old_points: Previous point coordinates
            old_values: Field values at old points
            new_points: New point coordinates (defaults to self.points)

        Returns:
            Interpolated values at new points
        """
        if new_points is None:
            new_points = self.points

        old_tree = cKDTree(old_points)
        k = min(8, len(old_points))
        distances, indices = old_tree.query(new_points, k=k)

        # Inverse distance weighting
        # Handle case where new point coincides with old point
        weights = np.where(distances > 1e-15, 1.0 / distances, 1e15)
        weight_sums = np.sum(weights, axis=1, keepdims=True)

        interpolated = np.sum(weights * old_values[indices], axis=1) / \
                       weight_sums.ravel()

        return interpolated

    def get_metrics(self) -> Dict:
        """Get mesh quality metrics."""
        if self.points is None:
            return {"n_points": 0}

        # Compute nearest-neighbor distances
        distances, _ = self.kdtree.query(self.points, k=2)
        nn_distances = distances[:, 1]  # Skip self (distance 0)

        return {
            "n_points": len(self.points),
            "min_spacing": float(np.min(nn_distances)),
            "max_spacing": float(np.max(nn_distances)),
            "mean_spacing": float(np.mean(nn_distances)),
            "mean_refinement_level": float(np.mean(self.refinement_levels)),
            "max_refinement_level": int(np.max(self.refinement_levels)),
        }
