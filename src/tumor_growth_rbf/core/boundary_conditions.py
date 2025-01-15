import numpy as np
from typing import Tuple, List
from enum import Enum

class BoundaryType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    NO_FLUX = "no_flux"

class BoundaryConditionHandler:
    """
    Handles boundary conditions for the tumor growth simulation.
    Replaces np.roll() with explicit boundary handling.
    """
    def __init__(self, domain_size: Tuple[float, float], mesh_handler):
        self.domain_size = domain_size
        self.mesh = mesh_handler
        self._identify_boundary_points()
        
    def _identify_boundary_points(self):
        """Identify points on domain boundaries."""
        points = self.mesh.points
        tol = 1e-10
        
        # Create masks for each boundary
        self.left_boundary = points[:, 0] < tol
        self.right_boundary = np.abs(points[:, 0] - self.domain_size[0]) < tol
        self.bottom_boundary = points[:, 1] < tol
        self.top_boundary = np.abs(points[:, 1] - self.domain_size[1]) < tol
        
        # All boundary points
        self.boundary_mask = (self.left_boundary | self.right_boundary |
                            self.bottom_boundary | self.top_boundary)
                            
    def apply_boundary_conditions(self,
                                field: np.ndarray,
                                boundary_type: BoundaryType,
                                boundary_value: float = 0.0) -> np.ndarray:
        """
        Apply specified boundary conditions to a field.
        
        Args:
            field: Field to apply boundary conditions to
            boundary_type: Type of boundary condition
            boundary_value: Value for Dirichlet conditions
            
        Returns:
            Field with boundary conditions applied
        """
        if boundary_type == BoundaryType.DIRICHLET:
            return self._apply_dirichlet(field, boundary_value)
        elif boundary_type == BoundaryType.NEUMANN:
            return self._apply_neumann(field)
        elif boundary_type == BoundaryType.NO_FLUX:
            return self._apply_no_flux(field)
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
            
    def _apply_dirichlet(self, field: np.ndarray, value: float) -> np.ndarray:
        """Apply Dirichlet (fixed value) boundary conditions."""
        result = field.copy()
        result[self.boundary_mask] = value
        return result
        
    def _apply_neumann(self, field: np.ndarray) -> np.ndarray:
        """Apply Neumann (fixed gradient) boundary conditions."""
        result = field.copy()
        
        # Zero gradient at boundaries
        for points, normal in [
            (self.left_boundary, [-1, 0]),
            (self.right_boundary, [1, 0]),
            (self.bottom_boundary, [0, -1]),
            (self.top_boundary, [0, 1])
        ]:
            if np.any(points):
                # Find nearest interior points
                interior_values = self._find_interior_values(field, points)
                result[points] = interior_values
                
        return result
        
    def _apply_no_flux(self, field: np.ndarray) -> np.ndarray:
        """Apply no-flux boundary conditions."""
        # For tumor growth, no-flux is equivalent to zero Neumann
        return self._apply_neumann(field)
        
    def _find_interior_values(self,
                            field: np.ndarray,
                            boundary_points: np.ndarray) -> np.ndarray:
        """Find values at nearest interior points for each boundary point."""
        boundary_indices = np.where(boundary_points)[0]
        interior_values = np.zeros(len(boundary_indices))
        
        for i, idx in enumerate(boundary_indices):
            # Get neighbors
            neighbors = self.mesh.neighbor_lists[idx]
            # Filter out other boundary points
            interior_neighbors = [n for n in neighbors 
                               if not self.boundary_mask[n]]
            
            if interior_neighbors:
                # Use average of interior neighbors
                interior_values[i] = np.mean(field[interior_neighbors])
            else:
                # Fallback if no interior neighbors found
                interior_values[i] = field[idx]
                
        return interior_values