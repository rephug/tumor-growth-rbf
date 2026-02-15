"""
parameter_fitting.py

Patient-specific parameter estimation via inverse problem solving.

CLINICAL CONTEXT
================
The two most important parameters for predicting glioma growth are:
- ρ (rho): proliferation rate (day⁻¹) — how fast tumor cells divide
- D: diffusion coefficient (mm²/day) — how fast tumor cells migrate

These vary enormously between patients (order of magnitude). Given two
MRI scans at times t₁ and t₂ showing the tumor boundary, we find the
(ρ, D) pair that best reproduces the observed growth.

METHOD: Grid search over (ρ, D) parameter space, scored by Dice
coefficient (overlap between simulated and observed tumor contours).
Optional Nelder-Mead refinement for higher precision.

References:
- Swanson KR et al. "A mathematical modelling tool for predicting
  survival of individual patients following resection of glioblastoma."
  Br J Cancer 98:113-119, 2008.
- Harpold HLP et al. "The evolution of mathematical modeling of glioma
  proliferation and invasion." J Neuropathol Exp Neurol 66(1):1-9, 2007.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from ..biology.tumor_model import TumorModel, TumorParameters

logger = logging.getLogger(__name__)


@dataclass
class FittingResult:
    """
    Result of parameter fitting.

    Contains the best-fit parameters plus the full objective landscape
    for visualization and confidence assessment.
    """
    best_rho: float           # Best-fit proliferation rate (day⁻¹)
    best_D: float             # Best-fit diffusion coefficient (mm²/day)
    best_dice: float          # Dice score at optimum (1.0 = perfect)
    rho_grid: np.ndarray      # ρ values searched
    D_grid: np.ndarray        # D values searched
    dice_surface: np.ndarray  # shape (n_rho, n_D) — full objective landscape
    n_simulations: int        # Total simulations run
    refinement_used: bool     # Whether Nelder-Mead was applied


def _evaluate_single(args):
    """
    Module-level function for parallel evaluation.

    Must be at module level (not a method) so it can be pickled
    by ProcessPoolExecutor.
    """
    rho, D, domain_size, n_points, dt, initial_density, observed_density, delta_t = args
    fitter = ParameterFitter(domain_size, n_points=n_points, dt=dt)
    return fitter._evaluate(rho, D, initial_density, observed_density, delta_t)


class ParameterFitter:
    """
    Estimates patient-specific (ρ, D) parameters from serial imaging.

    Usage:
        fitter = ParameterFitter(domain_size=(10.0, 10.0))
        result = fitter.fit(
            initial_density=t1_contour,
            observed_density=t2_contour,
            delta_t=30.0,  # 30 days between scans
        )
        print(f"Best fit: rho={result.best_rho}, D={result.best_D}")
        print(f"Dice score: {result.best_dice}")
    """

    def __init__(self,
                 domain_size: Tuple[float, ...],
                 n_points: int = 500,
                 dt: float = 0.1,
                 tissue_data=None):
        """
        Args:
            domain_size: Physical domain size in mm, e.g. (10.0, 10.0)
            n_points: Number of mesh points for simulations
            dt: Time step size in days
            tissue_data: Optional tissue information (reserved for future use)
        """
        self.domain_size = domain_size
        self.n_points = n_points
        self.dt = dt
        self.tissue_data = tissue_data

    def fit(self,
            initial_density: np.ndarray,
            observed_density: np.ndarray,
            delta_t: float,
            rho_range: Tuple[float, float] = (0.001, 0.1),
            D_range: Tuple[float, float] = (0.01, 1.0),
            n_grid: int = 20,
            refine: bool = False,
            parallel: bool = False
            ) -> FittingResult:
        """
        Find best-fit (ρ, D) via grid search.

        Args:
            initial_density: Tumor density at time t₁ (one value per mesh point)
            observed_density: Tumor density at time t₂ (one value per mesh point)
            delta_t: Time between scans in days
            rho_range: (min, max) for proliferation rate search
            D_range: (min, max) for diffusion coefficient search
            n_grid: Number of grid points per dimension (total = n_grid²)
            refine: If True, refine with Nelder-Mead after grid search
            parallel: If True, use parallel evaluation

        Returns:
            FittingResult with best parameters and objective landscape
        """
        rho_grid = np.linspace(rho_range[0], rho_range[1], n_grid)
        D_grid = np.linspace(D_range[0], D_range[1], n_grid)

        if parallel:
            dice_surface = self._grid_search_parallel(
                rho_grid, D_grid, initial_density, observed_density, delta_t
            )
        else:
            dice_surface = self._grid_search(
                rho_grid, D_grid, initial_density, observed_density, delta_t
            )

        # Find best parameters from grid search
        best_idx = np.unravel_index(np.argmax(dice_surface), dice_surface.shape)
        best_rho = rho_grid[best_idx[0]]
        best_D = D_grid[best_idx[1]]
        best_dice = dice_surface[best_idx]

        n_simulations = n_grid * n_grid
        refinement_used = False

        # Optional Nelder-Mead refinement
        if refine:
            refined_rho, refined_D, refined_dice = self._refine(
                best_rho, best_D, initial_density, observed_density, delta_t
            )
            if refined_dice >= best_dice:
                best_rho = refined_rho
                best_D = refined_D
                best_dice = refined_dice
                refinement_used = True
                logger.info(
                    f"Refinement improved Dice: {dice_surface[best_idx]:.4f} "
                    f"-> {best_dice:.4f}"
                )

        logger.info(
            f"Best fit: rho={best_rho:.4f}, D={best_D:.4f}, "
            f"Dice={best_dice:.4f}"
        )

        return FittingResult(
            best_rho=best_rho,
            best_D=best_D,
            best_dice=best_dice,
            rho_grid=rho_grid,
            D_grid=D_grid,
            dice_surface=dice_surface,
            n_simulations=n_simulations,
            refinement_used=refinement_used,
        )

    def _grid_search(self,
                     rho_grid: np.ndarray,
                     D_grid: np.ndarray,
                     initial_density: np.ndarray,
                     observed_density: np.ndarray,
                     delta_t: float
                     ) -> np.ndarray:
        """Sequential grid search over (ρ, D) parameter space."""
        n_rho = len(rho_grid)
        n_D = len(D_grid)
        dice_surface = np.zeros((n_rho, n_D))

        total = n_rho * n_D
        for i, rho in enumerate(rho_grid):
            for j, D in enumerate(D_grid):
                dice_surface[i, j] = self._evaluate(
                    rho, D, initial_density, observed_density, delta_t
                )
                count = i * n_D + j + 1
                if count % max(1, total // 10) == 0:
                    logger.debug(f"Grid search: {count}/{total} evaluations")

        return dice_surface

    def _grid_search_parallel(self,
                              rho_grid: np.ndarray,
                              D_grid: np.ndarray,
                              initial_density: np.ndarray,
                              observed_density: np.ndarray,
                              delta_t: float
                              ) -> np.ndarray:
        """Parallel grid search using ProcessPoolExecutor."""
        n_rho = len(rho_grid)
        n_D = len(D_grid)
        dice_surface = np.zeros((n_rho, n_D))

        # Build argument list for all evaluations
        args_list = []
        for i, rho in enumerate(rho_grid):
            for j, D in enumerate(D_grid):
                args_list.append((
                    rho, D, self.domain_size, self.n_points, self.dt,
                    initial_density, observed_density, delta_t
                ))

        # Run in parallel
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_evaluate_single, args): (i, j)
                for (i, j), args in zip(
                    [(i, j) for i in range(n_rho) for j in range(n_D)],
                    args_list
                )
            }
            for future in as_completed(futures):
                i, j = futures[future]
                dice_surface[i, j] = future.result()

        return dice_surface

    def _evaluate(self,
                  rho: float,
                  D: float,
                  initial_density: np.ndarray,
                  observed_density: np.ndarray,
                  delta_t: float
                  ) -> float:
        """
        Evaluate a single (ρ, D) candidate.

        Creates a fresh model, initializes with t₁ contour, simulates
        forward to t₂, and computes Dice coefficient vs observed.

        Returns:
            Dice coefficient (0 to 1, higher is better)
        """
        params = TumorParameters(
            growth_rate=rho,
            diffusion_white=D,
        )

        model = TumorModel(
            domain_size=self.domain_size,
            params=params,
            n_initial_points=self.n_points,
        )

        # Set patient-specific initial condition
        model.set_initial_density(initial_density)

        # Simulate forward
        n_steps = max(1, int(delta_t / self.dt))
        for _ in range(n_steps):
            model.update(self.dt)

        # Compare with observed
        return self.dice_coefficient(model.tumor_density, observed_density)

    @staticmethod
    def dice_coefficient(simulated: np.ndarray,
                         observed: np.ndarray,
                         threshold: float = 0.1
                         ) -> float:
        """
        Dice similarity coefficient between two density fields.

        Dice = 2 * |A ∩ B| / (|A| + |B|)

        where A = {points where simulated > threshold}
              B = {points where observed > threshold}

        Args:
            simulated: Simulated tumor density
            observed: Observed tumor density
            threshold: Density threshold for binarization

        Returns:
            Dice coefficient in [0, 1]. Returns 0.0 if both sets are empty.
        """
        sim_binary = simulated > threshold
        obs_binary = observed > threshold

        sum_both = np.sum(sim_binary) + np.sum(obs_binary)
        if sum_both == 0:
            return 0.0

        intersection = np.sum(sim_binary & obs_binary)
        return 2.0 * intersection / sum_both

    def _refine(self,
                rho0: float,
                D0: float,
                initial_density: np.ndarray,
                observed_density: np.ndarray,
                delta_t: float
                ) -> Tuple[float, float, float]:
        """
        Refine parameters using Nelder-Mead optimization.

        Starts from the grid search best and minimizes (1 - Dice).

        Returns:
            (refined_rho, refined_D, refined_dice)
        """
        from scipy.optimize import minimize

        def objective(params):
            rho, D = params
            if rho <= 0 or D <= 0:
                return 1.0  # Worst possible (1 - Dice)
            dice = self._evaluate(
                rho, D, initial_density, observed_density, delta_t
            )
            return 1.0 - dice

        result = minimize(
            objective,
            x0=[rho0, D0],
            method='Nelder-Mead',
            options={
                'maxiter': 50,
                'xatol': 1e-4,
                'fatol': 1e-4,
            }
        )

        refined_rho, refined_D = result.x
        refined_dice = 1.0 - result.fun

        return refined_rho, refined_D, refined_dice
