"""
visualization.py

Visualization tools for the meshless tumor growth model.

KEY DESIGN NOTE: Since we use scattered points (not a grid),
we use scatter plots instead of imshow. This actually gives a
nice visual effect where you can see the point distribution
and how it adapts to the tumor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TumorVisualizer:
    """
    Visualization tools for scattered-point tumor simulations.

    Uses Delaunay triangulation for smooth contour-like plots,
    and scatter plots for showing point distribution.
    """

    def __init__(self,
                 model,
                 figsize: Tuple[int, int] = (10, 8),
                 output_dir: Optional[str] = None):
        self.model = model
        self.figsize = figsize
        self.output_dir = Path(output_dir) if output_dir else None

        # Color schemes
        self.population_colors = {
            'G1': '#2ecc71', 'S': '#e74c3c', 'G2': '#3498db',
            'M': '#f1c40f', 'Q': '#95a5a6', 'N': '#2c3e50'
        }

    def create_state_visualization(self,
                                    time: float,
                                    show_oxygen: bool = True
                                    ) -> plt.Figure:
        """
        Create multi-panel visualization of current tumor state.

        Uses triangulated contour plots for smooth rendering
        of scattered point data.
        """
        n_plots = 2 if show_oxygen else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        points = self.model.mesh.points
        x, y = points[:, 0], points[:, 1]

        # Triangulate points for smooth plotting
        try:
            tri = Triangulation(x, y)
        except Exception:
            # Fallback to scatter if triangulation fails
            tri = None

        # Tumor density
        ax = axes[0]
        if tri is not None:
            tc = ax.tricontourf(tri, self.model.tumor_density,
                                levels=20, cmap='hot')
        else:
            tc = ax.scatter(x, y, c=self.model.tumor_density,
                            cmap='hot', s=10)
        plt.colorbar(tc, ax=ax, label='Density')
        ax.set_title(f'Tumor Density (t={time:.1f} days)')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')

        # Oxygen distribution
        if show_oxygen:
            ax = axes[1]
            if tri is not None:
                tc = ax.tricontourf(tri, self.model.oxygen,
                                    levels=20, cmap='RdYlBu_r')
            else:
                tc = ax.scatter(x, y, c=self.model.oxygen,
                                cmap='RdYlBu_r', s=10)
            plt.colorbar(tc, ax=ax, label='O₂ Concentration')
            ax.set_title('Oxygen Distribution')
            ax.set_xlabel('x (mm)')
            ax.set_aspect('equal')

        plt.tight_layout()
        return fig

    def plot_cell_populations(self) -> plt.Figure:
        """Plot spatial distribution of each cell cycle phase."""
        phases = ['G1', 'S', 'G2', 'M']
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        points = self.model.mesh.points
        x, y = points[:, 0], points[:, 1]

        for phase, ax in zip(phases, axes.ravel()):
            pop = self.model.cell_populations.populations[phase]
            sc = ax.scatter(x, y, c=pop, cmap='viridis', s=8,
                            vmin=0, vmax=max(0.01, np.max(pop)))
            plt.colorbar(sc, ax=ax)
            ax.set_title(f'{phase} Phase')
            ax.set_aspect('equal')

        plt.tight_layout()
        return fig

    def plot_metrics_history(self,
                             times: np.ndarray,
                             metrics: List[Dict]) -> plt.Figure:
        """Plot evolution of key metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Tumor mass
        ax = axes[0, 0]
        masses = [m['tumor']['total_mass'] for m in metrics]
        ax.plot(times[:len(masses)], masses, 'b-', linewidth=2)
        ax.set_ylabel('Total Tumor Mass')
        ax.set_xlabel('Time (days)')
        ax.set_title('Tumor Growth')
        ax.grid(True, alpha=0.3)

        # Hypoxic fraction
        ax = axes[0, 1]
        hypoxic = [m['tumor']['hypoxic_fraction'] for m in metrics]
        ax.plot(times[:len(hypoxic)], hypoxic, 'r-', linewidth=2)
        ax.set_ylabel('Hypoxic Fraction')
        ax.set_xlabel('Time (days)')
        ax.set_title('Hypoxia Development')
        ax.grid(True, alpha=0.3)

        # Population fractions
        ax = axes[1, 0]
        for phase, color in self.population_colors.items():
            key = f'{phase.lower()}_fraction'
            if key in metrics[0]['cell_populations']:
                fractions = [m['cell_populations'][key] for m in metrics]
                ax.plot(times[:len(fractions)], fractions,
                        color=color, label=phase, linewidth=2)
        ax.set_ylabel('Population Fraction')
        ax.set_xlabel('Time (days)')
        ax.set_title('Cell Cycle Distribution')
        ax.legend(ncol=3)
        ax.grid(True, alpha=0.3)

        # Immune response
        ax = axes[1, 1]
        immune_total = [m['immune']['total_immune_cells'] for m in metrics]
        ax.plot(times[:len(immune_total)], immune_total,
                'g-', linewidth=2, label='Immune cells')
        ax.set_ylabel('Total Immune Cells')
        ax.set_xlabel('Time (days)')
        ax.set_title('Immune Response')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 150):
        """Save figure to output directory."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved {filepath}")
