"""
visualization.py

Comprehensive visualization tools for tumor growth model analysis.
Handles multi-modal visualization of tumor growth, tissue properties,
cell populations, and treatment responses. Designed to work with
medical imaging data and produce publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TumorVisualizer:
    """
    Advanced visualization tools for tumor growth analysis.
    
    Features:
    - Multi-modal visualization of tumor state
    - Cell population distribution plots
    - Tissue-specific analysis
    - Treatment response visualization
    - Animation capabilities
    - Medical image overlay support
    """
    
    def __init__(self, 
                 model,
                 figsize: Tuple[int, int] = (10, 8),
                 cmap_tumor: str = 'hot',
                 cmap_oxygen: str = 'RdYlBu_r',
                 output_dir: Optional[str] = None):
        """
        Initialize visualizer with model and display preferences.
        
        Args:
            model: TumorModel instance
            figsize: Default figure size
            cmap_tumor: Colormap for tumor density
            cmap_oxygen: Colormap for oxygen distribution
            output_dir: Directory for saving figures
        """
        self.model = model
        self.figsize = figsize
        self.cmap_tumor = cmap_tumor
        self.cmap_oxygen = cmap_oxygen
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Create custom colormaps for different visualizations
        self.tissue_colors = {
            'white_matter': '#E6E6E6',
            'gray_matter': '#808080',
            'csf': '#87CEEB',
            'vessel': '#FF0000',
            'necrotic': '#4A0000'
        }
        
        # Cell population colors
        self.population_colors = {
            'G1': '#2ecc71',  # Green for growth
            'S': '#e74c3c',   # Red for synthesis
            'G2': '#3498db',  # Blue for gap 2
            'M': '#f1c40f',   # Yellow for mitosis
            'Q': '#95a5a6',   # Gray for quiescent
            'N': '#2c3e50'    # Dark blue for necrotic
        }
        
    def create_state_visualization(self,
                                 time: float,
                                 show_oxygen: bool = True,
                                 show_vessels: bool = True,
                                 show_tissue: bool = True) -> plt.Figure:
        """
        Create comprehensive visualization of current tumor state.
        
        Args:
            time: Current simulation time
            show_oxygen: Whether to show oxygen distribution
            show_vessels: Whether to show vessel positions
            show_tissue: Whether to show tissue types
            
        Returns:
            matplotlib Figure with multiple subplots
        """
        n_plots = 1 + show_oxygen + show_vessels + show_tissue
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
            
        plot_idx = 0
        
        # Plot tumor density
        self._plot_tumor_density(axes[plot_idx])
        axes[plot_idx].set_title(f'Tumor Density (t={time:.1f} days)')
        plot_idx += 1
        
        # Plot oxygen if requested
        if show_oxygen:
            self._plot_oxygen_distribution(axes[plot_idx])
            axes[plot_idx].set_title('Oxygen Distribution')
            plot_idx += 1
            
        # Plot vessels if requested
        if show_vessels:
            self._plot_vessel_map(axes[plot_idx])
            axes[plot_idx].set_title('Vessel Distribution')
            plot_idx += 1
            
        # Plot tissue types if requested
        if show_tissue and hasattr(self.model, 'tissue_model'):
            self._plot_tissue_types(axes[plot_idx])
            axes[plot_idx].set_title('Tissue Types')
            
        plt.tight_layout()
        return fig
        
    def plot_cell_populations(self) -> plt.Figure:
        """Create visualization of cell population distributions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Plot spatial distribution of each population
        phases = ['G1', 'S', 'G2', 'M']
        axes = [ax1, ax2, ax3, ax4]
        
        for phase, ax in zip(phases, axes):
            pop = self.model.cell_populations.populations[phase]
            im = ax.imshow(pop, cmap=plt.get_cmap('viridis'),
                          extent=[0, self.model.domain_size[0],
                                0, self.model.domain_size[1]])
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{phase} Phase')
            
        plt.tight_layout()
        return fig
        
    def create_treatment_response_plot(self,
                                     times: np.ndarray,
                                     metrics: List[Dict],
                                     treatments: Optional[List[Dict]] = None) -> plt.Figure:
        """
        Visualize tumor response to treatment.
        
        Args:
            times: Array of time points
            metrics: List of metric dictionaries
            treatments: Optional list of treatment events
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12),
                                      height_ratios=[2, 1])
        
        # Plot total mass over time
        masses = [m['tumor']['total_mass'] for m in metrics]
        ax1.plot(times, masses, 'b-', label='Total Mass')
        
        # Plot population fractions
        for phase, color in self.population_colors.items():
            if phase in metrics[0]['cell_populations']:
                fractions = [m['cell_populations'][f'{phase.lower()}_fraction']
                           for m in metrics]
                ax2.plot(times, fractions, color=color, label=f'{phase} Phase')
                
        # Add treatment markers if provided
        if treatments:
            for treatment in treatments:
                t = treatment['time']
                ax1.axvline(x=t, color='r', linestyle='--', alpha=0.3)
                ax2.axvline(x=t, color='r', linestyle='--', alpha=0.3)
                
        ax1.set_ylabel('Total Tumor Mass')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Population Fractions')
        
        ax1.legend()
        ax2.legend()
        ax1.grid(True)
        ax2.grid(True)
        
        return fig
        
    def create_animation(self,
                        times: np.ndarray,
                        states: List[Dict],
                        interval: int = 200) -> FuncAnimation:
        """
        Create animation of tumor evolution.
        
        Args:
            times: Array of time points
            states: List of model states
            interval: Animation interval in milliseconds
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        vmin = min(np.min(state['tumor_density']) for state in states)
        vmax = max(np.max(state['tumor_density']) for state in states)
        
        im = ax.imshow(states[0]['tumor_density'], 
                      cmap=self.cmap_tumor,
                      vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        
        def update(frame):
            im.set_array(states[frame]['tumor_density'])
            ax.set_title(f'Time: {times[frame]:.1f} days')
            return [im]
            
        anim = FuncAnimation(fig, update,
                           frames=len(states),
                           interval=interval,
                           blit=True)
                           
        return anim
        
    def _plot_tumor_density(self, ax: plt.Axes):
        """Plot tumor density distribution."""
        im = ax.imshow(self.model.tumor_density,
                      extent=[0, self.model.domain_size[0],
                             0, self.model.domain_size[1]],
                      cmap=self.cmap_tumor)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Position (mm)')
        
    def _plot_oxygen_distribution(self, ax: plt.Axes):
        """Plot oxygen concentration."""
        im = ax.imshow(self.model.oxygen,
                      extent=[0, self.model.domain_size[0],
                             0, self.model.domain_size[1]],
                      cmap=self.cmap_oxygen)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Position (mm)')
        
    def _plot_vessel_map(self, ax: plt.Axes):
        """Plot vessel distribution."""
        im = ax.imshow(self.model.vessels,
                      extent=[0, self.model.domain_size[0],
                             0, self.model.domain_size[1]],
                      cmap='Reds')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Position (mm)')
        
    def _plot_tissue_types(self, ax: plt.Axes):
        """Plot tissue type distribution."""
        if not hasattr(self.model, 'tissue_model') or \
           self.model.tissue_model.tissue_map is None:
            return
            
        tissue_values = np.zeros_like(
            self.model.tissue_model.tissue_map, dtype=float
        )
        
        for tissue_type, color in self.tissue_colors.items():
            mask = self.model.tissue_model.tissue_map == tissue_type
            tissue_values[mask] = list(self.tissue_colors.keys()).index(tissue_type)
            
        im = ax.imshow(tissue_values,
                      extent=[0, self.model.domain_size[0],
                             0, self.model.domain_size[1]],
                      cmap=plt.ListedColormap(list(self.tissue_colors.values())))
                      
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=tissue)
                         for tissue, color in self.tissue_colors.items()]
        ax.legend(handles=legend_elements, loc='center left',
                 bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Position (mm)')
        
    def save_figure(self,
                   fig: plt.Figure,
                   filename: str,
                   dpi: int = 300):
        """Save figure to file."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
            
    def plot_metrics_history(self,
                           times: np.ndarray,
                           metrics: List[Dict]) -> plt.Figure:
        """
        Plot evolution of key metrics over time.
        
        Args:
            times: Array of time points
            metrics: List of metric dictionaries
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot tumor mass
        ax = axes[0, 0]
        masses = [m['tumor']['total_mass'] for m in metrics]
        ax.plot(times, masses)
        ax.set_ylabel('Total Tumor Mass')
        ax.set_xlabel('Time (days)')
        ax.grid(True)
        
        # Plot hypoxic fraction
        ax = axes[0, 1]
        hypoxic = [m['tumor']['hypoxic_fraction'] for m in metrics]
        ax.plot(times, hypoxic)
        ax.set_ylabel('Hypoxic Fraction')
        ax.set_xlabel('Time (days)')
        ax.grid(True)
        
        # Plot population distribution
        ax = axes[1, 0]
        for phase, color in self.population_colors.items():
            if phase in metrics[0]['cell_populations']:
                fractions = [m['cell_populations'][f'{phase.lower()}_fraction']
                           for m in metrics]
                ax.plot(times, fractions, color=color, label=phase)
        ax.set_ylabel('Population Fractions')
        ax.set_xlabel('Time (days)')
        ax.legend()
        ax.grid(True)
        
        # Plot treatment metrics if available
        ax = axes[1, 1]
        if 'treatment' in metrics[0]:
            treatment_metrics = [m['treatment'] for m in metrics]
            if 'cumulative_radiation' in treatment_metrics[0]:
                cumulative = [m['cumulative_radiation'] for m in treatment_metrics]
                ax.plot(times, cumulative, label='Cumulative Radiation')
            if 'mean_drug_concentration' in treatment_metrics[0]:
                drug_conc = [m['mean_drug_concentration'] for m in treatment_metrics]
                ax.plot(times, drug_conc, label='Drug Concentration')
        ax.set_ylabel('Treatment Metrics')
        ax.set_xlabel('Time (days)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig