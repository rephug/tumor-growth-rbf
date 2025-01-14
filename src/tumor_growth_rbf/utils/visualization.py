#!/usr/bin/env python3
"""
visualization.py

Visualization utilities for tumor growth simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TumorVisualizer:
    """
    Handles visualization of tumor simulation results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        
    def plot_density(self,
                    tumor_density: np.ndarray,
                    title: str = "Tumor Density",
                    show_colorbar: bool = True) -> plt.Figure:
        """
        Plot tumor density distribution.
        
        Args:
            tumor_density: 2D array of tumor density values
            title: Plot title
            show_colorbar: Whether to show colorbar
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(tumor_density, cmap='hot', interpolation='nearest')
        
        if show_colorbar:
            plt.colorbar(im, ax=ax, label='Density')
            
        ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        return fig
        
    def plot_treatment_response(self,
                              time_points: np.ndarray,
                              metrics: List[dict],
                              treatments: Optional[List[dict]] = None) -> plt.Figure:
        """
        Plot tumor response to treatment.
        
        Args:
            time_points: Array of time points
            metrics: List of metric dictionaries for each time point
            treatments: Optional list of treatment events
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot total mass over time
        masses = [m['total_mass'] for m in metrics]
        ax.plot(time_points, masses, 'b-', label='Tumor Mass')
        
        # Add treatment markers if provided
        if treatments:
            for treatment in treatments:
                ax.axvline(x=treatment['time'], color='r', linestyle='--', alpha=0.3)
                ax.text(treatment['time'], ax.get_ylim()[1],
                       f"{treatment['type']}\n{treatment['intensity']:.1f}",
                       rotation=90, ha='right')
                
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Total Tumor Mass')
        ax.set_title('Treatment Response')
        ax.grid(True)
        
        if treatments:
            ax.legend(['Tumor Mass', 'Treatment'])
            
        return fig
        
    def create_animation(self,
                        tumor_states: List[np.ndarray],
                        time_points: np.ndarray,
                        interval: int = 200) -> FuncAnimation:
        """
        Create animation of tumor evolution.
        
        Args:
            tumor_states: List of tumor density arrays
            time_points: Array of time points
            interval: Animation interval in milliseconds
            
        Returns:
            matplotlib Animation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        vmin = min(state.min() for state in tumor_states)
        vmax = max(state.max() for state in tumor_states)
        
        im = ax.imshow(tumor_states[0], cmap='hot',
                      interpolation='nearest',
                      vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Density')
        
        def update(frame):
            im.set_array(tumor_states[frame])
            ax.set_title(f'Time: {time_points[frame]:.1f} days')
            return [im]
            
        anim = FuncAnimation(fig, update,
                           frames=len(tumor_states),
                           interval=interval,
                           blit=True)
                           
        return anim
        
    def plot_spatial_analysis(self,
                            tumor_density: np.ndarray,
                            oxygen: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create detailed spatial analysis plots.
        
        Args:
            tumor_density: 2D array of tumor density
            oxygen: Optional 2D array of oxygen concentration
            
        Returns:
            matplotlib Figure object
        """
        if oxygen is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
        # Tumor density
        im1 = ax1.imshow(tumor_density, cmap='hot')
        plt.colorbar(im1, ax=ax1, label='Density')
        ax1.set_title('Tumor Density')
        
        # Radial profile
        radial_profile = self._compute_radial_profile(tumor_density)
        ax2.plot(radial_profile)
        ax2.set_xlabel('Radius (pixels)')
        ax2.set_ylabel('Average Density')
        ax2.set_title('Radial Density Profile')
        
        # Oxygen overlay if provided
        if oxygen is not None:
            im3 = ax3.imshow(oxygen, cmap='Blues')
            plt.colorbar(im3, ax=ax3, label='Oxygen')
            ax3.contour(tumor_density, colors='r', alpha=0.5)
            ax3.set_title('Oxygen with Tumor Contours')
            
        plt.tight_layout()
        return fig
        
    @staticmethod
    def _compute_radial_profile(data: np.ndarray) -> np.ndarray:
        """Compute radial profile of 2D data."""
        y, x = np.indices(data.shape)
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

def create_multi_panel_figure(model,
                            time_points: List[float],
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create comprehensive multi-panel figure of simulation results.
    
    Args:
        model: TumorModel instance
        time_points: List of time points to show
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # Tumor evolution
    ax1 = fig.add_subplot(gs[0, :2])
    masses = [model.get_metrics()['total_mass'] for _ in time_points]
    ax1.plot(time_points, masses)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Total Tumor Mass')
    ax1.set_title('Tumor Growth')
    
    # Final density distribution
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(model.tumor_density, cmap='hot')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Final Density Distribution')
    
    # Oxygen distribution
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(model.oxygen, cmap='Blues')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Oxygen Distribution')
    
    # Metrics over time
    ax4 = fig.add_subplot(gs[1, 1:])
    metrics = [model.get_metrics() for _ in time_points]
    for key in ['max_density', 'hypoxic_fraction']:
        values = [m[key] for m in metrics]
        ax4.plot(time_points, values, label=key)
    ax4.legend()
    ax4.set_xlabel('Time (days)')
    ax4.set_title('Evolution of Metrics')
    
    plt.tight_layout()
    return fig