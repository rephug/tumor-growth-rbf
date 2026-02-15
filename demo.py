#!/usr/bin/env python3
"""
Tumor Growth RBF-FD Simulator — Learning Demo
==============================================

This demo walks through the key concepts of computational oncology
using the meshless RBF-FD framework. Each section builds on the previous
one and includes explanations of the biology and math.

Run: python demo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from tumor_growth_rbf import (
    TumorModel, TumorParameters,
    CellPopulationModel, CellCycleParameters,
    TreatmentModule, TreatmentParameters,
    ImmuneResponse, ImmuneParameters,
    TumorVisualizer, MeshHandler, RBFSolver
)

OUTPUT_DIR = Path("demo_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def demo_1_mesh_and_rbf():
    """
    DEMO 1: Understanding the Meshless Foundation
    =============================================

    Traditional PDE solvers use regular grids (like pixels in an image).
    RBF-FD uses scattered points instead. This section shows why and how.
    """
    print("=" * 60)
    print("DEMO 1: Meshless Points and RBF-FD")
    print("=" * 60)

    # Create a mesh handler
    mesh = MeshHandler(domain_size=(10.0, 10.0))

    # Compare different point distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, dist_type in zip(axes, ["grid", "random", "halton"]):
        mesh.initialize_points(200, distribution=dist_type)
        ax.scatter(mesh.points[:, 0], mesh.points[:, 1], s=10)
        ax.set_title(f"{dist_type.capitalize()} ({len(mesh.points)} points)")
        ax.set_aspect("equal")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Show neighbor connections for a few points
        for i in range(0, len(mesh.points), 40):
            for j in mesh.neighbor_lists[i][:5]:
                ax.plot([mesh.points[i, 0], mesh.points[j, 0]],
                        [mesh.points[i, 1], mesh.points[j, 1]],
                        'r-', alpha=0.15, linewidth=0.5)

    fig.suptitle("Point Distributions for Meshless Methods\n"
                 "(red lines show neighbor connections)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_point_distributions.png", dpi=150)
    plt.close()

    # Show RBF-FD operator accuracy
    print("\nRBF-FD Accuracy Test:")
    print("  Testing Laplacian of f(x,y) = x² + y² (exact Laplacian = 4)")

    rbf = RBFSolver(epsilon=1.0, poly_degree=2)
    mesh.initialize_points(500, distribution="halton")

    # Test function: f = x² + y²  →  ∇²f = 4
    f_values = mesh.points[:, 0] ** 2 + mesh.points[:, 1] ** 2
    L = rbf.assemble_global_operator(
        mesh.points, mesh.neighbor_lists, "laplacian"
    )
    laplacian_f = L @ f_values

    # Interior points should give ≈4 (boundary points may be off)
    interior = ((mesh.points[:, 0] > 1) & (mesh.points[:, 0] < 9) &
                (mesh.points[:, 1] > 1) & (mesh.points[:, 1] < 9))
    mean_error = np.mean(np.abs(laplacian_f[interior] - 4.0))
    print(f"  Mean absolute error (interior): {mean_error:.6f}")
    print(f"  This should be very small — RBF-FD exactly reproduces "
          f"polynomials up to degree {rbf.poly_degree}")


def demo_2_cell_cycle():
    """
    DEMO 2: Cell Cycle Biology
    ==========================

    Tumor cells go through phases: G1 → S → G2 → M → (division) → 2×G1
    Understanding this cycle is crucial because:
    - Different treatments target different phases
    - Phase distribution determines treatment timing
    - Quiescence (G0) is the main cause of treatment resistance
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Cell Cycle Dynamics")
    print("=" * 60)

    # Create standalone cell population model
    params = CellCycleParameters()
    pop_model = CellPopulationModel(params)
    pop_model.initialize((100,))  # 100 spatial points

    # Start all cells in G1
    pop_model.populations['G1'][:] = 1.0

    # Track phase fractions over time
    dt = 0.05  # days (= 1.2 hours)
    times = []
    fractions = {phase: [] for phase in ['G1', 'S', 'G2', 'M']}

    print("\nSimulating cell cycle under normal oxygen (1.0):")
    normal_oxygen = np.ones(100)

    for step in range(500):  # ~25 days
        pop_model.update(dt, normal_oxygen)
        if step % 5 == 0:
            times.append(step * dt)
            total = pop_model.get_total_density()
            for phase in fractions:
                frac = np.mean(pop_model.populations[phase] / (total + 1e-10))
                fractions[phase].append(frac)

    # Plot phase evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for phase, values in fractions.items():
        ax1.plot(times, values, label=phase, linewidth=2)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Phase Fraction")
    ax1.set_title("Cell Cycle Phase Distribution Over Time\n"
                   "(Starting from 100% G1)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Show total cell growth (should be exponential)
    totals = [pop_model.get_total_density()[0]]  # Track one point
    pop_model2 = CellPopulationModel(params)
    pop_model2.initialize((1,))
    pop_model2.populations['G1'][:] = 1.0
    growth_times = [0]
    for step in range(600):
        pop_model2.update(dt, np.ones(1))
        growth_times.append((step + 1) * dt)
        totals.append(pop_model2.get_total_density()[0])

    ax2.semilogy(growth_times, totals, 'b-', linewidth=2)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Total Cell Density (log scale)")
    ax2.set_title("Exponential Growth from Cell Division\n"
                   "(M phase → 2 daughter cells in G1)")
    ax2.grid(True, alpha=0.3)

    target = totals[0] * 2
    idx = np.searchsorted(totals, target)
    if idx < len(growth_times):
        doubling_time = growth_times[idx]
        ax2.axhline(y=target, color='r', linestyle='--', alpha=0.5)
        ax2.text(doubling_time + 0.5, target * 1.1,
                 f"Doubling time ≈ {doubling_time:.1f} days",
                 color='r', fontsize=10)
    else:
        doubling_time = None

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_cell_cycle.png", dpi=150)
    plt.close()
    print(f"  Doubling time: {doubling_time if doubling_time else 'N/A (extend simulation)'}")
    print(f"  Steady-state phase distribution:")
    for phase in ['G1', 'S', 'G2', 'M']:
        print(f"    {phase}: {fractions[phase][-1]:.1%}")


def demo_3_hypoxia():
    """
    DEMO 3: Oxygen and Hypoxia
    ==========================

    As tumors grow, the core becomes oxygen-starved (hypoxic).
    This creates the classic layered tumor structure:
        [proliferating rim] → [quiescent zone] → [necrotic core]

    Hypoxia also makes tumors resistant to radiation.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Hypoxia and Tumor Structure")
    print("=" * 60)

    # Run a full simulation and watch hypoxia develop
    model = TumorModel(
        domain_size=(10.0, 10.0),
        n_initial_points=400
    )

    dt = 0.1
    times = []
    metrics_list = []

    print("\nSimulating tumor growth with oxygen dynamics:")
    for step in range(100):
        model.update(dt)
        if step % 5 == 0:
            times.append(step * dt)
            m = model.get_metrics()
            metrics_list.append(m)
            if step % 20 == 0:
                print(f"  Day {step*dt:.1f}: "
                      f"mass={m['tumor']['total_mass']:.1f}, "
                      f"hypoxic={m['tumor']['hypoxic_fraction']:.1%}, "
                      f"necrotic={m['cell_populations']['necrotic_fraction']:.1%}")

    # Create hypoxia development visualization
    viz = TumorVisualizer(model)
    fig = viz.plot_metrics_history(np.array(times), metrics_list)
    fig.savefig(OUTPUT_DIR / "03_hypoxia_development.png", dpi=150)
    plt.close()

    fig = viz.create_state_visualization(time=10.0)
    fig.savefig(OUTPUT_DIR / "03_tumor_state.png", dpi=150)
    plt.close()

    print(f"\nFinal state:")
    final = metrics_list[-1]
    print(f"  Tumor mass: {final['tumor']['total_mass']:.1f}")
    print(f"  Hypoxic fraction: {final['tumor']['hypoxic_fraction']:.1%}")
    print(f"  Necrotic fraction: {final['cell_populations']['necrotic_fraction']:.1%}")


def demo_4_radiation():
    """
    DEMO 4: Radiation Therapy
    =========================

    The Linear-Quadratic model is the workhorse of radiation biology:
        Survival = exp(-α*D - β*D²)

    Key insights:
    - α term: lethal single-hit damage (dominates at low doses)
    - β term: accumulation damage (dominates at high doses)
    - α/β ratio ≈ 10 Gy for tumors, ≈ 3 Gy for late-responding tissues
    - This is why FRACTIONATION works: multiple small doses spare
      normal tissue more than a single large dose
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Radiation Therapy (Linear-Quadratic Model)")
    print("=" * 60)

    # Show LQ model survival curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    doses = np.linspace(0, 10, 100)
    params = TreatmentParameters()

    # Survival curves for different phases
    phase_factors = {
        'G1': params.radiation_g1_factor,
        'S (resistant)': params.radiation_s_factor,
        'G2': params.radiation_g2_factor,
        'M (sensitive)': params.radiation_m_factor,
    }

    colors = {'G1': '#2ecc71', 'S (resistant)': '#e74c3c',
              'G2': '#3498db', 'M (sensitive)': '#f1c40f'}

    for phase, factor in phase_factors.items():
        alpha = params.fractionation_alpha * factor
        beta = params.fractionation_beta * factor
        survival = np.exp(-alpha * doses - beta * doses ** 2)
        ax1.semilogy(doses, survival, label=phase, linewidth=2,
                     color=colors[phase])

    ax1.set_xlabel("Dose (Gy)")
    ax1.set_ylabel("Survival Fraction")
    ax1.set_title("LQ Model: Phase-Specific Radiosensitivity\n"
                   "S(D) = exp(-αD - βD²)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-3, 1.1)

    # Fractionation comparison
    total_dose = 60.0  # Standard total dose

    fractions_list = [1, 2, 5, 10, 30]
    alpha = params.fractionation_alpha
    beta = params.fractionation_beta

    for n_frac in fractions_list:
        dose_per_frac = total_dose / n_frac
        cumulative_survival = []
        current_survival = 1.0
        cum_doses = [0]

        for i in range(n_frac):
            current_survival *= np.exp(-alpha * dose_per_frac -
                                       beta * dose_per_frac ** 2)
            cumulative_survival.append(current_survival)
            cum_doses.append((i + 1) * dose_per_frac)

        ax2.semilogy(cum_doses[1:], cumulative_survival,
                     'o-', label=f'{n_frac} fractions × {dose_per_frac:.1f} Gy',
                     markersize=3)

    ax2.set_xlabel("Cumulative Dose (Gy)")
    ax2.set_ylabel("Survival Fraction")
    ax2.set_title(f"Fractionation Effect (Total: {total_dose} Gy)\n"
                   "More fractions → better normal tissue sparing")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_radiation_lq_model.png", dpi=150)
    plt.close()

    # Simulate actual treatment
    print("\nSimulating radiation treatment course:")
    model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=300)

    # Grow tumor for 5 days
    for step in range(50):
        model.update(0.1)

    pre_mass = model.get_metrics()['tumor']['total_mass']
    print(f"  Pre-treatment tumor mass: {pre_mass:.1f}")

    # Apply 5 fractions of 2 Gy
    for frac in range(5):
        result = model.apply_treatment("radiation", dose=2.0)
        model.update(0.1)  # One day between fractions
        m = model.get_metrics()
        print(f"  Fraction {frac+1}: killed {result['total_cells_killed']:.1f} cells, "
              f"mass now {m['tumor']['total_mass']:.1f}")

    post_mass = model.get_metrics()['tumor']['total_mass']
    print(f"  Post-treatment mass: {post_mass:.1f} "
          f"({(1 - post_mass/pre_mass)*100:.0f}% reduction)")


def demo_5_combined_treatment():
    """
    DEMO 5: Combined Treatment Modalities
    ======================================

    Real cancer treatment often combines multiple modalities:
    - Radiation: kills cells by DNA damage
    - Chemotherapy: kills dividing cells (especially S-phase)
    - Immunotherapy: boosts immune system to fight tumor

    The timing and combination matters enormously for outcomes.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Combined Treatment Simulation")
    print("=" * 60)

    model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=300)

    dt = 0.1
    times = []
    metrics_list = []
    treatment_events = []

    print("\nPhase 1: Tumor growth (days 0-5)")
    for step in range(50):
        model.update(dt)
        times.append(step * dt)
        metrics_list.append(model.get_metrics())

    print(f"  Mass at day 5: {metrics_list[-1]['tumor']['total_mass']:.1f}")

    print("\nPhase 2: Radiation + chemo (days 5-10)")
    for step in range(50):
        model.update(dt)
        day = 5.0 + step * dt
        times.append(day)

        # Radiation: daily fractions
        if step % 10 == 0:  # Every day
            result = model.apply_treatment("radiation", dose=2.0)
            treatment_events.append({'time': day, 'type': 'radiation'})

        # Chemo: every 3 days
        if step % 30 == 0:
            result = model.apply_treatment("chemo", drug_amount=0.5)
            treatment_events.append({'time': day, 'type': 'chemo'})

        metrics_list.append(model.get_metrics())

    print(f"  Mass at day 10: {metrics_list[-1]['tumor']['total_mass']:.1f}")

    print("\nPhase 3: Immunotherapy + recovery (days 10-20)")
    for step in range(100):
        model.update(dt)
        day = 10.0 + step * dt
        times.append(day)

        # Immunotherapy every 5 days
        if step % 50 == 0:
            model.apply_treatment("immunotherapy", boost_factor=2.0)
            treatment_events.append({'time': day, 'type': 'immunotherapy'})

        metrics_list.append(model.get_metrics())

    print(f"  Mass at day 20: {metrics_list[-1]['tumor']['total_mass']:.1f}")

    # Plot results
    viz = TumorVisualizer(model)
    fig = viz.plot_metrics_history(np.array(times), metrics_list)

    # Add treatment markers to the first subplot
    ax = fig.axes[0]
    for event in treatment_events:
        color = {'radiation': 'red', 'chemo': 'blue',
                 'immunotherapy': 'green'}[event['type']]
        ax.axvline(x=event['time'], color=color, linestyle='--',
                   alpha=0.3, linewidth=1)

    fig.savefig(OUTPUT_DIR / "05_combined_treatment.png", dpi=150)
    plt.close()


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  Tumor Growth RBF-FD Simulator — Demo       ║")
    print("║  A learning tool for computational oncology  ║")
    print("╚══════════════════════════════════════════════╝\n")

    demo_1_mesh_and_rbf()
    demo_2_cell_cycle()
    demo_3_hypoxia()
    demo_4_radiation()
    demo_5_combined_treatment()

    print("\n" + "=" * 60)
    print(f"All demos complete! Figures saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
