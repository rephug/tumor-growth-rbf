#!/usr/bin/env python3
"""
Clinical Workflow Example
=========================

This example demonstrates how a computational oncology researcher
would use the simulator to compare treatment strategies for a
glioblastoma (GBM) patient.

Scenario:
    A 55-year-old patient with newly diagnosed GBM in the right
    temporal lobe. The tumor has been partially resected. We want
    to compare the standard Stupp protocol against a hypofractionated
    regimen and explore adding immunotherapy.

This example uses synthetic tissue data but shows exactly how
you would substitute real MRI-derived data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import copy

from tumor_growth_rbf import (
    TumorModel, TumorParameters,
    TissueParameters, TissueType,
    TreatmentParameters, CellCycleParameters,
    TumorVisualizer,
)

OUTPUT = Path("clinical_output")
OUTPUT.mkdir(exist_ok=True)


# ============================================================
# STEP 1: Define Patient-Specific Parameters
# ============================================================

def estimate_growth_rate(radius_scan1_mm, radius_scan2_mm, days_between):
    """
    Estimate tumor growth rate from two MRI measurements.

    Uses the Fisher-KPP wavefront velocity relation:
        v = 2 * sqrt(D * rho)

    Where v is the radial expansion rate, D is diffusion coefficient,
    and rho is the proliferation rate.
    """
    velocity = (radius_scan2_mm - radius_scan1_mm) / days_between
    D_assumed = 0.1  # mm²/day, typical white matter value

    if velocity <= 0:
        print("  Warning: Tumor appears to be shrinking between scans")
        return 0.01  # Use a default low growth rate

    rho = velocity ** 2 / (4 * D_assumed)
    doubling_time = np.log(2) / rho if rho > 0 else float('inf')

    print(f"  Radial velocity: {velocity:.3f} mm/day")
    print(f"  Estimated growth rate: {rho:.4f} day⁻¹")
    print(f"  Estimated doubling time: {doubling_time:.0f} days")

    return rho


def create_synthetic_brain_tissue(size, n_points):
    """
    Create synthetic brain tissue map for demonstration.

    In practice, you would replace this with:
        tissue_image = nibabel.load("segmentation.nii.gz").get_fdata()[:,:,slice_idx]

    This synthetic version creates a simplified brain cross-section with:
    - White matter core
    - Gray matter cortex
    - CSF-filled ventricles
    - A few vessel regions
    """
    # Create tissue image (integer labels)
    tissue_image = np.ones((n_points, n_points), dtype=int)  # Default: gray matter

    # White matter: central oval
    y, x = np.ogrid[:n_points, :n_points]
    center = n_points // 2
    white_matter = ((x - center) ** 2 / (center * 0.7) ** 2 +
                    (y - center) ** 2 / (center * 0.5) ** 2) < 1
    tissue_image[white_matter] = 0  # White matter label

    # Ventricles (CSF): two small regions
    vent_l = ((x - center * 0.7) ** 2 + (y - center) ** 2) < (center * 0.1) ** 2
    vent_r = ((x - center * 1.3) ** 2 + (y - center) ** 2) < (center * 0.1) ** 2
    tissue_image[vent_l | vent_r] = 2  # CSF label

    tissue_labels = {
        0: TissueType.WHITE_MATTER,
        1: TissueType.GRAY_MATTER,
        2: TissueType.CSF,
    }

    # Vessel map: scattered vessel points
    vessel_image = np.zeros((n_points, n_points), dtype=bool)
    rng = np.random.RandomState(42)
    vessel_points = rng.randint(0, n_points, size=(50, 2))
    for vp in vessel_points:
        vessel_image[max(0, vp[0]-1):vp[0]+2, max(0, vp[1]-1):vp[1]+2] = True

    return tissue_image, tissue_labels, vessel_image


# ============================================================
# STEP 2: Build Patient Model
# ============================================================

def create_patient_model(growth_rate):
    """Create a TumorModel configured for this patient."""

    params = TumorParameters(
        growth_rate=growth_rate,
        carrying_capacity=1.0,
        diffusion_white=0.1,     # mm²/day
        diffusion_grey=0.01,     # mm²/day
        oxygen_consumption=0.1,
        oxygen_diffusion=1.0,
        hypoxia_threshold=0.1,
        min_spacing=0.1,
        max_spacing=1.0,
        refinement_threshold=0.1,
    )

    # GBM-specific treatment parameters
    treatment_params = TreatmentParameters(
        fractionation_alpha=0.15,   # α for GBM (α/β ≈ 10 Gy)
        fractionation_beta=0.015,   # β
        oxygen_enhancement=2.5,     # OER for GBM
        chemo_sensitivity=0.15,     # TMZ sensitivity
    )

    model = TumorModel(
        domain_size=(50.0, 50.0),   # 50mm × 50mm field
        params=params,
        treatment_params=treatment_params,
        n_initial_points=500,
    )

    return model


# ============================================================
# STEP 3: Simulate Treatment Arms
# ============================================================

def simulate_growth_only(model, days, dt=0.1):
    """Simulate tumor growth without treatment (natural history)."""
    n_steps = int(days / dt)
    times = []
    metrics = []

    for step in range(n_steps):
        model.update(dt)
        if step % max(1, n_steps // 50) == 0:
            times.append(step * dt)
            metrics.append(model.get_metrics())

    return np.array(times), metrics


def simulate_stupp_protocol(model, pre_growth_days=5, dt=0.1):
    """
    Stupp protocol (standard of care for GBM):
    - 6 weeks concurrent chemoradiation: 60 Gy / 30 fx + daily TMZ
    - Then 6 cycles adjuvant TMZ (5/28 days)

    Reference: Stupp et al., NEJM 2005
    """
    times = []
    metrics = []
    treatment_log = []

    # Pre-treatment growth
    for step in range(int(pre_growth_days / dt)):
        model.update(dt)

    day = 0
    total_days = 300  # Simulate ~10 months

    # Phase 1: Concurrent chemoradiation (6 weeks = 42 days)
    for step in range(int(42 / dt)):
        model.update(dt)
        day = step * dt

        week = int(day) // 7
        day_of_week = int(day) % 7

        # Radiation: 2 Gy Mon-Fri for 6 weeks
        if day_of_week < 5 and abs(day - round(day)) < dt / 2:
            result = model.apply_treatment("radiation", dose=2.0)
            treatment_log.append({
                'day': day, 'type': 'radiation',
                'dose': 2.0, 'killed': result['total_cells_killed']
            })

        # Concurrent TMZ: daily
        if abs(day - round(day)) < dt / 2:
            model.apply_treatment("chemo", drug_amount=0.2, duration=0.5)

        if step % max(1, int(2 / dt)) == 0:
            times.append(pre_growth_days + day)
            metrics.append(model.get_metrics())

    # Phase 2: Adjuvant TMZ (6 cycles, 28 days each, TMZ days 1-5)
    offset = 42.0
    for cycle in range(6):
        for step in range(int(28 / dt)):
            model.update(dt)
            day = offset + cycle * 28 + step * dt
            day_in_cycle = step * dt

            if day_in_cycle < 5 and abs(day_in_cycle - round(day_in_cycle)) < dt / 2:
                model.apply_treatment("chemo", drug_amount=0.3, duration=0.5)

            if step % max(1, int(5 / dt)) == 0:
                times.append(pre_growth_days + day)
                metrics.append(model.get_metrics())

    # Post-treatment follow-up
    remaining = total_days - (42 + 6 * 28)
    for step in range(int(max(0, remaining) / dt)):
        model.update(dt)
        if step % max(1, int(5 / dt)) == 0:
            times.append(pre_growth_days + 42 + 168 + step * dt)
            metrics.append(model.get_metrics())

    return np.array(times[:len(metrics)]), metrics, treatment_log


def simulate_hypofractionated(model, pre_growth_days=5, dt=0.1):
    """
    Hypofractionated regimen:
    - 40 Gy in 15 fractions (2.67 Gy/fx) over 3 weeks
    - Then adjuvant TMZ

    Used for elderly patients or those with poor performance status.
    """
    times = []
    metrics = []

    for step in range(int(pre_growth_days / dt)):
        model.update(dt)

    day = 0

    # Hypofractionated radiation: 3 weeks
    for step in range(int(21 / dt)):
        model.update(dt)
        day = step * dt

        day_of_week = int(day) % 7
        if day_of_week < 5 and abs(day - round(day)) < dt / 2:
            model.apply_treatment("radiation", dose=2.67)

        if step % max(1, int(2 / dt)) == 0:
            times.append(pre_growth_days + day)
            metrics.append(model.get_metrics())

    # Adjuvant TMZ: 6 cycles
    for cycle in range(6):
        for step in range(int(28 / dt)):
            model.update(dt)
            day_in_cycle = step * dt

            if day_in_cycle < 5 and abs(day_in_cycle - round(day_in_cycle)) < dt / 2:
                model.apply_treatment("chemo", drug_amount=0.3, duration=0.5)

            if step % max(1, int(5 / dt)) == 0:
                times.append(pre_growth_days + 21 + cycle * 28 + step * dt)
                metrics.append(model.get_metrics())

    # Follow-up
    for step in range(int(90 / dt)):
        model.update(dt)
        if step % max(1, int(5 / dt)) == 0:
            times.append(pre_growth_days + 21 + 168 + step * dt)
            metrics.append(model.get_metrics())

    return np.array(times[:len(metrics)]), metrics


def simulate_stupp_plus_immunotherapy(model, pre_growth_days=5, dt=0.1):
    """
    Stupp protocol + immunotherapy (experimental).
    Adds checkpoint inhibitor (anti-PD1) every 2 weeks starting week 3.
    """
    times = []
    metrics = []

    for step in range(int(pre_growth_days / dt)):
        model.update(dt)

    day = 0

    # Concurrent chemoradiation + immunotherapy
    for step in range(int(42 / dt)):
        model.update(dt)
        day = step * dt

        day_of_week = int(day) % 7
        if day_of_week < 5 and abs(day - round(day)) < dt / 2:
            model.apply_treatment("radiation", dose=2.0)

        if abs(day - round(day)) < dt / 2:
            model.apply_treatment("chemo", drug_amount=0.2, duration=0.5)

        # Immunotherapy every 14 days starting day 14
        if day >= 14 and abs(day % 14) < dt:
            model.apply_treatment("immunotherapy", boost_factor=2.0)

        if step % max(1, int(2 / dt)) == 0:
            times.append(pre_growth_days + day)
            metrics.append(model.get_metrics())

    # Adjuvant TMZ + continued immunotherapy
    for cycle in range(6):
        for step in range(int(28 / dt)):
            model.update(dt)
            day_in_cycle = step * dt
            day = 42 + cycle * 28 + day_in_cycle

            if day_in_cycle < 5 and abs(day_in_cycle - round(day_in_cycle)) < dt / 2:
                model.apply_treatment("chemo", drug_amount=0.3, duration=0.5)

            if abs(day_in_cycle % 14) < dt:
                model.apply_treatment("immunotherapy", boost_factor=2.0)

            if step % max(1, int(5 / dt)) == 0:
                times.append(pre_growth_days + day)
                metrics.append(model.get_metrics())

    # Follow-up with maintenance immunotherapy
    for step in range(int(90 / dt)):
        model.update(dt)
        day = step * dt
        if abs(day % 14) < dt:
            model.apply_treatment("immunotherapy", boost_factor=1.5)

        if step % max(1, int(5 / dt)) == 0:
            times.append(pre_growth_days + 210 + step * dt)
            metrics.append(model.get_metrics())

    return np.array(times[:len(metrics)]), metrics


# ============================================================
# STEP 4: Analysis and Visualization
# ============================================================

def compare_arms(results_dict):
    """Create comparison plots for all treatment arms."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        'Natural history': '#666666',
        'Stupp (standard)': '#2196F3',
        'Hypofractionated': '#FF9800',
        'Stupp + immunotherapy': '#4CAF50',
    }

    # Panel 1: Tumor mass over time
    ax = axes[0, 0]
    for name, (times, metrics) in results_dict.items():
        masses = [m['tumor']['total_mass'] for m in metrics]
        ax.plot(times[:len(masses)], masses,
                color=colors.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Total Tumor Mass')
    ax.set_title('Tumor Volume Over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Hypoxic fraction
    ax = axes[0, 1]
    for name, (times, metrics) in results_dict.items():
        hypoxic = [m['tumor']['hypoxic_fraction'] for m in metrics]
        ax.plot(times[:len(hypoxic)], hypoxic,
                color=colors.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Hypoxic Fraction')
    ax.set_title('Hypoxia Development')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Proliferating fraction
    ax = axes[1, 0]
    for name, (times, metrics) in results_dict.items():
        prolif = [m['cell_populations']['proliferating_fraction'] for m in metrics]
        ax.plot(times[:len(prolif)], prolif,
                color=colors.get(name, 'gray'), label=name, linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Proliferating Fraction')
    ax.set_title('Proliferating Cell Fraction')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary bar chart at end of simulation
    ax = axes[1, 1]
    names = list(results_dict.keys())
    final_masses = [results_dict[n][1][-1]['tumor']['total_mass'] for n in names]
    bar_colors = [colors.get(n, 'gray') for n in names]
    bars = ax.barh(range(len(names)), final_masses, color=bar_colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Final Tumor Mass')
    ax.set_title('Treatment Comparison (Final Mass)')
    ax.grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    if final_masses[0] > 0:
        for i, (bar, mass) in enumerate(zip(bars, final_masses)):
            if i > 0:
                reduction = (1 - mass / final_masses[0]) * 100
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{reduction:+.0f}%', va='center', fontsize=9)

    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CLINICAL WORKFLOW: GBM Treatment Comparison")
    print("=" * 60)

    # Step 1: Estimate growth parameters from clinical data
    print("\n--- Step 1: Growth Rate Estimation ---")
    print("  Pre-op MRI: tumor radius ≈ 18 mm")
    print("  6-month prior MRI: tumor radius ≈ 12 mm")
    growth_rate = estimate_growth_rate(
        radius_scan1_mm=12.0,
        radius_scan2_mm=18.0,
        days_between=180
    )

    # Step 2: Create base model
    print("\n--- Step 2: Creating Patient Model ---")
    base_model = create_patient_model(growth_rate)
    print(f"  Domain: {base_model.domain_size[0]:.0f} × "
          f"{base_model.domain_size[1]:.0f} mm")
    print(f"  Points: {len(base_model.mesh.points)}")
    print(f"  Growth rate: {growth_rate:.4f} day⁻¹")

    # Step 3: Run treatment arms
    print("\n--- Step 3: Simulating Treatment Arms ---")

    print("\n  Arm 1: Natural history (no treatment)...")
    model_natural = copy.deepcopy(base_model)
    t_nat, m_nat = simulate_growth_only(model_natural, days=100)
    print(f"    Final mass: {m_nat[-1]['tumor']['total_mass']:.1f}")

    print("\n  Arm 2: Stupp protocol (60 Gy/30 fx + TMZ)...")
    model_stupp = copy.deepcopy(base_model)
    t_stupp, m_stupp, log_stupp = simulate_stupp_protocol(model_stupp)
    print(f"    Radiation fractions delivered: "
          f"{sum(1 for e in log_stupp if e['type'] == 'radiation')}")
    print(f"    Total cells killed by radiation: "
          f"{sum(e['killed'] for e in log_stupp):.0f}")
    print(f"    Final mass: {m_stupp[-1]['tumor']['total_mass']:.1f}")

    print("\n  Arm 3: Hypofractionated (40 Gy/15 fx + TMZ)...")
    model_hypo = copy.deepcopy(base_model)
    t_hypo, m_hypo = simulate_hypofractionated(model_hypo)
    print(f"    Final mass: {m_hypo[-1]['tumor']['total_mass']:.1f}")

    print("\n  Arm 4: Stupp + immunotherapy...")
    model_immuno = copy.deepcopy(base_model)
    t_immuno, m_immuno = simulate_stupp_plus_immunotherapy(model_immuno)
    print(f"    Final mass: {m_immuno[-1]['tumor']['total_mass']:.1f}")

    # Step 4: Compare results
    print("\n--- Step 4: Generating Comparison Report ---")

    results = {
        'Natural history': (t_nat, m_nat),
        'Stupp (standard)': (t_stupp, m_stupp),
        'Hypofractionated': (t_hypo, m_hypo),
        'Stupp + immunotherapy': (t_immuno, m_immuno),
    }

    fig = compare_arms(results)
    fig.savefig(OUTPUT / "treatment_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison to {OUTPUT}/treatment_comparison.png")

    # Final state visualization
    viz = TumorVisualizer(model_stupp)
    fig2 = viz.create_state_visualization(time=300)
    fig2.savefig(OUTPUT / "final_state_stupp.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved final state to {OUTPUT}/final_state_stupp.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    nat_mass = m_nat[-1]['tumor']['total_mass']
    for name, (times, metrics) in results.items():
        final = metrics[-1]['tumor']['total_mass']
        if name == 'Natural history':
            print(f"  {name:30s}: mass = {final:.1f} (reference)")
        else:
            reduction = (1 - final / nat_mass) * 100
            print(f"  {name:30s}: mass = {final:.1f} "
                  f"({reduction:+.0f}% vs natural history)")

    print(f"\nNote: These are MODEL PREDICTIONS with synthetic data.")
    print(f"Real clinical decisions require validated patient-specific models.")


if __name__ == "__main__":
    main()
