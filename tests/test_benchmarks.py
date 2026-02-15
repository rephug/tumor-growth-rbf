"""
test_benchmarks.py

Phase 1C: Analytical validation benchmarks for the tumor growth simulator.

Each benchmark compares simulator output against a known analytical solution
to verify correctness of the underlying PDE discretization and treatment model.

Benchmarks:
    1. Fisher-KPP growth-diffusion coupling — mass growth rate matches ρ
    2. Pure diffusion — Gaussian broadening matches σ(t) = √(σ₀² + 2Dt)
    3. Exponential growth — density matches u₀exp(ρt) with D=0
    4. LQ cell survival — surviving fraction matches exp(-αd - βd²)
    5. Radial symmetry — centered Gaussian stays symmetric

References:
    Murray JD. "Mathematical Biology I: An Introduction." 3rd ed. Springer, 2002.
    Swanson KR. J Neurol Sci 216:1-10, 2003.
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

from tumor_growth_rbf import (
    TumorModel, TumorParameters,
    CellCycleParameters, TreatmentParameters,
    ImmuneParameters,
)


class TestBenchmarks:
    """Phase 1C: Analytical validation benchmarks (5 tests)."""

    @staticmethod
    def _frozen_cell_cycle():
        """Cell cycle parameters with all transitions disabled."""
        return CellCycleParameters(
            g1_to_s_rate=0.0,
            s_to_g2_rate=0.0,
            g2_to_m_rate=0.0,
            m_to_g1_rate=0.0,
        )

    @staticmethod
    def _silent_immune():
        """Immune parameters that produce zero effect."""
        return ImmuneParameters(
            recruitment_rate=0.0,
            killing_rate=0.0,
            chemokine_production=0.0,
            chemotaxis_strength=0.0,
        )

    @staticmethod
    def _stable_dt(model, D):
        """Compute a stable timestep for Forward Euler with diffusion coefficient D."""
        model._rebuild_operators()
        L = model._laplacian_op
        row_sums = np.array(np.abs(L).sum(axis=1)).flatten()
        max_row_sum = np.max(row_sums)
        dt_max = 2.0 / (D * max_row_sum + 1e-10)
        return min(0.01, dt_max * 0.8)

    # ------------------------------------------------------------------
    # Benchmark 1: Fisher-KPP Growth-Diffusion Coupling
    # ------------------------------------------------------------------

    def test_fisher_kpp_growth_diffusion(self):
        """
        Benchmark 1: Fisher-KPP growth-diffusion coupling.

        The Fisher-KPP equation ∂u/∂t = D∇²u + ρu(1-u) combines diffusion
        and logistic growth. When u ≪ K everywhere, diffusion conserves
        total mass while growth adds mass at rate ρ, so the total mass
        should grow as M(t) = M₀·exp(ρt). This validates that the RBF-FD
        diffusion operator conserves mass and doesn't interfere with growth.

        The asymptotic front speed v = 2√(Dρ) requires very long transients
        (hundreds of days) to converge, so we test the mass growth rate
        instead, which validates the same diffusion + growth coupling.
        """
        rho = 0.1   # growth rate (day⁻¹)
        D = 0.1     # diffusion coefficient (mm²/day)
        T_final = 10.0  # days

        params = TumorParameters(
            growth_rate=rho,
            diffusion_white=D,
            carrying_capacity=100.0,  # high K so u ≪ K throughout
            oxygen_consumption=0.0,
            hypoxia_threshold=0.0,
        )
        model = TumorModel(
            domain_size=(40.0, 40.0),  # large domain to avoid boundary effects
            params=params,
            cell_cycle_params=self._frozen_cell_cycle(),
            immune_params=self._silent_immune(),
            n_initial_points=2000,
        )

        # IC: small centered Gaussian (stays far from boundaries)
        center = np.array([20.0, 20.0])
        r_sq = np.sum((model.mesh.points - center) ** 2, axis=1)
        sigma = 2.0
        A = 0.05  # peak amplitude ≪ K=100
        density = A * np.exp(-r_sq / (2.0 * sigma ** 2))
        model.set_initial_density(density)

        M_0 = np.sum(model.tumor_density)
        M_expected = M_0 * np.exp(rho * T_final)

        # Simulate
        dt = self._stable_dt(model, D)
        n_steps = int(T_final / dt)
        for _ in range(n_steps):
            model.update(dt)

        M_final = np.sum(model.tumor_density)
        effective_rho = np.log(M_final / M_0) / T_final
        mass_error = abs(M_final - M_expected) / M_expected
        rho_error = abs(effective_rho - rho) / rho

        # Verify density stays well below K (validates the u ≪ K assumption)
        peak = np.max(model.tumor_density)

        # Diagnostics
        print(f"\n{'='*60}")
        print(f"Benchmark 1: Fisher-KPP Growth-Diffusion Coupling")
        print(f"{'='*60}")
        print(f"  Parameters: D={D}, rho={rho}, T={T_final}, K={params.carrying_capacity}")
        print(f"  Mass: initial={M_0:.4f}, final={M_final:.4f}, expected={M_expected:.4f}")
        print(f"  Mass error: {mass_error:.2%}")
        print(f"  Effective growth rate: {effective_rho:.4f} (expected {rho}), error={rho_error:.2%}")
        print(f"  Peak density: {peak:.4f} (K={params.carrying_capacity})")
        print(f"  Timestep: dt={dt:.6f}, n_steps={n_steps}")

        assert mass_error < 0.05, (
            f"Fisher-KPP mass growth error {mass_error:.2%} exceeds 5% "
            f"(M_final={M_final:.4f}, expected={M_expected:.4f})"
        )
        assert peak < params.carrying_capacity * 0.01, (
            f"Peak density {peak:.4f} too close to K={params.carrying_capacity} "
            f"— violates u ≪ K assumption"
        )

    # ------------------------------------------------------------------
    # Benchmark 2: Pure Diffusion (Gaussian Broadening)
    # ------------------------------------------------------------------

    def test_pure_diffusion_gaussian(self):
        """
        Benchmark 2: Pure diffusion broadens a Gaussian.

        For ∂u/∂t = D∇²u with Gaussian IC of variance σ₀², the solution
        at time t is a Gaussian with variance σ(t)² = σ₀² + 2Dt.
        We verify this by fitting a Gaussian to the simulated density
        profile and comparing the fitted width to the analytical prediction.
        """
        D = 0.1        # diffusion (mm²/day)
        sigma_0 = 1.0  # initial std dev (mm)
        A = 0.3        # peak amplitude (well below K)
        T_final = 2.0  # simulation time (days)

        sigma_T_sq = sigma_0 ** 2 + 2.0 * D * T_final  # 1.4
        sigma_T = np.sqrt(sigma_T_sq)  # ~1.183 mm
        peak_analytical = A * sigma_0 ** 2 / sigma_T_sq

        params = TumorParameters(
            growth_rate=0.0,
            diffusion_white=D,
            carrying_capacity=10.0,
            oxygen_consumption=0.0,
        )
        model = TumorModel(
            domain_size=(20.0, 20.0),  # large domain: Gaussian stays far from edges
            params=params,
            cell_cycle_params=self._frozen_cell_cycle(),
            immune_params=self._silent_immune(),
            n_initial_points=800,
        )

        # IC: centered Gaussian
        center = np.array([10.0, 10.0])
        r_sq = np.sum((model.mesh.points - center) ** 2, axis=1)
        r = np.sqrt(r_sq)
        density = A * np.exp(-r_sq / (2.0 * sigma_0 ** 2))
        model.set_initial_density(density)

        # Simulate with stable dt
        dt = self._stable_dt(model, D)
        n_steps = int(T_final / dt)
        for _ in range(n_steps):
            model.update(dt)

        # Measure by fitting a Gaussian to the density profile
        mask = model.tumor_density > 0.005  # only fit meaningful density

        def gaussian_radial(r_vals, A_fit, sigma_fit):
            return A_fit * np.exp(-r_vals ** 2 / (2.0 * sigma_fit ** 2))

        popt, _ = curve_fit(
            gaussian_radial, r[mask], model.tumor_density[mask],
            p0=[peak_analytical, sigma_T]
        )
        sigma_fit = abs(popt[1])
        A_fit = popt[0]

        sigma_error = abs(sigma_fit - sigma_T) / sigma_T
        peak_error = abs(A_fit - peak_analytical) / peak_analytical

        # Diagnostics
        print(f"\n{'='*60}")
        print(f"Benchmark 2: Pure Diffusion (Gaussian Broadening)")
        print(f"{'='*60}")
        print(f"  Parameters: D={D}, sigma_0={sigma_0}, T={T_final}")
        print(f"  Sigma: fit={sigma_fit:.4f}, expected={sigma_T:.4f}, "
              f"error={sigma_error:.2%}")
        print(f"  Peak:  fit={A_fit:.4f}, expected={peak_analytical:.4f}, "
              f"error={peak_error:.2%}")
        print(f"  Timestep: dt={dt:.6f}, n_steps={n_steps}")

        assert sigma_error < 0.03, (
            f"Gaussian width error {sigma_error:.2%} exceeds 3% "
            f"(fit={sigma_fit:.4f}, expected={sigma_T:.4f})"
        )

    # ------------------------------------------------------------------
    # Benchmark 3: Exponential Growth
    # ------------------------------------------------------------------

    def test_exponential_growth(self):
        """
        Benchmark 3: Exponential growth u(t) = u₀exp(ρt) when D=0, u≪K.

        With no diffusion and very high carrying capacity, the logistic
        growth term ρu(1-u/K) ≈ ρu, giving pure exponential growth.
        """
        rho = 0.1    # growth rate (day⁻¹)
        u_0 = 0.01   # initial density (≪ K=100)
        T_final = 10.0  # days

        expected_density = u_0 * np.exp(rho * T_final)  # 0.02718...

        params = TumorParameters(
            growth_rate=rho,
            diffusion_white=0.0,
            carrying_capacity=100.0,
            oxygen_consumption=0.0,
            hypoxia_threshold=0.0,
        )
        model = TumorModel(
            domain_size=(10.0, 10.0),
            params=params,
            cell_cycle_params=self._frozen_cell_cycle(),
            immune_params=self._silent_immune(),
            n_initial_points=200,
        )

        # IC: uniform low density
        density = np.full(len(model.mesh.points), u_0)
        model.set_initial_density(density)

        # Simulate with small dt for Forward Euler accuracy
        dt = 0.01
        n_steps = int(T_final / dt)
        for _ in range(n_steps):
            model.update(dt)

        # Measure
        mean_density = np.mean(model.tumor_density)
        std_density = np.std(model.tumor_density)
        relative_error = abs(mean_density - expected_density) / expected_density
        uniformity = std_density / mean_density if mean_density > 0 else 0.0

        # Forward Euler theoretical error for comparison
        fe_density = u_0 * (1.0 + rho * dt) ** n_steps
        fe_error = abs(fe_density - expected_density) / expected_density

        # Diagnostics
        print(f"\n{'='*60}")
        print(f"Benchmark 3: Exponential Growth")
        print(f"{'='*60}")
        print(f"  Parameters: rho={rho}, D=0, u_0={u_0}, T={T_final}")
        print(f"  Mean density: measured={mean_density:.6f}, "
              f"expected={expected_density:.6f}, error={relative_error:.2%}")
        print(f"  Spatial uniformity: std/mean = {uniformity:.6f}")
        print(f"  Forward Euler theoretical error: {fe_error:.4%}")

        assert relative_error < 0.01, (
            f"Exponential growth error {relative_error:.2%} exceeds 1% "
            f"(measured={mean_density:.6f}, expected={expected_density:.6f})"
        )
        assert uniformity < 0.001, (
            f"Spatial non-uniformity {uniformity:.4%} exceeds 0.1% "
            f"(std={std_density:.6f}, mean={mean_density:.6f})"
        )

    # ------------------------------------------------------------------
    # Benchmark 4: Linear-Quadratic Cell Survival
    # ------------------------------------------------------------------

    def test_lq_cell_survival(self):
        """
        Benchmark 4: LQ model SF = exp(-αd - βd²) at multiple dose levels.

        With uniform phase sensitivity (all factors=1.0), OER disabled
        (oer_max=1.0), and no resistant fraction, every cell experiences
        the same α, β, and the survival fraction matches the standard
        LQ formula exactly.
        """
        alpha = 0.15  # Gy⁻¹
        beta = 0.05   # Gy⁻²

        treatment_params = TreatmentParameters(
            fractionation_alpha=alpha,
            fractionation_beta=beta,
            oer_max=1.0,
            resistant_fraction=0.0,
            radiation_g1_factor=1.0,
            radiation_s_factor=1.0,
            radiation_g2_factor=1.0,
            radiation_m_factor=1.0,
            radiation_q_factor=1.0,
        )

        dose_levels = [1.0, 2.0, 5.0, 10.0]

        print(f"\n{'='*60}")
        print(f"Benchmark 4: Linear-Quadratic Cell Survival")
        print(f"{'='*60}")
        print(f"  Parameters: alpha={alpha}, beta={beta}, oer_max=1.0")

        for dose in dose_levels:
            model = TumorModel(
                domain_size=(10.0, 10.0),
                params=TumorParameters(),
                treatment_params=treatment_params,
                n_initial_points=200,
            )

            # Set uniform density
            u_init = 0.5
            density = np.full(len(model.mesh.points), u_init)
            model.set_initial_density(density)

            total_before = np.sum(model.tumor_density)
            model.apply_treatment("radiation", dose=dose)
            total_after = np.sum(model.tumor_density)

            sf_measured = total_after / total_before
            sf_expected = np.exp(-alpha * dose - beta * dose ** 2)
            relative_error = abs(sf_measured - sf_expected) / sf_expected

            print(f"  Dose={dose:5.1f} Gy: SF_measured={sf_measured:.6f}, "
                  f"SF_expected={sf_expected:.6f}, error={relative_error:.2e}")

            assert relative_error < 0.01, (
                f"LQ survival at {dose} Gy: error {relative_error:.4%} exceeds 1% "
                f"(measured={sf_measured:.6f}, expected={sf_expected:.6f})"
            )

    # ------------------------------------------------------------------
    # Benchmark 5: Radial Symmetry Preservation
    # ------------------------------------------------------------------

    def test_radial_symmetry_preservation(self):
        """
        Benchmark 5: Radial symmetry of a centered Gaussian is preserved.

        A radially symmetric initial condition in a uniform, isotropic medium
        should remain radially symmetric. We measure symmetry by fitting
        a radial Gaussian to the density profile and computing the RMS
        deviation — the fraction of density not explained by the radial fit.
        """
        D = 0.1

        params = TumorParameters(
            growth_rate=0.0,   # pure diffusion for clean symmetry test
            diffusion_white=D,
            carrying_capacity=10.0,
            oxygen_consumption=0.0,
        )
        model = TumorModel(
            domain_size=(20.0, 20.0),
            params=params,
            cell_cycle_params=self._frozen_cell_cycle(),
            immune_params=self._silent_immune(),
            n_initial_points=1500,
        )

        # IC: centered Gaussian
        center = np.array([10.0, 10.0])
        sigma = 2.0
        A = 0.3
        r_sq = np.sum((model.mesh.points - center) ** 2, axis=1)
        r = np.sqrt(r_sq)
        density = A * np.exp(-r_sq / (2.0 * sigma ** 2))
        model.set_initial_density(density)

        # Simulate for 2 days with stable dt
        dt = self._stable_dt(model, D)
        T_final = 2.0
        n_steps = int(T_final / dt)
        for _ in range(n_steps):
            model.update(dt)

        # Measure symmetry: fit radial Gaussian, compute residual
        mask = (r < 8.0) & (model.tumor_density > 0.005)

        def gaussian_radial(r_vals, A_fit, sigma_fit):
            return A_fit * np.exp(-r_vals ** 2 / (2.0 * sigma_fit ** 2))

        popt, _ = curve_fit(
            gaussian_radial, r[mask], model.tumor_density[mask],
            p0=[0.2, 2.5]
        )

        predicted = gaussian_radial(r[mask], *popt)
        residual = model.tumor_density[mask] - predicted
        rms_residual = np.sqrt(np.mean(residual ** 2))
        mean_density = np.mean(model.tumor_density[mask])
        relative_rms = rms_residual / mean_density

        # Also compute per-shell CV for diagnostic insight
        distances = r
        shell_stats = []
        for r_lo in np.arange(1.0, 6.0, 1.0):
            r_hi = r_lo + 1.0
            shell_mask = (distances >= r_lo) & (distances < r_hi)
            n_in_shell = np.sum(shell_mask)
            if n_in_shell < 10:
                continue
            shell_d = model.tumor_density[shell_mask]
            mean_d = np.mean(shell_d)
            if mean_d < 0.01:
                continue
            cv = np.std(shell_d) / mean_d
            shell_stats.append((r_lo, r_hi, n_in_shell, mean_d, cv))

        # Diagnostics
        print(f"\n{'='*60}")
        print(f"Benchmark 5: Radial Symmetry Preservation")
        print(f"{'='*60}")
        print(f"  Parameters: D={D}, T={T_final} days, n_points={len(model.mesh.points)}")
        print(f"  Gaussian fit: A={popt[0]:.4f}, sigma={popt[1]:.4f}")
        print(f"  RMS residual / mean density: {relative_rms:.4f} ({relative_rms*100:.2f}%)")
        print(f"  Per-shell diagnostics:")
        print(f"  {'Shell':>12s}  {'N':>4s}  {'Mean':>8s}  {'CV':>8s}")
        for r_lo, r_hi, n, mean_d, cv in shell_stats:
            print(f"  [{r_lo:.1f}, {r_hi:.1f})  {n:4d}  {mean_d:8.4f}  {cv:8.4f}")

        assert relative_rms < 0.05, (
            f"Radial symmetry RMS residual {relative_rms:.4f} ({relative_rms*100:.2f}%) "
            f"exceeds 5% threshold"
        )
