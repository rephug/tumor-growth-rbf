# Development Manual: Roadmap to Clinical Utility

## Tumor Growth RBF-FD Simulator

**Purpose of this document:** This manual provides a detailed, step-by-step development plan for taking the tumor growth simulator from its current research prototype to a validated tool that can meaningfully assist in clinical oncology research and, eventually, treatment planning. Each section describes *what* needs to be built, *why* it matters, *how* to implement it technically, *what to test*, and *what literature to reference*.

**Current state (v0.3):** Working 2D/3D simulator with meshless RBF-FD spatial discretization, cell cycle dynamics (G1/S/G2/M/Q/N), tissue-heterogeneous diffusion, treatment modeling (radiation via Linear-Quadratic model with Alper-Howard-Flanders OER, chemotherapy with phase-specific sensitivity, immunotherapy), immune response, adaptive mesh refinement, patient-specific parameter fitting, treatment-resistant subpopulations, and accelerated post-treatment repopulation. All 71 tests pass across 3 test files. 5 analytical benchmarks validated. Synthetic initial conditions; no real patient data integration yet.

---

## Table of Contents

1. [Development Philosophy](#1-development-philosophy)
2. [Phase 1: Foundation for Clinical Relevance](#2-phase-1-foundation-for-clinical-relevance)
   - 1A. Three-Dimensional Extension
   - 1B. Patient-Specific Parameter Fitting
   - 1C. Analytical Validation
3. [Phase 2: Imaging Integration and Anisotropy](#3-phase-2-imaging-integration-and-anisotropy)
   - 2A. Medical Image I/O
   - 2B. DTI-Based Anisotropic Diffusion
   - 2C. Realistic Radiation Dose Import
4. [Phase 3: Toward Clinical Utility](#4-phase-3-toward-clinical-utility)
   - 3A. Uncertainty Quantification
   - 3B. Clinical Workflow Integration
   - 3C. Quantitative Clinical Validation
5. [Phase 4: Research Frontiers](#5-phase-4-research-frontiers)
   - 4A. Machine Learning Acceleration
   - 4B. Multi-Scale Biology
   - 4C. Treatment Optimization Under Uncertainty
   - 4D. Emerging Treatment Modalities
6. [Infrastructure and Engineering](#6-infrastructure-and-engineering)
7. [Regulatory Considerations](#7-regulatory-considerations)
8. [Key Literature by Topic](#8-key-literature-by-topic)
9. [Appendix: Current Architecture Reference](#9-appendix-current-architecture-reference)

---

## 1. Development Philosophy

### Build incrementally, validate constantly

Every feature should have a corresponding validation test before it's considered complete. The most dangerous thing in computational medicine is a tool that produces plausible-looking but incorrect results. Each development phase ends with a validation milestone.

### Prioritize by clinical impact, not technical elegance

The ordering of phases below reflects what will most improve the tool's ability to help real patients, not what's most interesting to implement. 3D comes before angiogenesis. Parameter fitting comes before multi-scale biology.

### Be honest about uncertainty

A prediction with a confidence interval is more useful than a precise prediction with unknown reliability. Every output should eventually carry uncertainty estimates.

### Interoperate with existing tools

Don't rebuild what exists. Read standard medical imaging formats. Import dose distributions from existing treatment planning systems. Export results that clinicians can visualize in tools they already use.

---

## 2. Phase 1: Foundation for Clinical Relevance

*Estimated effort: 3–6 months for an experienced developer*
*Goal: A 3D simulator that can fit patient-specific parameters and has been validated against known analytical solutions*

---

### 1A. Three-Dimensional Extension

**Why this matters:** Tumors are three-dimensional. A 2D slice cannot capture growth along fiber tracts that leave the imaging plane, cannot predict volumetric tumor burden, and cannot model realistic radiation dose distributions. Every published clinical validation study uses 3D models. Without this, the tool cannot make predictions that a clinician could act on.

**What changes in each module:**

#### Core: mesh_handler.py

The MeshHandler currently generates 2D points and finds 2D neighbors. The extension to 3D is conceptually straightforward but requires care.

Changes needed:
- `initialize_points()`: Generate 3D Halton sequences. The current implementation uses bases 2 and 3 for the x and y coordinates. For 3D, add base 5 for the z coordinate. The Halton sequence generalizes naturally to any dimension.

```python
# Current 2D Halton
def _halton_2d(n, base1=2, base2=3):
    ...

# New 3D Halton
def _halton_3d(n, base1=2, base2=3, base3=5):
    # Same algorithm, one more coordinate
    ...
```

- `domain_size`: Change from `Tuple[float, float]` to `Tuple[float, float, float]`. All internal references to `domain_size[0]` and `domain_size[1]` need a `domain_size[2]` counterpart.

- `_build_neighbor_lists()`: The k-d tree (`scipy.spatial.cKDTree`) already works in any dimension — no change needed to the neighbor-finding algorithm itself. However, the number of neighbors per stencil should increase. In 2D, 15–25 neighbors is typical. In 3D, 30–50 neighbors is recommended for stable RBF-FD stencils with polynomial augmentation up to degree 2.

- Refinement/coarsening: The current cross-pattern refinement (4 points around a center) should become an octahedral pattern (6 points) or small cube pattern (8 points) in 3D.

- Memory considerations: 3D meshes are much larger. A 2D simulation with 2,000 points becomes 50,000–200,000 points in 3D for equivalent resolution. Sparse matrix storage is critical.

#### Core: rbf_solver.py

The RBF kernel (Gaussian) is already dimension-independent: `phi(r) = exp(-(epsilon*r)^2)` where `r` is Euclidean distance. No change needed to the kernel itself.

Changes needed:
- `_laplacian_gaussian_rbf()`: **Critical fix.** The Laplacian of the Gaussian RBF depends on dimension:

```
2D: L[phi] = 2*eps^2 * (2*eps^2*r^2 - 2) * exp(-(eps*r)^2)
3D: L[phi] = 2*eps^2 * (2*eps^2*r^2 - 3) * exp(-(eps*r)^2)
```

The coefficient changes from -2 to -3. This was actually the original bug in the codebase (it had -3 for a 2D problem). The solver needs to know what dimension it's operating in.

Recommended approach: Add a `dim` parameter to the solver and dispatch the correct formula.

- `_build_poly_matrix()`: Currently builds `[1, x, y, x^2, xy, y^2]` for degree-2 polynomials. In 3D this becomes `[1, x, y, z, x^2, xy, xz, y^2, yz, z^2]` — 10 terms instead of 6. This means the augmented system matrix is larger (N+10 × N+10 instead of N+6 × N+6) and requires more neighbors per stencil to be well-conditioned.

- Gradient operators: Currently `gradient_x` and `gradient_y`. Add `gradient_z`.

#### Core: pde_assembler.py

- Add `"gradient_z"` operator support.
- No other changes needed — the assembler is already dimension-agnostic in its structure.

#### Biology modules

**No changes needed.** This is the payoff of the clean architecture. All biology modules (cell_populations.py, immune_response.py, treatments.py, tissue_properties.py) operate pointwise on 1D arrays. They don't know or care about spatial dimension. The spatial derivatives are computed by the core layer and passed in.

The one exception: `treatments.py` has a `_calculate_dose_distribution()` method that creates a synthetic dose map on a grid. In practice, this should be replaced with imported dose data (see Phase 2C). For the 3D transition, this method should be updated or marked as synthetic-only.

#### tumor_model.py

- Update `_initialize_state()` to create a 3D Gaussian initial condition.
- Update `_compute_gradient_magnitude()` to include the z-component.
- All operator-building calls remain the same — the assembler handles dimensions automatically once the solver is updated.

#### Performance considerations

3D simulations are 10–100× more expensive than 2D. Key optimizations:

1. **Sparse matrix operations** (already using `scipy.sparse.csr_matrix` — good)
2. **Parallel weight computation**: The RBF-FD weight calculation for each point is independent. Use `concurrent.futures.ProcessPoolExecutor` or `joblib` to parallelize across points.
3. **Operator caching**: The current `_operators_dirty` flag approach is correct. In 3D, rebuilding operators is very expensive, so aggressive caching matters more.
4. **Consider using Polyharmonic Splines (PHS)**: Instead of Gaussians, PHS-based RBF-FD (e.g., `r^5` or `r^7` with polynomial augmentation) avoids the shape parameter sensitivity problem that Gaussians have. This becomes more important in 3D where conditioning issues are more severe. See Flyer et al. (2016), "On the role of polynomials in RBF-FD approximations."

#### Testing for 3D

- Verify `∇²(x² + y² + z²) = 6` on scattered 3D points (should be exact with degree-2 polynomial augmentation)
- Verify `∂/∂z (az + b) = a` for gradient_z
- Run a 3D spherically symmetric tumor growth and verify the growth front is spherical (not faceted or dimension-dependent)
- Compare 3D results on a thin slab with 2D results — they should approximately agree

#### References for 3D RBF-FD

- Flyer N, Fornberg B, Bayona V, Barnett GA. "On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy." J Comput Phys 321:21-38, 2016.
- Bayona V, Flyer N, Fornberg B, Barnett GA. "On the role of polynomials in RBF-FD approximations: II. Numerical solution of elliptic PDEs." J Comput Phys 332:257-273, 2017.

---

### 1B. Patient-Specific Parameter Fitting (Inverse Problem)

**Why this matters:** The two most important parameters — proliferation rate (ρ) and diffusion coefficient (D) — vary enormously between patients. Literature values span an order of magnitude. The clinical utility of the model depends on estimating these parameters for a specific patient from their own imaging data.

**The problem:** Given two MRI scans of the same patient taken at different times (showing the tumor boundary at t₁ and t₂), find the values of ρ and D that best reproduce the observed growth.

**Approach: Grid search over (ρ, D) parameter space**

This is the approach used by Swanson, Harpold, and others and is well-validated clinically. It works because the parameter space is only 2D (or at most 3–4D if you include tissue-specific diffusion ratios).

Implementation plan:

```
1. Load patient's tumor contour at time t1 (from MRI segmentation)
2. Load patient's tumor contour at time t2 (later MRI)
3. For each candidate (rho, D) pair on a grid:
   a. Initialize simulation with t1 contour
   b. Simulate forward to time t2
   c. Compare simulated tumor boundary with observed t2 contour
   d. Record the mismatch (objective function value)
4. The (rho, D) pair with minimum mismatch is the patient's estimated parameters
```

**Objective function options (how to measure mismatch):**

Option A — Dice coefficient (overlap-based):
```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```
where A is the simulated tumor region (density > threshold) and B is the observed tumor region. Maximize Dice. This is the most commonly used metric in the literature.

Option B — Boundary distance (Hausdorff or mean surface distance):
Compute the distance between the simulated and observed tumor boundaries. Minimize mean distance. More sensitive to boundary shape differences.

Option C — Log-likelihood (for probabilistic formulation):
Treat the observed contour as data with noise and compute the likelihood of the observation given the model parameters. This naturally extends to Bayesian parameter estimation (Phase 3A).

**New file: `src/tumor_growth_rbf/utils/parameter_fitting.py`**

```python
class ParameterFitter:
    """
    Estimates patient-specific (rho, D) parameters from serial imaging.
    """
    def __init__(self, domain_size, tissue_data=None):
        ...

    def fit_from_contours(self,
                         contour_t1: np.ndarray,  # Binary mask at time 1
                         contour_t2: np.ndarray,  # Binary mask at time 2
                         delta_t: float,           # Days between scans
                         rho_range=(0.001, 0.1),   # Search range for rho
                         D_range=(0.01, 1.0),      # Search range for D
                         n_grid=20,                # Grid resolution
                         ) -> dict:
        """
        Grid search over (rho, D) to find best-fit parameters.
        Returns dict with best rho, best D, confidence region, etc.
        """
        ...

    def compute_dice(self, simulated, observed, threshold=0.1):
        """Dice similarity coefficient between two contours."""
        sim_binary = simulated > threshold
        obs_binary = observed > 0.5
        intersection = np.sum(sim_binary & obs_binary)
        return 2 * intersection / (np.sum(sim_binary) + np.sum(obs_binary))
```

**Parallelization:** Each (ρ, D) evaluation is independent. With a 20×20 grid, you have 400 simulations. These can be run in parallel using `concurrent.futures` or `joblib`. On a modern workstation, this should take minutes, not hours, for a 2D model. In 3D, it becomes more expensive — 30 minutes to a few hours depending on resolution.

**More sophisticated optimization (optional but better):**

After the grid search identifies the approximate region, refine with:
- `scipy.optimize.minimize` (Nelder-Mead or L-BFGS-B) starting from the grid search minimum
- Or Bayesian optimization (`scikit-optimize`) which is sample-efficient

**Testing:**

1. Generate a synthetic "patient": run a forward simulation with known (ρ*, D*), save the tumor contour at two times
2. Run the parameter fitter on those two contours
3. Verify that the fitted parameters are close to (ρ*, D*) — should recover them within 10% on a 20×20 grid

**Key references:**

- Harpold HLP et al. "The evolution of mathematical modeling of glioma proliferation and invasion." J Neuropathol Exp Neurol 66(1):1-9, 2007.
- Swanson KR et al. "A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma." Br J Cancer 98:113-119, 2008.
- Jackson PR et al. "Patient-specific mathematical neuro-oncology: using a simple proliferation and invasion tumor model to inform clinical practice." Bull Math Biol 77:846-856, 2015.

---

### 1C. Analytical Validation (Benchmark Problems)

**Why this matters:** Before the tool can be trusted for any clinical application, we need to demonstrate that the numerical methods produce correct results on problems where the exact answer is known. This is non-negotiable for any computational tool in medicine.

**Benchmark 1: Fisher-KPP traveling wave**

The Fisher-KPP equation `∂u/∂t = D∇²u + ρu(1-u)` has an exact traveling wave solution. In 1D, the wave front moves at speed `v = 2√(Dρ)` with a specific shape. 

Test procedure:
1. Initialize with a step function (u=1 for x<0, u=0 for x>0)
2. Simulate with known D and ρ, no oxygen dependence, no immune, no treatments
3. Measure the front velocity from the simulation
4. Compare with analytical prediction v = 2√(Dρ)
5. Pass criterion: velocity error < 5%

This validates the core diffusion + logistic growth coupling.

**Benchmark 2: Pure diffusion (heat equation)**

The heat equation `∂u/∂t = D∇²u` with a Gaussian initial condition has an exact solution: the Gaussian broadens with standard deviation σ(t) = √(σ₀² + 2Dt).

Test procedure:
1. Initialize with a Gaussian of known σ₀
2. Simulate pure diffusion (growth rate = 0) for time T
3. Measure the width of the simulated distribution
4. Compare with analytical prediction σ(T) = √(σ₀² + 2DT)
5. Pass criterion: width error < 2%

This validates the diffusion operator in isolation.

**Benchmark 3: Exponential growth**

With no diffusion (D=0) and no carrying capacity limit, the equation `∂u/∂t = ρu` has solution `u(t) = u₀ exp(ρt)`.

Test procedure:
1. Initialize with uniform density
2. Simulate with D=0, high carrying capacity
3. Compare density at time T with u₀ exp(ρT)
4. Pass criterion: relative error < 1%

**Benchmark 4: Linear-Quadratic cell survival**

After a single radiation dose d, the surviving fraction should be `SF = exp(-αd - βd²)`.

Test procedure:
1. Create uniform population
2. Apply single radiation dose
3. Compare surviving fraction with LQ prediction
4. Test for multiple dose levels (1, 2, 5, 10 Gy)
5. Pass criterion: SF error < 1%

**Benchmark 5: Radial symmetry preservation**

A radially symmetric initial condition in a uniform medium should produce a radially symmetric solution at all times.

Test procedure:
1. Initialize with centered Gaussian in 2D (and later 3D)
2. Simulate for many timesteps
3. Measure deviation from radial symmetry (variance of density at fixed radius)
4. Pass criterion: symmetry error < 5% of mean density

**New file: `tests/test_benchmarks.py`**

This should be a separate test file focused purely on these validation benchmarks. Each test should print quantitative comparison metrics, not just pass/fail.

**References:**

- Murray JD. "Mathematical Biology I: An Introduction." 3rd ed. Springer, 2002. (Fisher-KPP traveling wave theory, Chapter 13)
- Swanson KR. "Virtual and real brain tumors: using mathematical modeling to quantify glioma growth and invasion." J Neurol Sci 216:1-10, 2003.

---

## 3. Phase 2: Imaging Integration and Anisotropy

*Estimated effort: 3–6 months*
*Goal: Read real medical imaging data, model fiber-tract-guided diffusion, and import realistic radiation dose distributions*

---

### 2A. Medical Image I/O

**Why this matters:** Currently, the tool requires users to manually convert their imaging data to NumPy arrays. For practical use, it needs to read standard medical imaging formats directly.

**Formats to support:**

| Format | Extension | Use Case | Library |
|--------|-----------|----------|---------|
| NIfTI | .nii, .nii.gz | MRI volumes, segmentations | `nibabel` |
| DICOM | .dcm | Raw scanner output | `pydicom` |
| DICOM-RT | .dcm (RT types) | Radiation dose, contours | `pydicom` + `dicompyler-core` |
| NRRD | .nrrd | 3D Slicer exports | `pynrrd` |

**New file: `src/tumor_growth_rbf/io/image_loader.py`**

```python
class MedicalImageLoader:
    """
    Loads and preprocesses medical imaging data for simulation.
    """
    def load_nifti(self, filepath: str) -> dict:
        """
        Load NIfTI file and extract:
        - volume data (3D numpy array)
        - voxel dimensions (mm)
        - affine transform (for coordinate mapping)
        - orientation info
        """
        import nibabel as nib
        img = nib.load(filepath)
        return {
            'data': img.get_fdata(),
            'voxel_size': img.header.get_zooms()[:3],
            'affine': img.affine,
            'shape': img.shape,
        }

    def load_segmentation(self, filepath: str,
                         label_map: dict) -> np.ndarray:
        """
        Load tissue segmentation and map integer labels
        to TissueType enums.

        label_map example:
            {2: TissueType.WHITE_MATTER,
             3: TissueType.GRAY_MATTER,
             4: TissueType.CSF}
        """
        ...

    def load_dti(self, filepath: str) -> np.ndarray:
        """
        Load DTI tensor data. Returns array of shape (X, Y, Z, 3, 3)
        representing the diffusion tensor at each voxel.
        """
        ...

    def load_dicom_rt_dose(self, filepath: str) -> dict:
        """
        Load radiation dose distribution from DICOM-RT dose file.
        Returns dose grid with spatial coordinates.
        """
        ...

    def extract_slice(self, volume: np.ndarray,
                     slice_axis: int, slice_idx: int) -> np.ndarray:
        """Extract a 2D slice from a 3D volume."""
        ...

    def resample_to_simulation_grid(self, image_data, sim_points):
        """
        Interpolate image data (on regular voxel grid) to
        scattered simulation points. Uses scipy.interpolate.
        """
        from scipy.interpolate import RegularGridInterpolator
        ...
```

**Key implementation detail — coordinate mapping:**

Medical images use a specific coordinate system (usually RAS — Right, Anterior, Superior). The simulator uses physical coordinates in mm. The NIfTI affine transform maps between voxel indices and physical coordinates. This mapping must be handled correctly or the tissue properties will be applied to the wrong locations.

```python
# Voxel (i, j, k) → physical (x, y, z) in mm
physical_coords = affine @ [i, j, k, 1]
```

**New dependencies:**
- `nibabel>=4.0` (NIfTI I/O)
- `pydicom>=2.3` (DICOM I/O)
- `dicompyler-core>=0.5` (DICOM-RT dose parsing)

**Testing:**
- Load a publicly available brain MRI (e.g., from OASIS or IXI dataset)
- Verify correct voxel dimensions, orientation
- Verify tissue segmentation labels map correctly
- Round-trip test: load → extract slice → verify against known values

---

### 2B. DTI-Based Anisotropic Diffusion

**Why this matters:** This is the single feature most likely to improve prediction accuracy for gliomas. Gliomas spread preferentially along white matter fiber tracts — the corpus callosum, corticospinal tract, superior longitudinal fasciculus, etc. Isotropic diffusion misses this entirely. DTI-guided anisotropic diffusion is the feature that distinguishes research-grade glioma models from toy models.

**Mathematical background:**

Currently the diffusion term is:
```
∇·(D ∇u) where D is a scalar
```

With DTI data, this becomes:
```
∇·(D̃ ∇u) where D̃ is a 3×3 symmetric positive-definite tensor
```

At each point in space, the DTI scan gives you three eigenvalues (λ₁ ≥ λ₂ ≥ λ₃) and three eigenvectors (v₁, v₂, v₃). The largest eigenvalue/eigenvector pair represents the primary fiber direction. The diffusion tensor for tumor spread is constructed from this:

```python
# At each voxel:
# D_water = DTI tensor (from imaging)
# D_tumor = f(D_water) - mapping from water diffusion to tumor diffusion
#
# Common approach (Jbabdi et al., Clatz et al.):
D_tumor = d_white * (ratio * fiber_direction @ fiber_direction.T +
                     (1 - ratio) * np.eye(3))
```

where `ratio` controls the anisotropy (0 = isotropic, 1 = purely along fibers) and `d_white` is the base white matter diffusion coefficient.

**RBF-FD implementation of tensor diffusion:**

The key challenge is computing `∇·(D̃∇u)` with RBF-FD on scattered points. There are two approaches:

**Approach A (recommended): Direct tensor operator**

Expand `∇·(D̃∇u)` in components:
```
∂/∂x(D_xx ∂u/∂x + D_xy ∂u/∂y + D_xz ∂u/∂z) +
∂/∂y(D_xy ∂u/∂x + D_yy ∂u/∂y + D_yz ∂u/∂z) +
∂/∂z(D_xz ∂u/∂x + D_yz ∂u/∂y + D_zz ∂u/∂z)
```

This can be computed using combinations of first-derivative operators that the RBF solver already supports (gradient_x, gradient_y, gradient_z).

Implementation outline:
```python
def compute_anisotropic_diffusion(self, u, D_tensor):
    """
    Compute ∇·(D̃∇u) using first-derivative operators.

    D_tensor: array of shape (n_points, 3, 3)
    """
    # Compute gradient of u
    du_dx = grad_x_operator @ u
    du_dy = grad_y_operator @ u
    du_dz = grad_z_operator @ u

    # Compute D̃∇u at each point
    flux_x = D_tensor[:,0,0]*du_dx + D_tensor[:,0,1]*du_dy + D_tensor[:,0,2]*du_dz
    flux_y = D_tensor[:,1,0]*du_dx + D_tensor[:,1,1]*du_dy + D_tensor[:,1,2]*du_dz
    flux_z = D_tensor[:,2,0]*du_dx + D_tensor[:,2,1]*du_dy + D_tensor[:,2,2]*du_dz

    # Compute divergence of flux
    div = grad_x_operator @ flux_x + grad_y_operator @ flux_y + grad_z_operator @ flux_z

    return div
```

Note: This uses first-derivative operators twice (for gradient then divergence), which introduces more numerical error than a direct second-derivative operator. For RBF-FD with polynomial augmentation degree ≥ 2, this is generally acceptable.

**Approach B (more accurate but complex): Custom tensor-Laplacian weights**

Compute RBF-FD weights that directly approximate the tensor diffusion operator at each stencil. This requires modifying the weight computation in `rbf_solver.py` to solve for the specific tensor operator rather than the standard Laplacian. More accurate but requires significant changes to the solver.

Recommendation: Start with Approach A. It's simpler, can be implemented with existing operators, and is accurate enough for clinical purposes.

**New file: `src/tumor_growth_rbf/io/dti_processor.py`**

```python
class DTIProcessor:
    """
    Processes DTI data into tumor diffusion tensors.
    """
    def __init__(self, anisotropy_ratio=0.5):
        self.anisotropy_ratio = anisotropy_ratio

    def load_dti_tensors(self, dti_filepath):
        """Load DTI tensor field from NIfTI."""
        ...

    def compute_tumor_diffusion_tensors(self,
                                        dti_tensors: np.ndarray,
                                        tissue_map: np.ndarray,
                                        d_white: float = 0.1,
                                        d_gray: float = 0.01) -> np.ndarray:
        """
        Convert water diffusion tensors to tumor diffusion tensors.

        In white matter: anisotropic, guided by fiber direction
        In gray matter: isotropic, lower coefficient
        """
        ...

    def extract_fiber_directions(self, dti_tensors):
        """
        Extract primary fiber direction (first eigenvector)
        at each voxel.
        """
        ...
```

**Testing:**
- Create a synthetic tensor field with fibers running in the x-direction
- Initialize a tumor at the center
- Verify that the tumor spreads faster along x than y or z
- Quantify the anisotropy ratio of the simulated tumor and compare with the input tensor anisotropy

**Key references:**

- Jbabdi S et al. "Simulation of anisotropic growth of low-grade gliomas using diffusion tensor imaging." Magn Reson Med 54:616-624, 2005.
- Clatz O et al. "Realistic simulation of the 3-D growth of brain tumors in MR images coupling diffusion with biomechanical deformation." IEEE Trans Med Imaging 24:1334-1346, 2005.
- Painter KJ, Hillen T. "Mathematical modelling of glioma growth: the use of Diffusion Tensor Imaging (DTI) data to predict the anisotropic pathways of cancer invasion." J Theor Biol 323:25-39, 2013.

---

### 2C. Realistic Radiation Dose Import

**Why this matters:** The current simulator creates a synthetic dose distribution (exponential falloff from beam direction). Real radiation therapy uses sophisticated treatment planning systems (TPS) that compute precise dose distributions accounting for tissue density, beam geometry, multi-leaf collimator shapes, and patient anatomy. For the treatment response predictions to be meaningful, the simulator should use the actual planned dose distribution, not a synthetic approximation.

**How radiation dose data works:**

Radiation dose is typically stored in DICOM-RT Dose format. The dose is on a regular 3D grid, with dose in Gray (Gy) at each voxel. Each treatment fraction delivers a scaled version of this distribution.

```python
class RTDoseLoader:
    """
    Load radiation dose distributions from DICOM-RT.
    """
    def load_dicom_rt_dose(self, filepath):
        """
        Load DICOM-RT dose file.
        Returns dose grid (Gy), grid coordinates, and grid spacing.
        """
        import pydicom
        ds = pydicom.dcmread(filepath)

        dose_grid = ds.pixel_array * ds.DoseGridScaling
        # dose_grid shape: (n_frames, n_rows, n_cols)

        # Extract spatial coordinates from Image Position and Pixel Spacing
        origin = np.array(ds.ImagePositionPatient)
        spacing = np.array([
            float(ds.PixelSpacing[0]),
            float(ds.PixelSpacing[1]),
            float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])
        ])

        return {
            'dose': dose_grid,
            'origin': origin,
            'spacing': spacing,
            'total_prescribed_dose': float(dose_grid.max()),
        }

    def interpolate_to_sim_points(self, dose_data, sim_points):
        """
        Interpolate dose grid to scattered simulation points.
        Uses trilinear interpolation.
        """
        from scipy.interpolate import RegularGridInterpolator
        ...
```

**Integration with treatment module:**

Replace the synthetic `_calculate_dose_distribution()` method in treatments.py:

```python
# Before (synthetic):
dose_map = self._calculate_dose_distribution(dose, beam_angles)

# After (real data):
if self.dose_distribution is not None:
    # Scale the planned dose distribution by the fraction dose
    dose_map = self.dose_distribution * (dose / self.prescribed_dose_per_fraction)
else:
    # Fall back to synthetic for cases without RT plan
    dose_map = self._calculate_dose_distribution(dose, beam_angles)
```

**Testing:**
- Load a publicly available DICOM-RT dose file (e.g., from the AAPM TG-119 benchmark datasets)
- Verify correct spatial coordinates and dose values
- Verify interpolation to scattered points preserves dose volume histogram (DVH) statistics

**References:**
- DICOM-RT standard: https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.html
- `dicompyler-core` library documentation: https://dicompyler-core.readthedocs.io/

---

## 4. Phase 3: Toward Clinical Utility

*Estimated effort: 6–12 months*
*Goal: Uncertainty-aware predictions, integration with clinical workflows, and validation against real patient data*

---

### 3A. Uncertainty Quantification

**Why this matters:** A model prediction without uncertainty bounds is clinically useless — or worse, dangerously misleading. Every parameter has uncertainty. Every measurement has noise. The model itself is an approximation. Clinicians need to know not just "the tumor will likely reach this boundary in 6 months" but "there is a 90% probability the tumor boundary will be between here and here."

**Approach: Bayesian parameter estimation**

Instead of finding a single best-fit (ρ, D) pair, compute the *posterior probability distribution* P(ρ, D | data). This distribution captures all (ρ, D) values that are consistent with the observed imaging data, weighted by how well each explains the observations.

**Implementation options, in order of complexity:**

**Option 1: Simple sampling from grid search (easiest)**

The grid search from Phase 1B already evaluates the objective function (Dice score) on a grid. Convert these scores to an approximate posterior:

```python
# From grid search, you have:
# scores[i,j] = Dice(simulation(rho_i, D_j), observed)

# Convert to approximate likelihood:
log_likelihood = np.log(scores + 1e-10)

# Posterior (with flat prior):
posterior = np.exp(log_likelihood - log_likelihood.max())
posterior /= posterior.sum()

# Now sample from posterior:
# Each (rho, D) sample gives a different forward prediction
# The spread of predictions = uncertainty
```

**Option 2: Markov Chain Monte Carlo (MCMC)**

Use `emcee` or `PyMC` to sample the posterior distribution efficiently:

```python
import emcee

def log_posterior(params, data):
    rho, log_D = params
    D = 10**log_D

    # Prior
    if rho < 0 or rho > 0.5 or D < 0.001 or D > 10:
        return -np.inf

    # Likelihood
    simulated = run_simulation(rho, D)
    dice = compute_dice(simulated, data['observed_contour'])
    log_like = -0.5 * ((1 - dice) / 0.1)**2  # Gaussian likelihood

    return log_like

# Run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                 args=[data])
sampler.run_mcmc(initial_positions, nsteps=1000)
```

Each MCMC sample gives a (ρ, D) pair. Running the forward model with each sample and collecting the results gives a distribution of predictions — the spread IS the uncertainty.

**Option 3: Polynomial Chaos Expansion (most efficient for propagation)**

Instead of running thousands of forward simulations, fit a polynomial surrogate model to the input-output relationship, then propagate uncertainty through the surrogate analytically. Libraries: `chaospy`, `UQLab`.

**Recommended approach:** Start with Option 1 (it's free — you already have the grid search). Move to Option 2 when higher-quality uncertainty estimates are needed.

**What to present to the user:**

```
Tumor extent prediction at 6 months:
  Median radius: 25.3 mm
  90% credible interval: [20.1 mm, 32.7 mm]
  Probability of crossing midline: 12%
```

**New file: `src/tumor_growth_rbf/utils/uncertainty.py`**

**References:**

- Neal RM. "MCMC using Hamiltonian dynamics." Handbook of Markov Chain Monte Carlo, 2011.
- Konukoglu E et al. "Image guided personalization of reaction-diffusion type tumor growth models using modified anisotropic eikonal equations." IEEE Trans Med Imaging 29:77-95, 2010.
- Le M et al. "Bayesian personalization of brain tumor growth model." MICCAI 2015.

---

### 3B. Clinical Workflow Integration

**Why this matters:** Even a technically perfect tool is useless if clinicians can't use it. The tool needs to fit into existing clinical workflows, not require clinicians to learn Python programming.

**Components to build:**

**1. Command-line interface (CLI)**

```bash
# Quick simulation from command line
tumor-sim run \
  --tissue-seg patient_segmentation.nii.gz \
  --tumor-mask tumor_contour.nii.gz \
  --growth-rate 0.012 \
  --diffusion 0.1 \
  --days 180 \
  --output results/

# Parameter fitting
tumor-sim fit \
  --contour-t1 tumor_t1.nii.gz \
  --contour-t2 tumor_t2.nii.gz \
  --days-between 90 \
  --tissue-seg segmentation.nii.gz \
  --output fitted_params.json

# Treatment comparison
tumor-sim compare \
  --config treatment_arms.yaml \
  --output comparison_report/
```

Use `argparse` or `click` for the CLI. This is the minimum viable interface for a researcher.

**2. Configuration files (YAML)**

Allow users to define simulations and treatment schedules in configuration files rather than Python code:

```yaml
# simulation_config.yaml
patient:
  domain_size: [120.0, 120.0, 80.0]
  tissue_segmentation: "data/segmentation.nii.gz"
  tumor_contour: "data/tumor_t1.nii.gz"
  vessel_map: "data/vessels.nii.gz"

parameters:
  growth_rate: 0.012
  diffusion_white: 0.1
  diffusion_gray: 0.01
  hypoxia_threshold: 0.1

treatment:
  protocol: "stupp"
  radiation:
    total_dose: 60.0
    fractions: 30
    dose_file: "data/rt_dose.dcm"  # Optional: real dose distribution
  chemotherapy:
    drug: "temozolomide"
    cycle_length: 28
    cycles: 6

simulation:
  timestep: 0.1
  duration: 365
  output_interval: 7  # Save state every 7 days
```

**3. 3D Slicer plugin (stretch goal)**

3D Slicer is the most widely used open-source medical image viewer in research. A Slicer plugin would allow users to:
- Select tissue segmentation and tumor contour from loaded images
- Set parameters via GUI
- Run simulation and visualize results overlaid on the original MRI
- Compare treatment arms side-by-side

This is a significant development effort (weeks to months) but would dramatically increase adoption. Slicer plugins are written in Python and can call the existing simulator code directly.

**4. Export results in standard formats**

- NIfTI volumes showing predicted tumor density at future timepoints
- DICOM-RT structure sets with predicted tumor contours (for import into treatment planning systems)
- PDF reports with summary statistics and visualizations

---

### 3C. Quantitative Clinical Validation

**Why this matters:** This is the step that determines whether the tool actually works. Everything before this is building the tool. This step tests whether it produces correct predictions on real patients.

**Publicly available datasets:**

| Dataset | Contents | Access |
|---------|----------|--------|
| TCIA GBM collection | Pre/post-treatment MRI, clinical outcomes | https://www.cancerimagingarchive.net |
| BraTS challenge | Segmented brain tumors, multi-institutional | https://www.synapse.org/brats |
| TCGA-GBM | Imaging + genomics + clinical data | https://portal.gdc.cancer.gov |
| IvyGAP | Anatomic features of GBM with imaging | https://glioblastoma.alleninstitute.org |
| OASIS | Normal brain MRI (for tissue segmentation testing) | https://www.oasis-brains.org |

**Validation study design:**

**Retrospective validation (first):**

1. Select patients with at least 2 pre-treatment MRI scans (for parameter fitting) and 1 or more post-treatment scans (for prediction validation)
2. Use scans 1 and 2 to fit (ρ, D) parameters
3. Predict tumor extent at the time of scan 3
4. Compare predicted extent with observed extent using Dice coefficient and Hausdorff distance
5. Report statistics across all patients (mean Dice, confidence intervals)

Target: Dice > 0.7 would be competitive with published results. Dice > 0.8 would be excellent.

**Treatment response validation:**

1. Select patients with known treatment protocols and serial imaging
2. Fit pre-treatment growth parameters
3. Simulate the known treatment protocol
4. Compare predicted post-treatment tumor volume with observed
5. Report volume prediction error and correlation

**Reporting standards:**

Follow the guidelines in:
- Stable L et al. "Quantitative metrics for evaluating tumor growth simulations." (Review the relevant computational oncology validation literature)
- Use standard statistical measures: Dice, Hausdorff distance, volume correlation, Bland-Altman analysis

---

## 5. Phase 4: Research Frontiers

*These are longer-term directions that go beyond the minimum for clinical utility but represent the cutting edge of the field.*

---

### 4A. Machine Learning Acceleration

**Problem:** Patient-specific parameter fitting requires running many forward simulations (hundreds for grid search, thousands for MCMC). In 3D, each simulation takes minutes. Total fitting time: hours to days.

**Solution:** Train a neural network surrogate that predicts simulation outputs from parameters in milliseconds.

**Approach: Physics-Informed Neural Networks (PINNs) or neural operator surrogates**

1. Generate a training dataset: run the full simulator for many (ρ, D, tissue) combinations, save the tumor density at several timepoints
2. Train a neural network to predict tumor density given (ρ, D, tissue map, time)
3. Use the trained network as a fast surrogate in the parameter fitting loop

Libraries: `PyTorch`, `DeepXDE` (for PINNs), `Fourier Neural Operator` library.

**Alternative: Transfer learning for parameter estimation**

Train a convolutional neural network to directly map from (MRI pair) → (ρ, D) without iterative simulation. This has been demonstrated by several groups:

- Ezhov I et al. "Neural parameters estimation for brain tumor growth modeling." MICCAI 2019.
- Mang A et al. "PDE-constrained optimization in medical image analysis." Optimization and Engineering, 2018.

**Effort:** Significant (months). Requires ML expertise. But the payoff is transformative for clinical usability — fitting that takes hours becomes seconds.

---

### 4B. Multi-Scale Biology

**Current model:** Tissue-level PDE (millimeter scale) with cell-cycle-phase tracking.

**What's missing:** Molecular-level heterogeneity within the tumor. GBM is notoriously heterogeneous — different regions of the same tumor can have different genetic profiles, different growth rates, and different treatment sensitivities.

**Extensions to consider:**

1. **Multiple tumor cell subpopulations**: Instead of one tumor density field with cell cycle phases, model 2–3 genetically distinct subclones (e.g., treatment-sensitive and treatment-resistant) that compete for resources and may have different growth/diffusion parameters.

2. **Intracellular signaling**: Model key pathways (EGFR, PI3K/AKT, p53) that affect growth and treatment response. This is complex but there are published ODE models for these pathways that could be coupled to the spatial PDE model.

3. **Metabolic modeling**: Tumor cells can switch between aerobic and anaerobic metabolism (Warburg effect). Modeling glucose and lactate in addition to oxygen provides a more complete picture of the tumor microenvironment.

4. **Extracellular matrix**: Tumor cells remodel the extracellular matrix (ECM) as they invade. Modeling ECM density and its effect on cell migration adds realism to the invasion model.

**Recommendation:** Start with multiple subclones (conceptually simple extension of existing architecture — add another density field with different parameters). This is clinically relevant because treatment resistance is often driven by selection of resistant subclones.

**References:**

- Alfonso JCL et al. "The biology and mathematical modelling of glioma invasion: a review." J R Soc Interface 14:20170490, 2017.
- Swanson KR et al. "Quantifying the role of angiogenesis in malignant progression of gliomas: in silico modeling integrates imaging and histology." Cancer Res 71:7366-7375, 2011.

---

### 4C. Treatment Optimization Under Uncertainty

**Current model:** Can compare pre-defined treatment schedules.

**Goal:** Automatically find the optimal treatment schedule that maximizes tumor control while minimizing normal tissue damage, accounting for parameter uncertainty.

**This is a stochastic optimization problem:**

```
maximize  E[tumor_reduction(schedule, θ)]
subject to P[normal_tissue_dose > limit] < 5%
           schedule satisfies clinical constraints
```

where θ represents uncertain parameters sampled from the posterior distribution (from Phase 3A).

**Approaches:**

1. **Robust optimization**: Optimize for the worst-case scenario within the uncertainty bounds. Conservative but safe.

2. **Stochastic programming**: Optimize the expected outcome averaged over parameter uncertainty. Better average performance but some scenarios may be poor.

3. **Reinforcement learning**: Train an agent to make sequential treatment decisions (dose, timing) that maximize long-term tumor control. The simulator serves as the "environment." This is an active research area with promising early results.

**This is genuinely cutting-edge research** — few groups have demonstrated this convincingly for gliomas. A clean, modular simulator with built-in uncertainty quantification (from Phase 3A) would be a strong foundation for this work.

---

### 4D. Emerging Treatment Modalities

**Tumor Treating Fields (TTFields / Optune):**
Alternating electric fields that disrupt mitosis. FDA-approved for GBM. The simulator could model this by adding a spatially-varying mitosis disruption term based on the electric field distribution (which depends on electrode placement and tissue conductivity). This is an active modeling area.

**CAR-T cell therapy:**
Chimeric Antigen Receptor T-cell therapy is being explored for GBM. The immune response module could be extended to model engineered immune cells with different trafficking, persistence, and killing characteristics.

**Convection-Enhanced Delivery (CED):**
Direct injection of drugs into the brain, bypassing the BBB. Modeling this requires an advection term (fluid flow) in addition to diffusion. The PDE assembler already supports advection — this is relatively straightforward to add.

---

## 6. Infrastructure and Engineering

### Testing Standards

As the tool matures toward clinical relevance, testing standards must increase:

| Level | What | Current | Target |
|-------|------|---------|--------|
| Unit tests | Individual functions | 47 tests | 100+ tests |
| Integration tests | Module interactions | Basic | Comprehensive |
| Benchmark tests | Known analytical solutions | 5 benchmarks | 5+ benchmarks (Phase 1C) |
| Parameter fitting tests | Known parameter recovery | 16 tests | Expand coverage |
| Regression tests | Results don't change unexpectedly | None | Full regression suite |
| Clinical validation | Comparison with patient data | None | Phase 3C |

**Continuous integration:** Set up GitHub Actions to run all tests on every pull request. Include benchmark tests that fail if numerical accuracy degrades.

### Documentation Standards

| Document | Audience | Status |
|----------|----------|--------|
| README.md | Everyone | Done |
| CLINICAL_GUIDE.md | Researchers/clinicians | Done |
| This development manual | Developers | Done |
| API documentation | Developers | Needed — use Sphinx with autodoc |
| Mathematical documentation | Reviewers | Needed — LaTeX document describing all equations |
| Validation report | Regulators/reviewers | Needed — generated from benchmark tests |

**Mathematical documentation** is especially important. Every equation in the code should be traceable to a published reference or derivation. Create a LaTeX document (or Jupyter notebook) that walks through:
- The reaction-diffusion equation and its biological meaning
- The RBF-FD spatial discretization and why it's appropriate
- The cell cycle ODE system
- The Linear-Quadratic model derivation
- The immune response equations
- How tissue heterogeneity enters each equation

### Performance Profiling

As the tool scales to 3D, performance will become important. Profile early:

```python
import cProfile
cProfile.run('model.update(0.1)', 'profile_output')

# Analyze
import pstats
stats = pstats.Stats('profile_output')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 time consumers
```

Expected bottlenecks (in order):
1. RBF-FD weight computation (O(N × k³) where k = stencil size)
2. Sparse matrix assembly
3. Sparse matrix-vector products
4. Biology updates (cheap — pointwise operations)

### Version Control and Release Strategy

- **Semantic versioning**: MAJOR.MINOR.PATCH
  - v0.2: 2D, synthetic data, working biology
  - v0.3: Current (3D extension, parameter fitting, analytical benchmarks, OER, resistant fraction, repopulation)
  - v0.4: Medical image I/O
  - v0.5: DTI anisotropic diffusion
  - v1.0: First release with clinical validation results
- **Changelog**: Maintain a CHANGELOG.md documenting every change
- **Tagged releases**: Every milestone gets a tagged release on GitHub
- **Zenodo DOI**: Register a DOI for citability in publications

---

## 7. Regulatory Considerations

**Important:** If the tool is ever intended to influence clinical treatment decisions (even indirectly), regulatory frameworks apply.

**In the United States:**
- The FDA classifies software that provides treatment recommendations as a medical device (Software as a Medical Device, SaMD)
- A research/educational tool that explicitly disclaims clinical decision-making is not a medical device
- If the tool is used to generate predictions that inform treatment planning, it may fall under FDA 510(k) or De Novo classification

**In the European Union:**
- The Medical Device Regulation (MDR 2017/745) applies to software that provides diagnosis or treatment recommendations
- CE marking may be required

**Practical recommendations:**
1. Until clinical validation is complete, clearly label all outputs: "FOR RESEARCH USE ONLY — NOT FOR CLINICAL DECISION-MAKING"
2. Document all assumptions and limitations
3. If pursuing clinical use, engage regulatory consultants early
4. Consider the regulatory pathway from the start — it affects development decisions (e.g., quality management systems, design controls, risk analysis)

**Software quality for medical devices:**
- IEC 62304: Software lifecycle standard for medical devices
- ISO 14971: Risk management for medical devices
- These standards require formal requirements, design documents, traceability, and verification/validation records

---

## 8. Key Literature by Topic

### Foundational Glioma Growth Modeling
1. Swanson KR, Bridge C, Murray JD, Alvord EC. "Virtual and real brain tumors: using mathematical modeling to quantify glioma growth and invasion." J Neurol Sci 216:1-10, 2003.
2. Harpold HLP, Alvord EC, Swanson KR. "The evolution of mathematical modeling of glioma proliferation and invasion." J Neuropathol Exp Neurol 66(1):1-9, 2007.
3. Swanson KR, Rostomily RC, Alvord EC. "A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma." Br J Cancer 98:113-119, 2008.

### DTI-Guided Anisotropic Diffusion
4. Jbabdi S, Mandonnet E, Duffau H, et al. "Simulation of anisotropic growth of low-grade gliomas using diffusion tensor imaging." Magn Reson Med 54:616-624, 2005.
5. Clatz O, Sermesant M, Bondiau PY, et al. "Realistic simulation of the 3-D growth of brain tumors in MR images coupling diffusion with biomechanical deformation." IEEE Trans Med Imaging 24:1334-1346, 2005.
6. Painter KJ, Hillen T. "Mathematical modelling of glioma growth: the use of DTI data to predict the anisotropic pathways of cancer invasion." J Theor Biol 323:25-39, 2013.

### RBF-FD Numerical Methods
7. Fornberg B, Flyer N. "A Primer on Radial Basis Functions with Applications to the Geosciences." SIAM, 2015.
8. Flyer N, Fornberg B, Bayona V, Barnett GA. "On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy." J Comput Phys 321:21-38, 2016.
9. Bayona V, Flyer N, Fornberg B, Barnett GA. "On the role of polynomials in RBF-FD approximations: II. Numerical solution of elliptic PDEs." J Comput Phys 332:257-273, 2017.

### Radiation Biology and Treatment Modeling
10. McMahon SJ. "The linear quadratic model: usage, interpretation and challenges." Phys Med Biol 64:01TR01, 2019.
11. Joiner MC, van der Kogel AJ. "Basic Clinical Radiobiology." 5th ed. CRC Press, 2018.
12. Stupp R, Mason WP, van den Bent MJ, et al. "Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma." N Engl J Med 352:987-996, 2005.

### Patient-Specific Parameter Estimation
13. Konukoglu E, Clatz O, Menze BH, et al. "Image guided personalization of reaction-diffusion type tumor growth models using modified anisotropic eikonal equations." IEEE Trans Med Imaging 29:77-95, 2010.
14. Jackson PR, Swanson KR. "Patient-specific mathematical neuro-oncology: using a simple proliferation and invasion tumor model to inform clinical practice." Bull Math Biol 77:846-856, 2015.
15. Le M, Delingette H, Kalpathy-Cramer J, et al. "Bayesian personalization of brain tumor growth model." MICCAI 2015.

### Uncertainty Quantification and Optimization
16. Neal RM. "MCMC using Hamiltonian dynamics." Handbook of MCMC, Chapman and Hall, 2011.
17. Mang A, Biros G. "PDE-constrained optimization in medical image analysis." Optimization and Engineering 19:765-812, 2018.

### Machine Learning for Tumor Modeling
18. Ezhov I, Scibilia K, Franitza K, et al. "Neural parameters estimation for brain tumor growth modeling." MICCAI 2019.
19. Karniadakis GE, Kevrekidis IG, Lu L, et al. "Physics-informed machine learning." Nature Reviews Physics 3:422-440, 2021.

### Reviews and Perspectives
20. Alfonso JCL, Talkenberger K, Seifert M, et al. "The biology and mathematical modelling of glioma invasion: a review." J R Soc Interface 14:20170490, 2017.
21. Yankeelov TE, Atuegwu N, Hormuth D, et al. "Clinically relevant modeling of tumor growth and treatment response." Science Translational Medicine 5:187ps9, 2013.

---

## 9. Appendix: Current Architecture Reference

### Module Dependency Graph

```
tumor_model.py (integration layer)
├── mesh_handler.py      (scattered point management)
├── rbf_solver.py        (RBF-FD weight computation)
├── pde_assembler.py     (sparse operator assembly)
├── cell_populations.py  (G1/S/G2/M/Q/N dynamics)
├── treatments.py        (radiation/chemo/immunotherapy)
├── immune_response.py   (chemokine-mediated immune response)
├── tissue_properties.py (tissue-specific parameter maps)
└── visualization.py     (plotting and animation)
```

### Data Flow (Single Timestep)

```
1. Build spatial operators (Laplacian, gradients) via pde_assembler
   └── Uses rbf_solver to compute stencil weights
   └── Cached; only rebuilt when mesh changes

2. Update oxygen: ∂O/∂t = D∇²O - consumption + perfusion
   └── Consumption depends on cell cycle phase (S/G2/M consume more)
   └── Perfusion depends on tissue type and vessel proximity

3. Update cell populations based on oxygen
   └── Normal O₂: cycle progression G1→S→G2→M→2×G1
   └── Hypoxic: proliferating → quiescent
   └── Severely hypoxic: all → necrotic

4. Compute diffusion: D(x)∇²u with tissue-varying D
5. Compute growth: ρ·u·(1 - u/K) with tissue modifiers
6. Update immune response (if active)
7. Apply combined effects: u += dt * (diffusion + growth + immune)
8. Enforce constraints: 0 ≤ u ≤ carrying_capacity
9. Redistribute density across cell cycle phases
10. Adapt mesh if density gradients have changed significantly
```

### Key Design Decisions (and why)

| Decision | Rationale |
|----------|-----------|
| RBF-FD (not FEM/FDM) | Meshless: no mesh generation needed, easy refinement, works in irregular domains |
| Scattered points (not grid) | Natural for adaptive refinement; tumor boundary gets more points automatically |
| Biology modules are pointwise | Clean separation; biology doesn't know about spatial dimension; same code works 2D/3D |
| Spatial operators passed to biology | Biology modules never build operators; they receive ∇²c, ∇c from the integration layer |
| PHS RBF-FD with polynomial augmentation | Shape-parameter-free; exact polynomial reproduction; better conditioned than Gaussian (Flyer et al. 2016) |
| Cell cycle as separate populations | Enables phase-specific treatment effects (the whole point of modeling the cell cycle) |

### Files and Line Counts (v0.3)

| File | Lines | Purpose |
|------|-------|---------|
| mesh_handler.py | ~380 | Point generation, neighbors, refinement (2D/3D) |
| rbf_solver.py | ~480 | RBF-FD weight computation (2D/3D) |
| pde_assembler.py | ~185 | Sparse operator assembly |
| cell_populations.py | ~320 | Cell cycle dynamics |
| treatments.py | ~450 | Radiation (LQ + OER), chemo, immunotherapy, resistant fraction |
| immune_response.py | ~250 | Immune cell dynamics |
| tissue_properties.py | ~240 | Tissue-specific parameters |
| tumor_model.py | ~550 | Main integration class (2D/3D, repopulation) |
| visualization.py | ~180 | Plotting utilities |
| parameter_fitting.py | ~350 | Grid search + Nelder-Mead refinement |
| test_all.py | ~650 | 47 unit tests (2D, 3D, OER, repopulation) |
| test_benchmarks.py | ~455 | 5 analytical validation benchmarks |
| test_parameter_fitting.py | ~340 | 16 parameter fitting tests |
| demo.py | ~450 | 5 interactive demos |
| clinical_workflow.py | ~530 | Clinical Stupp protocol example |
| **Total** | **~5,810** | |

---

*This document should be updated as development progresses. Each completed phase should be marked with a completion date and the validation results achieved.*
