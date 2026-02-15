# Clinical Usage Guide

A practical guide for researchers and clinicians using the Tumor Growth RBF-FD Simulator with real patient data.

## Table of Contents

1. [Overview: What Can This Tool Do?](#overview)
2. [Data Requirements](#data-requirements)
3. [Preparing Your Data](#preparing-your-data)
4. [Parameter Selection](#parameter-selection)
5. [Running Simulations](#running-simulations)
6. [Interpreting Results](#interpreting-results)
7. [Treatment Planning Workflows](#treatment-planning-workflows)
8. [Limitations and Caveats](#limitations-and-caveats)
9. [Literature References for Parameters](#literature-references)

---

## Overview

This simulator models glioma growth in a 2D cross-section of brain tissue. It can help answer questions like:

- **"How fast will this tumor grow?"** — Given estimated growth parameters, predict tumor extent over time.
- **"What happens if we change the fractionation schedule?"** — Compare 2 Gy × 30 fractions vs. 3 Gy × 20 fractions.
- **"How does tissue type affect growth pattern?"** — Model differential spread through white vs. gray matter.
- **"When is the optimal time for the next treatment?"** — Explore cell-cycle-aware scheduling.
- **"What's the expected treatment response?"** — Predict tumor volume reduction under different regimens.

### What This Tool Does NOT Do

- This is a **research and educational tool**, not a clinical decision-support system
- It operates in **2D** (a single axial slice); it does not model full 3D tumor geometry
- It does **not** account for mechanical effects (mass effect, midline shift)
- Parameter values have **significant uncertainty** — results are best used for relative comparisons, not absolute predictions

---

## Data Requirements

### Tier 1: No Imaging Data (Synthetic Mode)

You can run simulations immediately with no data at all. The simulator creates a synthetic Gaussian tumor in a uniform tissue domain. Useful for:

- Learning the tool
- Parameter sensitivity studies
- Comparing treatment schedules
- Understanding biological mechanisms

**What you specify:**
- Domain size (e.g., 10 mm × 10 mm)
- Growth rate (from literature or clinical doubling time)
- Treatment parameters

### Tier 2: Basic Imaging Data

For patient-specific tissue heterogeneity, you need:

| Data | Source | Format | Purpose |
|------|--------|--------|---------|
| **Tissue segmentation** | T1/T2 MRI + segmentation tool | 2D integer array (NumPy) | White/gray matter diffusion coefficients |
| **Initial tumor extent** | T1-contrast or FLAIR MRI | Binary mask or density map | Initial tumor location and size |

**How to get tissue segmentation:**
- **FreeSurfer** (`recon-all`): Gold standard for brain segmentation from T1 MRI
- **FSL FAST**: Fast tissue-type segmentation (white matter, gray matter, CSF)
- **SPM12**: Statistical Parametric Mapping tissue classification
- **Manual segmentation**: In 3D Slicer, ITK-SNAP, or similar tool

**How to get initial tumor contour:**
- Manual contouring of contrast-enhancing region on T1+Gd MRI
- Or FLAIR hyperintensity boundary for infiltrative extent
- Export as binary mask (NIfTI → NumPy via `nibabel`)

### Tier 3: Full Imaging Data

For the most detailed simulations, additionally provide:

| Data | Source | Format | Purpose |
|------|--------|--------|---------|
| **Vessel map** | Contrast-enhanced MRI, MRA, or DSC perfusion | Binary mask | Oxygen supply, drug delivery |
| **DTI fiber orientation** | Diffusion Tensor Imaging | Tensor field | Anisotropic diffusion along white matter tracts |
| **Perfusion map** | DSC-MRI or ASL | Continuous map | Spatially varying oxygen perfusion |

> **Note:** DTI-based anisotropic diffusion is not yet implemented in the current version but is planned. The current model uses isotropic diffusion with tissue-type-dependent coefficients.

---

## Preparing Your Data

### Converting MRI to Simulator Input

```python
import nibabel as nib
import numpy as np

# Load NIfTI file (from FreeSurfer, FSL, etc.)
img = nib.load("tissue_segmentation.nii.gz")
volume = img.get_fdata()

# Extract a single axial slice (e.g., slice containing tumor center)
slice_idx = 90  # Choose the slice with the largest tumor cross-section
tissue_slice = volume[:, :, slice_idx].astype(int)

# Map segmentation labels to simulator tissue types
# These mappings depend on your segmentation tool's label convention
# FreeSurfer example:
#   2 = left cerebral white matter
#   3 = left cerebral cortex (gray matter)
#   41 = right cerebral white matter
#   42 = right cerebral cortex
from tumor_growth_rbf import TissueType

tissue_labels = {
    2: TissueType.WHITE_MATTER,
    41: TissueType.WHITE_MATTER,
    3: TissueType.GRAY_MATTER,
    42: TissueType.GRAY_MATTER,
    4: TissueType.CSF,     # lateral ventricle
    24: TissueType.CSF,    # CSF
}
```

### Loading Tumor Contour

```python
# From manual segmentation
tumor_mask = nib.load("tumor_segmentation.nii.gz").get_fdata()
tumor_slice = tumor_mask[:, :, slice_idx]

# Convert binary mask to initial density
# Option A: Binary (sharp boundary)
initial_density = tumor_slice.astype(float)

# Option B: Smoothed (more realistic infiltrative edge)
from scipy.ndimage import gaussian_filter
initial_density = gaussian_filter(tumor_slice.astype(float), sigma=2.0)
```

### Loading Vessel Data

```python
# From contrast-enhanced MRI or perfusion map
vessel_img = nib.load("vessel_mask.nii.gz").get_fdata()
vessel_slice = vessel_img[:, :, slice_idx] > 0.5  # Threshold to binary
```

### Putting It Together

```python
from tumor_growth_rbf import TumorModel, TumorParameters, TissueParameters

# Physical size of your slice (from DICOM/NIfTI header)
# pixel_spacing = img.header.get_zooms()[:2]  # mm per pixel
# domain_size = (tissue_slice.shape[0] * pixel_spacing[0],
#                tissue_slice.shape[1] * pixel_spacing[1])
domain_size = (120.0, 120.0)  # Example: 120mm × 120mm

model = TumorModel(
    domain_size=domain_size,
    params=TumorParameters(
        growth_rate=0.012,      # From clinical doubling time
        diffusion_white=0.1,    # mm²/day
        diffusion_grey=0.01,    # mm²/day
    ),
    n_initial_points=2000  # Higher for clinical use
)

# Load tissue data
model.load_tissue_data(tissue_slice, tissue_labels, vessel_slice)
```

---

## Parameter Selection

### Growth Rate

The growth rate `ρ` can be estimated from clinical tumor doubling time:

```
ρ = ln(2) / T_doubling
```

| Tumor Grade | Typical Doubling Time | Growth Rate (day⁻¹) |
|-------------|----------------------|---------------------|
| Low-grade glioma (WHO II) | 200–400 days | 0.002–0.004 |
| Anaplastic glioma (WHO III) | 50–150 days | 0.005–0.014 |
| Glioblastoma (WHO IV) | 20–80 days | 0.009–0.035 |

**Source:** Harpold et al., "The evolution of mathematical modeling of glioma proliferation and invasion" (J Neuropathol Exp Neurol, 2007)

### Diffusion Coefficients

| Tissue Type | Diffusion (mm²/day) | Source |
|-------------|---------------------|--------|
| White matter | 0.05–0.50 | Swanson et al. (2000, 2008) |
| Gray matter | 0.005–0.05 | Typically D_gray ≈ D_white / 10 |

The ratio D_white/D_gray (typically 5–10) is often more important than absolute values.

**From clinical imaging:** If you have DTI data, the mean diffusivity map gives local diffusion estimates. Higher mean diffusivity correlates with faster tumor spread.

### Radiation Parameters (Linear-Quadratic Model)

| Parameter | Tumors | Late-responding normal tissue |
|-----------|--------|------------------------------|
| α (Gy⁻¹) | 0.10–0.35 | 0.01–0.10 |
| β (Gy⁻²) | 0.01–0.05 | 0.01–0.07 |
| α/β ratio | 8–15 Gy | 1–5 Gy |

**Standard fractionation schedules:**

| Regimen | Fractions | Dose/fraction | Total | Typical Use |
|---------|-----------|---------------|-------|-------------|
| Conventional | 30 | 2.0 Gy | 60 Gy | GBM standard |
| Hypofractionated | 15 | 2.67 Gy | 40 Gy | Elderly/poor KPS |
| Radiosurgery | 1–5 | 8–20 Gy | 8–30 Gy | Small tumors, metastases |

**Source:** McMahon, "The linear quadratic model: usage, interpretation and challenges" (Phys. Med. Biol., 2019)

### Chemotherapy Parameters

Temozolomide (TMZ), the standard GBM chemotherapy:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Drug sensitivity | 0.1–0.3 | Varies by MGMT methylation status |
| Cycle length | 28 days | Standard: 5/28 day cycle |
| Drug decay | 0.1–0.3 day⁻¹ | TMZ half-life ~1.8 hours in plasma |
| S-phase specificity | 2.0× | TMZ primarily targets S-phase |

### Oxygen Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hypoxia threshold | 0.05–0.15 | pO₂ < 10 mmHg |
| OER | 2.0–3.0 | Oxygen Enhancement Ratio for radiation |
| Consumption rate | 0.05–0.20 | Depends on metabolic activity |

---

## Running Simulations

### Basic Growth Prediction

```python
from tumor_growth_rbf import TumorModel, TumorParameters

model = TumorModel(
    domain_size=(100.0, 100.0),  # 100mm × 100mm slice
    params=TumorParameters(
        growth_rate=0.012,       # GBM-like
        diffusion_white=0.1,
    ),
    n_initial_points=1000
)

# Simulate 180 days (6 months)
dt = 0.5  # 12-hour timestep (larger dt = faster but less accurate)
metrics_over_time = []

for day in range(360):  # 360 half-day steps = 180 days
    model.update(dt)
    if day % 30 == 0:  # Record every 15 days
        metrics_over_time.append({
            'day': day * dt,
            'metrics': model.get_metrics()
        })
```

### Standard Radiation Course (Stupp Protocol)

```python
# Stupp protocol: 60 Gy in 30 fractions + concurrent TMZ
dt = 0.1  # Smaller timestep for treatment accuracy

for day_step in range(600):  # 60 days
    model.update(dt)
    day = day_step * dt

    # Radiation: weekdays for 6 weeks
    week = int(day) // 7
    day_of_week = int(day) % 7
    if week < 6 and day_of_week < 5:  # Mon-Fri, weeks 0-5
        if abs(day - round(day)) < dt/2:  # Once per day
            model.apply_treatment("radiation", dose=2.0)

    # Concurrent TMZ: daily during radiation
    if week < 6:
        if abs(day - round(day)) < dt/2:
            model.apply_treatment("chemo", drug_amount=0.3)
```

### Comparing Treatment Schedules

```python
import copy

# Create two identical models
model_standard = TumorModel(domain_size=(50.0, 50.0), n_initial_points=500)
model_hypo = copy.deepcopy(model_standard)

# Grow both for 30 days
for _ in range(300):
    model_standard.update(0.1)
    model_hypo.update(0.1)

# Standard: 2 Gy × 30 fractions
for frac in range(30):
    model_standard.apply_treatment("radiation", dose=2.0)
    for _ in range(10):
        model_standard.update(0.1)  # 1 day between fractions

# Hypofractionated: 2.67 Gy × 15 fractions
for frac in range(15):
    model_hypo.apply_treatment("radiation", dose=2.67)
    for _ in range(10):
        model_hypo.update(0.1)

# Compare outcomes
print(f"Standard: {model_standard.get_metrics()['tumor']['total_mass']:.1f}")
print(f"Hypo:     {model_hypo.get_metrics()['tumor']['total_mass']:.1f}")
```

---

## Interpreting Results

### Key Metrics

| Metric | What It Means | Clinical Relevance |
|--------|--------------|-------------------|
| `total_mass` | Integrated tumor density | Correlates with tumor volume |
| `max_density` | Peak local density | Indicates most aggressive region |
| `hypoxic_fraction` | Fraction of domain with low O₂ | Predicts radiation resistance |
| `g1_fraction` through `m_fraction` | Cell cycle distribution | Treatment sensitivity window |
| `quiescent_fraction` | Dormant cell proportion | Treatment resistance reservoir |
| `necrotic_fraction` | Dead cell proportion | Indicates tumor maturity |
| `proliferating_fraction` | Actively dividing cells | Tumor aggressiveness |

### What to Look For

**Growth dynamics:**
- Exponential early growth transitioning to logistic saturation
- Development of hypoxic core with proliferating rim
- Faster spread along white matter tracts (if tissue data loaded)

**Treatment response:**
- Immediate mass reduction after radiation (proportional to dose)
- Greater kill in G2/M phases (check phase fraction changes)
- Tumor regrowth rate after treatment (residual proliferating cells)
- Drug concentration decay between chemo cycles

**Red flags in your simulation:**
- Tumor density exceeding carrying capacity → reduce timestep
- Negative densities appearing → reduce timestep
- Mass increasing during pure diffusion → numerical instability

---

## Treatment Planning Workflows

### Workflow 1: Growth Rate Estimation

Given two MRI scans at different times:

1. Measure tumor radius at both timepoints
2. Estimate velocity of the visible tumor boundary: `v = Δr / Δt`
3. The growth rate relates to velocity via: `v = 2 × sqrt(D × ρ)`
4. With an assumed D (from tissue type), solve for ρ

```python
import numpy as np

# From two MRI scans
radius_1 = 15.0  # mm at time 1
radius_2 = 20.0  # mm at time 2
delta_t = 90.0   # days between scans

velocity = (radius_2 - radius_1) / delta_t  # mm/day
D_assumed = 0.1  # mm²/day (white matter)

# v = 2*sqrt(D*rho) → rho = v²/(4*D)
rho_estimated = velocity**2 / (4 * D_assumed)
print(f"Estimated growth rate: {rho_estimated:.4f} day⁻¹")
print(f"Doubling time: {np.log(2)/rho_estimated:.0f} days")
```

### Workflow 2: Fractionation Optimization

```python
# Compare different fraction sizes for same total BED
from tumor_growth_rbf import TumorModel
import numpy as np

alpha_beta = 10.0  # Gy for tumors
target_bed = 72.0  # Gy₁₀ (standard 60 Gy / 30 fx = 72 BED)

fraction_sizes = [1.5, 1.8, 2.0, 2.5, 3.0]
results = {}

for d in fraction_sizes:
    # Calculate number of fractions for equivalent BED
    # BED = n*d*(1 + d/(α/β))
    n = int(target_bed / (d * (1 + d / alpha_beta)))
    total_dose = n * d

    model = TumorModel(domain_size=(10.0, 10.0), n_initial_points=300)
    for _ in range(50):
        model.update(0.1)  # Grow 5 days

    pre = model.get_metrics()['tumor']['total_mass']
    for frac in range(n):
        model.apply_treatment("radiation", dose=d)
        model.update(0.1)
    post = model.get_metrics()['tumor']['total_mass']

    results[d] = {
        'n_fractions': n,
        'total_dose': total_dose,
        'reduction': (1 - post/pre) * 100
    }
    print(f"  {d:.1f} Gy × {n} fx = {total_dose:.0f} Gy total → "
          f"{results[d]['reduction']:.0f}% reduction")
```

---

## Limitations and Caveats

### Model Limitations

1. **2D only** — Real tumors are 3D. The 2D model captures qualitative behavior but not volumetric predictions.
2. **No mechanical coupling** — Does not model tissue deformation, mass effect, or increased intracranial pressure.
3. **Simplified vasculature** — Static vessel map; no angiogenesis or vascular co-option.
4. **Isotropic diffusion** — White matter diffusion is actually anisotropic (along fiber tracts). DTI integration is planned.
5. **Fixed cell cycle parameters** — In reality, these vary spatially and evolve with treatment.
6. **No blood-brain barrier** — Drug delivery model is simplified; does not account for BBB permeability.

### Numerical Considerations

- **Timestep selection:** Use `dt ≤ 0.1` days for treatment simulations, `dt ≤ 0.5` days for growth-only.
- **Point density:** 500–1000 points for exploratory work; 2000–5000 for publication-quality results.
- **RBF shape parameter (ε):** Default ε=1.0 works for most cases. If you see oscillations, try ε=0.5–2.0.

### Validation Status

The simulator has been validated for:
- ✅ Polynomial reproduction (exact for degree ≤ 2)
- ✅ Positivity preservation
- ✅ Carrying capacity enforcement
- ✅ Qualitatively correct cell cycle dynamics
- ✅ Phase-specific treatment sensitivity ordering (M > G2 > G1 > Q > S for radiation)

Not yet validated against:
- ⬜ Clinical patient datasets
- ⬜ In vitro cell culture growth curves
- ⬜ Published benchmark problems (e.g., Fisher-KPP exact solutions)

---

## Literature References

### Tumor Growth Modeling
- Swanson KR et al. "A quantitative model for differential motility of gliomas in grey and white matter." *Cell Proliferation* 33(5):317-329, 2000.
- Harpold HLP et al. "The evolution of mathematical modeling of glioma proliferation and invasion." *J Neuropathol Exp Neurol* 66(1):1-9, 2007.
- Swanson KR et al. "A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma." *Br J Cancer* 98(1):113-119, 2008.

### Radiation Biology
- McMahon SJ. "The linear quadratic model: usage, interpretation and challenges." *Phys Med Biol* 64(1):01TR01, 2019.
- Joiner MC & van der Kogel AJ. *Basic Clinical Radiobiology*. 5th ed. CRC Press, 2018.

### RBF-FD Methods
- Fornberg B & Flyer N. *A Primer on Radial Basis Functions with Applications to the Geosciences.* SIAM, 2015.
- Bayona V et al. "On the role of polynomials in RBF-FD approximations." *J Comput Phys* 348:21-38, 2017.

### Cell Cycle and Treatment
- Shah MA & Schwartz GK. "Cell cycle-mediated drug resistance: an emerging concept in cancer therapy." *Clin Cancer Res* 7(8):2168-2181, 2001.
- Stupp R et al. "Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma." *N Engl J Med* 352(10):987-996, 2005.
