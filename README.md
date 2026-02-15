# Tumor Growth RBF-FD Simulator

A meshless tumor growth simulation framework for computational oncology research, using Radial Basis Function-generated Finite Differences (RBF-FD).

## What This Tool Does

This simulator models brain tumor (glioma) growth and treatment response in 2D and 3D, incorporating:

- **Cell cycle dynamics** — G1, S, G2, M phases with oxygen-dependent transitions to quiescence and necrosis
- **Tissue heterogeneity** — different diffusion rates in white matter vs. gray matter (from MRI segmentation)
- **Treatment modeling** — radiation (Linear-Quadratic model), chemotherapy (phase-specific drug effects), and immunotherapy (checkpoint inhibitor model)
- **Immune response** — chemokine-mediated recruitment, infiltration, and tumor-immune interactions
- **Adaptive mesh refinement** — automatic resolution increase at tumor boundaries
- **3D support** — full 3D meshless simulation with spherical domains
- **Patient-specific parameter fitting** — grid search over (ρ, D) using Dice coefficient with optional Nelder-Mead refinement
- **Analytical validation** — 5 benchmarks against known solutions (Fisher-KPP, diffusion, exponential growth, LQ model, symmetry)

## Who This Is For

- **Computational oncology researchers** studying tumor growth dynamics
- **Radiation oncologists** exploring fractionation schedules and treatment timing
- **Graduate students** learning PDE-based biological modeling
- **Pharmacologists** modeling drug delivery and cell-cycle-specific effects

## Quick Start

```bash
git clone https://github.com/rephug/tumor-growth-rbf.git
cd tumor-growth-rbf
pip install -e .
python demo.py  # Runs 5 interactive learning demos
```

### Minimal Simulation

```python
from tumor_growth_rbf import TumorModel

model = TumorModel(domain_size=(10.0, 10.0))  # 10mm × 10mm

# Simulate 10 days
for step in range(100):
    model.update(dt=0.1)

# Apply 2 Gy radiation
result = model.apply_treatment("radiation", dose=2.0)
print(f"Cells killed: {result['total_cells_killed']:.0f}")

# Get metrics
metrics = model.get_metrics()
print(f"Tumor mass: {metrics['tumor']['total_mass']:.1f}")
print(f"Hypoxic fraction: {metrics['tumor']['hypoxic_fraction']:.1%}")
```

## Data Requirements

See `docs/CLINICAL_GUIDE.md` for complete data preparation instructions.

### Minimum (no imaging data needed)
The simulator runs with synthetic initial conditions out of the box. You only need to specify:
- Domain size (mm)
- Growth rate (day⁻¹) — from literature or clinical doubling time
- Treatment schedule

### With Medical Imaging
For patient-specific simulations, you can provide:
- **Tissue segmentation map** — from T1/T2 MRI (white matter, gray matter, CSF labels)
- **Vessel map** — from contrast-enhanced MRI or MRA (binary mask)
- **Initial tumor contour** — from T1-contrast or FLAIR MRI

## Project Structure

```
tumor-growth-rbf/
├── src/tumor_growth_rbf/
│   ├── core/              # Numerical foundation
│   │   ├── mesh_handler.py    # Scattered point management
│   │   ├── rbf_solver.py      # RBF-FD weight computation
│   │   └── pde_assembler.py   # PDE operator assembly
│   ├── biology/           # Biological models
│   │   ├── tumor_model.py     # Main simulation engine
│   │   ├── cell_populations.py # Cell cycle model
│   │   ├── treatments.py      # Radiation/chemo/immunotherapy
│   │   ├── immune_response.py # Immune system dynamics
│   │   └── tissue_properties.py # Tissue-specific parameters
│   └── utils/             # Visualization and tools
│       ├── visualization.py
│       └── parameter_fitting.py
├── tests/                 # Test suite (63 tests)
│   ├── test_all.py            # Unit tests
│   ├── test_benchmarks.py     # Analytical validation benchmarks
│   └── test_parameter_fitting.py  # Parameter fitting tests
├── docs/                  # Documentation
│   └── CLINICAL_GUIDE.md  # Data preparation and clinical usage
├── demo.py                # Interactive learning demos
└── examples/              # Clinical workflow examples
    └── clinical_workflow.py
```

## Key Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `growth_rate` | 0.1 | day⁻¹ | Tumor proliferation rate |
| `carrying_capacity` | 1.0 | — | Max normalized density |
| `diffusion_white` | 0.1 | mm²/day | Diffusion in white matter |
| `diffusion_grey` | 0.01 | mm²/day | Diffusion in grey matter |
| `hypoxia_threshold` | 0.1 | — | O₂ level for quiescence |
| `fractionation_alpha` | 0.15 | Gy⁻¹ | LQ model α parameter |
| `fractionation_beta` | 0.05 | Gy⁻² | LQ model β parameter |

## Testing

```bash
pip install pytest
pytest tests/ -v
```

All 63 tests cover: mesh operations, RBF-FD accuracy, cell cycle biology, treatment effects, positivity, carrying capacity enforcement, 3D operations, parameter fitting, and analytical validation benchmarks.

### Analytical Validation Benchmarks

| Benchmark | Metric | Error | Tolerance |
|-----------|--------|-------|-----------|
| Fisher-KPP growth-diffusion | Mass growth rate vs analytical | 0.32% | < 5% |
| Pure diffusion (Gaussian) | Width σ(t) = √(σ₀² + 2Dt) | 0.25% | < 2% |
| Exponential growth | Density u₀exp(ρt) | 0.07% | < 1% |
| LQ cell survival | SF = exp(-αd - βd²) | ~10⁻¹⁴ | < 1% |
| Radial symmetry | RMS deviation from radial Gaussian | 0.15% | < 5% |

## License

Apache License 2.0

## Citation

If you use this software in research, please cite:
```
Fuge, R. (2025). Tumor Growth RBF-FD Simulator: A meshless framework
for computational oncology. https://github.com/rephug/tumor-growth-rbf
```

## References

- Fornberg & Flyer, "A Primer on Radial Basis Functions with Applications to the Geosciences" (SIAM, 2015)
- Swanson et al., "A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma" (British J. Cancer, 2008)
- McMahon, "The linear quadratic model: usage, interpretation and challenges" (Phys. Med. Biol., 2019)
