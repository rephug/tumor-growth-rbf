# Tumor Growth RBF Simulator

A high-performance, meshless tumor growth simulation framework using Radial Basis Function-generated Finite Differences (RBF-FD). This project implements a comprehensive biological model for simulating tumor growth dynamics, incorporating immune response, treatment effects, and adaptive mesh refinement.

## Key Features

- **Meshless PDE Solver**: Uses RBF-FD method for flexible spatial adaptivity
- **Comprehensive Biological Model**:
  - Tumor growth and diffusion
  - Oxygen-dependent dynamics
  - Immune system response
  - Multiple treatment modalities (radiation, chemotherapy, immunotherapy)
- **Adaptive Refinement**: Dynamic point distribution based on tumor activity
- **Treatment Planning**: Supports various treatment scheduling and optimization

## Technical Details

The simulator combines:
- RBF-FD for spatial discretization
- Adaptive mesh refinement for computational efficiency
- Reaction-diffusion equations for tumor dynamics
- Immune system and treatment response models

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
from tumor_growth_rbf import TumorModel

# Initialize model
domain_size = (10.0, 10.0)  # 10mm x 10mm domain
model = TumorModel(domain_size)

# Simulate for 10 days
dt = 0.1  # time step
for t in range(100):
    model.update(dt)
    metrics = model.get_metrics()
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
