Below is a revised **README** template reflecting your requests: it now references the **Apache License 2.0**, includes your **personal details**, and **removes** any reference to the VSCode project export extension. Feel free to adapt and refine it as necessary.

---

# Tumor Growth RBF Simulator

A **high-performance, meshless tumor growth simulation framework** leveraging **Radial Basis Function-generated Finite Differences (RBF-FD)**. This repository integrates complex biological modeling (immune response, multi-phase cell cycle, treatment interventions) with advanced numerical methods (adaptive mesh refinement, PDE solvers) to study tumor growth in 2D.

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage Examples](#usage-examples)  
6. [Documentation & Tutorials](#documentation--tutorials)  
7. [Testing & Validation](#testing--validation)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Contact](#contact)

---

## Overview

Accurate simulation of tumor growth is critical for advancing our understanding of cancer progression and treatment planning. Traditional numerical methods (e.g., finite elements) often require cumbersome mesh generation and can struggle with dynamic interfaces. **RBF-FD** (Radial Basis Function Finite Differences) offers a more flexible meshless approach, allowing local point refinement, simpler PDE assembly, and robust handling of irregular domains.

This simulator models:

- Tumor cell populations with **cell cycle phases** (G1, S, G2, M, Q, Necrotic).
- **Immune response** via chemokine signaling, immune cell infiltration, and tumor-immune interactions.
- Multiple **treatment modalities**—radiation, chemotherapy, immunotherapy—with **phase-specific** sensitivities.
- **Tissue heterogeneity**, capturing white matter, gray matter, necrotic tissue, and vasculature differences.
- **Adaptive mesh refinement** to efficiently capture tumor boundary evolution and other significant spatial gradients.

We aim to facilitate research by offering a modular, extensible framework that biologists, clinicians, and computational scientists can customize.

---

## Key Features

- **Meshless PDE Solver**  
  Utilizes RBF-FD for spatial discretization. Eliminates the need for a structured mesh, simplifying domain setup.

- **Cell Cycle Modeling**  
  Detailed cell cycle representation (G1, S, G2, M, Q, N) with transitions governed by oxygen availability, treatment pressures, and biological rates.

- **Immune Response Module**  
  Models immune cell recruitment, chemokine diffusion, and tumor–immune cell interactions.

- **Multi-Modal Treatment**  
  Radiation, chemotherapy, and immunotherapy can be applied separately or combined. Treatment scheduling, dosing, and synergy all considered.

- **Adaptive Refinement**  
  Dynamically adds or removes points based on tumor gradients, curvature, or hypoxic regions for computational efficiency and accuracy.

- **Tissue-Specific Properties**  
  Tissue maps enable different diffusion coefficients, growth modifiers, and oxygen perfusion rates for white matter, gray matter, CSF, vessels, etc.

- **Validation & Visualization**  
  Built-in **test suites** (with `pytest`) and advanced **visualization** (matplotlib) for analyzing simulation results, tumor density, oxygen, and cell populations over time.

---

## Project Structure

```plaintext
tumor-growth-rbf/
├── src/
│   └── tumor_growth_rbf/
│       ├── biology/
│       │   ├── cell_populations.py      (Cell cycle logic)
│       │   ├── immune_response.py       (Immune infiltration & killing)
│       │   ├── treatments.py           (Radiation/chemo/immunotherapy)
│       │   ├── tumor_model.py          (Integrates everything into a main class)
│       │   └── ... (other biology modules)
│       ├── core/
│       │   ├── rbf_solver.py           (RBF-FD solver implementation)
│       │   ├── pde_assembler.py        (Build PDE operators)
│       │   ├── mesh_handler.py         (Basic mesh handling)
│       │   └── boundary_conditions.py  (BC management)
│       ├── utils/
│       │   ├── optimization.py         (Performance & time stepping)
│       │   ├── validation.py           (Validation and metrics)
│       │   └── visualization.py        (Visualizer tools)
│       ├── __init__.py                (Package init, exports main classes)
│       └── ...
├── tests/
│   ├── test_tumor_cell_populations.py
│   ├── test_tumor_model.py
│   └── ...
├── setup.py
├── README.md
├── LICENSE
└── ...
```

---

## Installation

### 1. Clone the Repo

```bash
git clone https://github.com/rephug/tumor-growth-rbf.git
cd tumor-growth-rbf
```

### 2. Python Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # For Linux/Mac
# or
.\venv\Scripts\activate    # For Windows
```

### 3. Install Python Dependencies

```bash
pip install -e .
```
This installs the package (`tumor_growth_rbf`) in editable mode along with its core dependencies (NumPy, SciPy, Matplotlib, etc.).

### 4. (Optional) Install Development Dependencies

If you plan to run tests or work on the source code:

```bash
pip install -r dev_requirements.txt
```

(This file might include `pytest`, `black`, `flake8`, `isort`, etc.)

---

## Usage Examples

Below are some quick examples to get you started. For more detailed tutorials, see the [Documentation & Tutorials](#documentation--tutorials).

### Basic Tumor Growth Simulation

```python
import numpy as np
from tumor_growth_rbf import TumorModel

# Initialize model with a 10x10 domain
model = TumorModel(domain_size=(10.0, 10.0))

# Time-step for 10 days in increments of 0.1
dt = 0.1
num_steps = int(10.0 / dt)
for step in range(num_steps):
    model.update(dt)
    if step % 10 == 0:
        metrics = model.get_metrics()
        print(f"Day {step*dt:.1f} -> Total tumor mass: {metrics['tumor']['total_mass']:.2f}")

# Final metrics
final_metrics = model.get_metrics()
print("Final tumor mass:", final_metrics['tumor']['total_mass'])
```

### Applying Treatments

```python
# For instance, apply 2 Gy of radiation at day 5
if step*dt == 5.0:
    model.apply_treatment("radiation", dose=2.0)
```

### Visualization

```python
from tumor_growth_rbf.utils.visualization import TumorVisualizer
import matplotlib.pyplot as plt

viz = TumorVisualizer(model)
fig = viz.create_state_visualization(time=10.0)
plt.show()
```

---

## Documentation & Tutorials

### 1. API Documentation

- You can generate full API docs (e.g., using Sphinx):
  ```bash
  cd docs
  make html
  ```
  Then open `docs/_build/html/index.html` in your browser.

### 2. Tutorials / Notebooks

- `examples/` folder (if available) hosts Jupyter notebooks demonstrating:
  - Basic tumor simulation
  - Immune system modeling
  - Applying multi-modal treatment
  - Real-time mesh refinement examples

### 3. Reference Papers

If you’re new to RBF-FD or tumor growth modeling, here are a few references:
- [Fornberg & Flyer, “A Primer on Radial Basis Functions with Applications to the Geosciences”](https://epubs.siam.org/doi/book/10.1137/1.9781611974041)
- [Wise, Lowengrub, Frieboes, Cristini: “Three-dimensional multispecies nonlinear tumor growth–I Model and numerical method” (2008)](https://www.sciencedirect.com/science/article/pii/S0021999108001301)

---

## Testing & Validation

We rely on **pytest** for testing. You can run all tests by:

```bash
pytest tests/
```

Key test categories include:
- **Cell Population Tests**: Validating cell cycle transitions, oxygen-dependent quiescence, necrosis triggers, etc.
- **Tumor Model Integration**: Checking mass conservation, positivity, mesh adaptivity, boundary conditions.
- **Treatment Tests**: Ensuring correct dose distribution for radiation or drug concentration for chemotherapy.
- **Parameter Sensitivity & Validation**: The `ModelValidator` can compare simulation outputs to experimental or theoretical benchmarks.

For coverage, install `pytest-cov` and run:
```bash
pytest --cov=tumor_growth_rbf --cov-report=term-missing
```

---

## Contributing

Contributions are welcome! To add features, fix bugs, or improve the documentation, please:

1. **Fork** the repository  
2. **Create a new branch** for your feature/fix  
3. **Commit** your changes with descriptive messages  
4. **Push** to your fork  
5. **Create a Pull Request** on GitHub  

Ensure your PR passes all tests and follows **code style** guidelines (e.g., via `black` or `flake8`).

---

## License

This project is licensed under the **Apache License, Version 2.0**.  
A copy of the license is available in the [LICENSE](LICENSE) file, or you can read the official text at [Apache.org](https://www.apache.org/licenses/LICENSE-2.0).

---

## Contact

- **Author**: [Robert Fuge](https://github.com/rephug)  
- **Email**: [rephug@gmail.com](mailto:rephug@gmail.com)

We hope you find this simulator useful for your research and applications in computational oncology. If you use this work in a scientific publication, please consider citing it or referencing the code. Thank you!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
