# Variational Quantum Classifier

This repository implements a modular and extensible **Variational Quantum
Classifier (VQC)**. Built with [PennyLane](https://pennylane.ai/) and
[PyTorch](https://pytorch.org/), it serves as a template for hybrid
quantum–classical machine learning experiments. The project is engineered to
be **scalable**, **robust**, and easy to extend, featuring a clean package
structure, unit tests, and continuous integration support.

## Project Structure

```
vqc_project/
├── vqc/                 # Python package with source code
│   ├── __init__.py
│   ├── data.py          # Synthetic data generation & preprocessing
│   ├── circuit.py       # Embedding & ansatz primitives
│   ├── model.py         # Variational quantum classifier class
│   └── train.py         # Command‑line training script
├── tests/               # PyTest unit tests
│   └── test_model.py
├── pyproject.toml       # Build & dependency specification
└── README.md            # You are here
```

## Installation

The package follows PEP 517/518 standards and can be installed in editable
mode (recommended for development) as follows:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

This installs runtime dependencies (`numpy`, `pennylane`, `torch`,
`scikit-learn`) along with development extras (`pytest`). To install
dependencies without test tools, run:

```bash
pip install -e .
```

## Training a Classifier

The `train.py` module provides a convenient CLI for training a variational
quantum classifier on synthetic data:

```bash
python -m vqc.train --n-qubits 4 --layers 2 --embedding angle --dataset moons \
    --epochs 200 --shots 1000 --noise 0.2
```

Key command‑line flags:

| Flag           | Description                                                  | Default |
|---------------|--------------------------------------------------------------|--------|
| `--dataset`    | Dataset type: `moons` (requires `scikit‑learn`) or `blobs` | moons  |
| `--n-qubits`   | Number of qubits / features                                 | 4      |
| `--layers`     | Number of variational layers                                | 2      |
| `--embedding`  | Data embedding scheme: `angle` or `amplitude`               | angle  |
| `--shots`      | Measurement shots (0 for analytic expectation)              | 1000   |
| `--epochs`     | Training epochs                                             | 200    |
| `--lr`         | Learning rate                                              | 0.05   |
| `--train-size` | Number of training samples                                  | 600    |
| `--test-size`  | Number of test samples                                      | 400    |
| `--noise`      | Noise level for synthetic data                              | 0.2    |
| `--output-dir` | Directory to save the trained model                         | artifacts |

During training the script logs loss and accuracy and saves the model
parameters (`state_dict`) to `vqc_model.pt` in the specified output directory.

## Advanced Features

This project includes several advanced implementations to improve quantum classifier performance:

### 1. Enhanced Models

* **Noise-Robust VQC:** Implementation with error mitigation techniques (`noise_robust_model.py`)
* **Hardware-Efficient Circuits:** Optimized ansatz designs for better hardware compatibility (`hardware_efficient.py`)
* **Feature Engineering:** Advanced data preprocessing specialized for quantum embeddings (`feature_engineering.py`)
* **Hyperparameter Tuning:** Grid search for optimal model configuration (`hyperparameter_tuning.py`)

### 2. Advanced Training

An enhanced training script (`advanced_train.py`) is provided with features including:

* **Complex Datasets:** Beyond simple moons/blobs with nonlinear, spirals, XOR patterns
* **Feature Engineering:** PCA, Kernel PCA, Fourier features, angle-optimized encodings
* **Real-world Datasets:** Support for UCI datasets like Iris, Wine, Breast Cancer
* **Noise Mitigation:** Zero-noise extrapolation and other techniques
* **Optimization Strategies:** Learning rate scheduling, early stopping, regularization
* **Hardware-Efficient Ansätze:** Specialized circuit designs with optimized gate counts

```bash
python -m vqc.advanced_train --n-qubits 4 --layers 2 --model-type noise-robust \
    --dataset xor --embedding amplitude --feature-eng fourier --epochs 50 \
    --plot --early-stopping
```

### 3. Hyperparameter Tuning

A comprehensive hyperparameter tuning script to find optimal configurations:

```bash
python -m vqc.hyperparameter_tuning
```

This will test various combinations of parameters and save the results in an experiments directory.

## Extending the Project

This template is designed with flexibility in mind:

* **Custom embeddings:** Pass your own embedding callable to the
  `VariationalQuantumClassifier` constructor to implement alternative
  feature encodings.
* **Entanglement patterns:** Provide a custom entanglement function via
  the `entanglement` argument (e.g., nearest‑neighbour, all‑to‑all).
* **Datasets:** Swap out the synthetic data generators in `data.py` with
  functions that load your own datasets (e.g., from CSV files or quantum
  simulators). Normalisation and dimension matching utilities are provided.
* **Hyperparameters:** Adjust the CLI or training script to expose
  additional hyperparameters such as regularisation strength, optimiser
  choice, or early stopping criteria.

## Testing and Continuous Integration

Unit tests live in the `tests/` directory and can be executed with
`pytest`:

```bash
pytest -q
```

The provided example tests ensure that the model forwards run for both
embedding types. To maintain robustness, you can add more tests covering
training convergence, gradient correctness, and integration with
different datasets.

For continuous integration on GitHub, you can add a workflow under
`.github/workflows` similar to the following:

```yaml
name: Python package

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      - name: Run tests
        run: pytest -q
```

This configuration will automatically install dependencies and run tests on
each push or pull request, helping you maintain code quality.

## Recommended CV Blurb

> *Developed a scalable variational quantum classifier with a modular
> architecture, supporting multiple data embeddings and configurable
> entanglement patterns. Integrated PennyLane QNodes with PyTorch autograd for
> seamless hybrid optimisation. Implemented unit tests, packaging via
> pyproject.toml, and continuous integration to ensure reproducible and
> maintainable research software.*
