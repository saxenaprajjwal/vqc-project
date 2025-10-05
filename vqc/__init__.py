"""Top‑level package for the Variational Quantum Classifier.

This package provides utilities for generating datasets, building variational
quantum circuits, defining hybrid quantum–classical models, and training
classifiers. To run a training session from the command line use

```
python -m vqc.train --help
```

or import and use the classes/functions programmatically.
"""

from .data import generate_data, normalize_features, match_dimensions  # noqa: F401
from .model import VariationalQuantumClassifier  # noqa: F401
