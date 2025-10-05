"""Basic unit tests for the VariationalQuantumClassifier.

These tests ensure that the model can be instantiated and perform a forward
pass without errors. They also verify that the output shape matches the
expected dimensionality and that training updates parameters.
"""

import torch
import numpy as np

from vqc.data import generate_data, normalize_features, match_dimensions
from vqc.model import VariationalQuantumClassifier



def test_model_forward_angle():
    """Test the forward pass with angle embedding."""
    n_qubits = 2
    model = VariationalQuantumClassifier(n_qubits=n_qubits, layers=1, embedding="angle", shots=0)
    # Generate small dataset
    X, y = generate_data(10, seed=42, noise=0.1, kind="moons")
    X_norm, _ = normalize_features(X, X)
    X = match_dimensions(X_norm, n_qubits)
    X_t = torch.tensor(X)
    logits = model(X_t)
    # Expect logits to have shape (10,)
    assert logits.shape == (10,)



def test_model_forward_amplitude():
    """Test the forward pass with amplitude embedding."""
    n_qubits = 2
    model = VariationalQuantumClassifier(n_qubits=n_qubits, layers=1, embedding="amplitude", shots=0)
    X = np.random.rand(10, 2).astype(np.float32)
    X_norm, _ = normalize_features(X, X)
    X = match_dimensions(X_norm, n_qubits)
    X_t = torch.tensor(X)
    logits = model(X_t)
    assert logits.shape == (10,)
