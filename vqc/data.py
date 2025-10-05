"""Data generation and preprocessing utilities for quantum classifiers.

This module contains helper functions to create synthetic datasets for binary
classification, normalise features to a fixed range, and match feature
dimensions to the number of qubits in a quantum circuit. The functions are
designed to be standalone so that they can be re-used or extended to load
custom datasets.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = ["generate_data", "normalize_features", "match_dimensions"]


def generate_data(
    n: int,
    noise: float = 0.2,
    seed: int = 0,
    kind: str = "moons",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a two-dimensional binary classification dataset.

    This function supports multiple dataset types. Currently available:

    * ``moons`` – two interleaving half circles (requires ``scikit-learn``).
    * ``blobs`` – two noisy clusters on a unit circle (implemented as fallback).

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise : float
        Standard deviation of Gaussian noise added to the data.
    seed : int
        Random seed for reproducibility.
    kind : str
        Type of dataset to generate (``"moons"`` or ``"blobs"``).

    Returns
    -------
    X : ndarray
        Feature matrix of shape ``(n, 2)``.
    y : ndarray
        Binary class labels of shape ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    kind = kind.lower()
    if kind == "moons":
        try:
            from sklearn.datasets import make_moons  # type: ignore

            X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
        except Exception:
            # Fall back to blobs if scikit-learn is unavailable
            kind = "blobs"
    if kind == "blobs":
        r = 1.0 + 0.3 * rng.standard_normal(n)
        theta = 2 * np.pi * rng.random(n)
        X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        X += noise * rng.standard_normal(X.shape)
    return X.astype(np.float32), y.astype(np.int64)


def normalize_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise features to the range ``[-1, 1]`` on a per-feature basis.

    This is important for quantum circuits that embed data via rotation angles,
    ensuring that inputs lie within a consistent range. The normalisation
    parameters are computed from the training data and applied to both the
    training and test sets.
    """
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train_norm = 2 * (X_train - X_min) / (X_max - X_min + 1e-8) - 1
    X_test_norm = 2 * (X_test - X_min) / (X_max - X_min + 1e-8) - 1
    return X_train_norm, X_test_norm


def match_dimensions(X: np.ndarray, d: int) -> np.ndarray:
    """Match the number of feature columns to ``d``.

    If the input data has fewer than ``d`` features, the columns are tiled
    until the desired dimension is reached. If the input has more than ``d``
    features, the extra columns are truncated.
    """
    m = X.shape[1]
    if m == d:
        return X
    if m > d:
        return X[:, :d]
    rep = int(math.ceil(d / m))
    X_rep = np.tile(X, (1, rep))[:, :d]
    return X_rep
