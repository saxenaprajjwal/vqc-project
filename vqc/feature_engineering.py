"""Enhanced data handling and feature engineering for quantum classifiers.

This module extends the basic data generation capabilities with:
1. More sophisticated feature engineering techniques 
2. Support for complex datasets
3. Dimensionality reduction methods optimized for quantum embedding
4. Data augmentation strategies
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Callable, Union
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from .data import normalize_features, match_dimensions


class QuantumFeatureEngineering:
    """Feature engineering techniques optimized for quantum embedding.
    
    This class provides methods to transform classical data into forms
    that are more suitable for quantum processing, including:
    - Dimension reduction
    - Feature scaling
    - Fourier features
    - Kernel methods
    """
    
    def __init__(self, n_qubits: int, scaling: str = "minmax"):
        """Initialize feature engineering.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the target quantum circuit.
        scaling : str
            Type of feature scaling to apply: "minmax", "standard", or "none".
        """
        self.n_qubits = n_qubits
        self.scaling = scaling
        
        # Dimensionality the quantum circuit can handle
        self.target_dim = n_qubits
        
        # Initialize transformers
        if scaling == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif scaling == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None
            
        self.pca = None
        self.kpca = None
    
    def fit_transform(self, X_train: np.ndarray, X_test: np.ndarray, 
                     method: str = "simple") -> Tuple[np.ndarray, np.ndarray]:
        """Fit transformations on training data and apply to both sets.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training data matrix of shape (n_samples, n_features).
        X_test : np.ndarray
            Test data matrix of shape (n_samples, n_features).
        method : str
            Transformation method to apply: "simple", "pca", "kpca", 
            "fourier", or "angle".
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed training and test data.
        """
        if method == "simple":
            # Just scale and match dimensions
            if self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
                
            # Match dimensions
            X_train_final = match_dimensions(X_train_scaled, self.target_dim)
            X_test_final = match_dimensions(X_test_scaled, self.target_dim)
        
        elif method == "pca":
            # Apply PCA to reduce dimensionality
            n_components = min(self.target_dim, X_train.shape[1])
            self.pca = PCA(n_components=n_components)
            
            # First scale the data if needed
            if self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
                
            # Then apply PCA
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_test_pca = self.pca.transform(X_test_scaled)
            
            # Match dimensions if needed
            X_train_final = match_dimensions(X_train_pca, self.target_dim)
            X_test_final = match_dimensions(X_test_pca, self.target_dim)
        
        elif method == "kpca":
            # Apply Kernel PCA for nonlinear dimensionality reduction
            n_components = min(self.target_dim, X_train.shape[1])
            self.kpca = KernelPCA(n_components=n_components, kernel="rbf")
            
            # Scale first
            if self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
            
            # Apply Kernel PCA
            X_train_kpca = self.kpca.fit_transform(X_train_scaled)
            X_test_kpca = self.kpca.transform(X_test_scaled)
            
            # Match dimensions if needed
            X_train_final = match_dimensions(X_train_kpca, self.target_dim)
            X_test_final = match_dimensions(X_test_kpca, self.target_dim)
        
        elif method == "fourier":
            # Apply Random Fourier Features
            # This approximates kernel methods and can improve performance
            # Scale first
            if self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
            
            # Generate random Fourier features
            X_train_final, X_test_final = self._fourier_features(
                X_train_scaled, X_test_scaled, n_components=self.target_dim
            )
        
        elif method == "angle":
            # Custom encoding optimized for angle embedding
            # Scale to [-π, π] range
            X_train_scaled = 2 * np.pi * (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8) - np.pi
            X_test_scaled = 2 * np.pi * (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8) - np.pi
            
            # Match dimensions
            X_train_final = match_dimensions(X_train_scaled, self.target_dim)
            X_test_final = match_dimensions(X_test_scaled, self.target_dim)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return X_train_final, X_test_final
    
    def _fourier_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                         n_components: int, gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Random Fourier Features for kernel approximation.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training data matrix.
        X_test : np.ndarray
            Test data matrix.
        n_components : int
            Number of Fourier components.
        gamma : float
            RBF kernel parameter.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Fourier features for train and test sets.
        """
        # Generate random weights for Fourier transform
        n_features = X_train.shape[1]
        self.rff_weights = np.random.normal(0, 1, (n_features, n_components // 2)) * np.sqrt(2 * gamma)
        self.rff_bias = np.random.uniform(0, 2 * np.pi, n_components // 2)
        
        # Generate features
        X_train_proj = np.dot(X_train, self.rff_weights) + self.rff_bias
        X_test_proj = np.dot(X_test, self.rff_weights) + self.rff_bias
        
        # Apply sine and cosine
        X_train_fourier = np.column_stack([
            np.cos(X_train_proj),
            np.sin(X_train_proj)
        ]) / np.sqrt(n_components // 2)
        
        X_test_fourier = np.column_stack([
            np.cos(X_test_proj),
            np.sin(X_test_proj)
        ]) / np.sqrt(n_components // 2)
        
        return X_train_fourier, X_test_fourier
    
    def plot_transformed_data(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           title: str = "Transformed Data") -> None:
        """Plot the first two dimensions of transformed data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Transformed training data.
        y_train : np.ndarray
            Training labels.
        X_test : np.ndarray
            Transformed test data.
        y_test : np.ndarray
            Test labels.
        title : str
            Plot title.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training data
        for c in np.unique(y_train):
            mask = y_train == c
            plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                       alpha=0.6, label=f"Train Class {c}")
        
        # Plot test data
        for c in np.unique(y_test):
            mask = y_test == c
            plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                       marker='x', alpha=0.8, label=f"Test Class {c}")
        
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png")
        plt.close()


def generate_complex_data(
    n_samples: int,
    n_features: int = 2,
    n_classes: int = 2,
    dataset_type: str = "nonlinear",
    noise: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate more complex synthetic datasets beyond moons and blobs.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features (dimensions).
    n_classes : int
        Number of classes.
    dataset_type : str
        Type of dataset: "nonlinear", "spirals", "xor", "concentric", or "gaussian".
    noise : float
        Noise level.
    seed : int
        Random seed.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix X and labels y.
    """
    np.random.seed(seed)
    
    if dataset_type == "nonlinear":
        # Nonlinear decision boundary dataset
        X = np.random.rand(n_samples, n_features) * 4 - 2  # Range [-2, 2]
        y = np.zeros(n_samples)
        
        # Decision boundary: x^2 + y^2 < r^2
        radius = 1.0
        for i in range(n_samples):
            if np.sum(X[i, :2] ** 2) < radius**2:
                y[i] = 0
            else:
                y[i] = 1
                
        # Add noise
        X += noise * np.random.randn(*X.shape)
    
    elif dataset_type == "spirals":
        # Intertwined spirals
        n_per_class = n_samples // n_classes
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        for i in range(n_classes):
            ix = range(n_per_class * i, n_per_class * (i + 1))
            r = np.linspace(0.0, 1, n_per_class)  # radius
            t = np.linspace(i * 4, (i + 1) * 4, n_per_class) + np.random.randn(n_per_class) * noise  # theta
            X[ix, 0] = r * np.sin(t * 2.5)
            X[ix, 1] = r * np.cos(t * 2.5)
            y[ix] = i
            
        # Fill remaining features with random noise
        if n_features > 2:
            X[:, 2:] = np.random.randn(n_samples, n_features - 2) * 0.1
    
    elif dataset_type == "xor":
        # XOR-like dataset
        X = np.random.rand(n_samples, n_features) * 2 - 1  # Range [-1, 1]
        y = np.zeros(n_samples)
        
        # XOR decision boundary
        for i in range(n_samples):
            if (X[i, 0] * X[i, 1]) > 0:
                y[i] = 1
            else:
                y[i] = 0
                
        # Add noise
        X += noise * np.random.randn(*X.shape)
    
    elif dataset_type == "concentric":
        # Concentric circles
        n_per_class = n_samples // n_classes
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        for i in range(n_classes):
            ix = range(n_per_class * i, n_per_class * (i + 1))
            r = (i + 1) * (1.0 / n_classes) * 2  # radius
            t = np.linspace(0, 2 * np.pi, n_per_class) + np.random.randn(n_per_class) * noise  # theta
            X[ix, 0] = r * np.sin(t)
            X[ix, 1] = r * np.cos(t)
            y[ix] = i
            
        # Fill remaining features with random noise
        if n_features > 2:
            X[:, 2:] = np.random.randn(n_samples, n_features - 2) * 0.1
    
    elif dataset_type == "gaussian":
        # Gaussian clusters
        n_per_class = n_samples // n_classes
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        for i in range(n_classes):
            ix = range(n_per_class * i, n_per_class * (i + 1))
            # Generate cluster center
            center = np.random.randn(n_features) * 2
            # Generate points around center
            X[ix] = center + np.random.randn(n_per_class, n_features) * noise
            y[ix] = i
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X.astype(np.float32), y.astype(np.int64)


def load_real_dataset(
    dataset_name: str,
    data_dir: str = "datasets",
    test_size: float = 0.2,
    binary: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load real-world datasets.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset: "iris", "wine", "breast_cancer", "digits".
    data_dir : str
        Directory to save/load datasets.
    test_size : float
        Fraction of data to use for testing.
    binary : bool
        Whether to convert to a binary classification problem.
    seed : int
        Random seed.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    try:
        from sklearn import datasets
        
        if dataset_name == "iris":
            data = datasets.load_iris()
        elif dataset_name == "wine":
            data = datasets.load_wine()
        elif dataset_name == "breast_cancer":
            data = datasets.load_breast_cancer()
        elif dataset_name == "digits":
            data = datasets.load_digits()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)
        
        # Convert to binary if requested
        if binary and len(np.unique(y)) > 2:
            # Take first two classes only
            mask = y < 2
            X = X[mask]
            y = y[mask]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    except ImportError:
        print("scikit-learn not available. Using synthetic data instead.")
        # Fall back to synthetic data
        X, y = generate_complex_data(n_samples=150, dataset_type="gaussian", seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        return X_train, X_test, y_train, y_test


def data_augmentation(
    X: np.ndarray, 
    y: np.ndarray, 
    augmentation_factor: int = 2,
    noise_level: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment data with noisy copies and synthetic examples.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    augmentation_factor : int
        Factor by which to increase dataset size.
    noise_level : float
        Standard deviation of Gaussian noise to add.
    seed : int
        Random seed.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Augmented features and labels.
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    
    # Create output arrays
    X_aug = np.zeros((n_samples * augmentation_factor, n_features))
    y_aug = np.zeros(n_samples * augmentation_factor)
    
    # Copy original data
    X_aug[:n_samples] = X
    y_aug[:n_samples] = y
    
    # Add noisy copies
    for i in range(1, augmentation_factor):
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples
        
        # Add Gaussian noise
        X_aug[start_idx:end_idx] = X + np.random.normal(0, noise_level, X.shape)
        y_aug[start_idx:end_idx] = y
    
    return X_aug.astype(np.float32), y_aug.astype(np.int64)