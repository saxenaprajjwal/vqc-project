"""Hardware-efficient circuit ansätze and optimized embedding strategies.

This module provides circuit designs optimized for both simulators and real
quantum hardware, with reduced gate count and improved coherence properties.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from typing import List, Optional, Callable, Tuple

__all__ = [
    "hardware_efficient_ansatz",
    "ry_ansatz",
    "cnot_ring_ansatz",
    "batched_circuits",
    "adaptive_embedding"
]


def hardware_efficient_ansatz(params: np.ndarray, n_qubits: int) -> None:
    """Hardware-efficient ansatz using RY and CZ gates.
    
    This circuit design minimizes the number of two-qubit gates while
    maintaining expressivity. It uses a layer of single-qubit RY rotations
    followed by a ring of CZ gates.
    
    Parameters
    ----------
    params : np.ndarray
        Array of shape (n_qubits,) containing rotation angles.
    n_qubits : int
        Number of qubits in the circuit.
    """
    # Apply RY rotations to each qubit
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    
    # Apply CZ gates in a ring pattern
    for i in range(n_qubits):
        qml.CZ(wires=[i, (i + 1) % n_qubits])


def ry_ansatz(params: np.ndarray, n_qubits: int) -> None:
    """Simple RY rotation ansatz with minimal gates.
    
    This ansatz uses only RY rotations with no entangling gates.
    Useful as a baseline or for devices with poor two-qubit gate fidelity.
    
    Parameters
    ----------
    params : np.ndarray
        Array of shape (n_qubits,) containing rotation angles.
    n_qubits : int
        Number of qubits in the circuit.
    """
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)


def cnot_ring_ansatz(params: np.ndarray, n_qubits: int) -> None:
    """CNOT ring ansatz with RX, RZ rotations.
    
    This ansatz applies RX and RZ rotations followed by CNOT gates in a ring.
    It provides good expressivity while being relatively hardware-efficient.
    
    Parameters
    ----------
    params : np.ndarray
        Array of shape (n_qubits, 2) containing rotation angles.
    n_qubits : int
        Number of qubits in the circuit.
    """
    # Apply RX and RZ rotations
    for i in range(n_qubits):
        qml.RX(params[i, 0], wires=i)
        qml.RZ(params[i, 1], wires=i)
    
    # Apply CNOT gates in a ring
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


class BatchedCircuitExecutor:
    """Class for efficient batched execution of quantum circuits.
    
    This class implements techniques to reduce the overhead of circuit execution
    by batching multiple circuit evaluations together when possible.
    """
    
    def __init__(
        self, 
        circuit_fn: Callable, 
        n_qubits: int,
        batch_size: int = 8,
        device: str = "default.qubit"
    ):
        """Initialize the batched circuit executor.
        
        Parameters
        ----------
        circuit_fn : Callable
            Function that defines the quantum circuit.
        n_qubits : int
            Number of qubits in the circuit.
        batch_size : int
            Maximum number of circuits to batch together.
        device : str
            PennyLane device to use.
        """
        self.circuit_fn = circuit_fn
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.device = device
        
        # Initialize the quantum device
        self.qdev = qml.device(device, wires=n_qubits)
        
        # Define the batched circuit
        @qml.batch_params
        @qml.qnode(self.qdev, interface="torch")
        def batched_circuit(x, params):
            self.circuit_fn(x, params)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.batched_circuit = batched_circuit
    
    def execute(self, X: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Execute the circuit in batches.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (batch_size, n_features).
        params : torch.Tensor
            Circuit parameters.
            
        Returns
        -------
        torch.Tensor
            Circuit output expectation values.
        """
        batch_size = X.shape[0]
        results = []
        
        for i in range(0, batch_size, self.batch_size):
            end_idx = min(i + self.batch_size, batch_size)
            batch_X = X[i:end_idx]
            batch_results = self.batched_circuit(batch_X, params)
            results.append(batch_results)
        
        return torch.cat(results, dim=0)


def batched_circuits(X: torch.Tensor, circuit_fn: Callable, params: torch.Tensor, 
                    n_qubits: int, batch_size: int = 8) -> torch.Tensor:
    """Execute quantum circuits in batches for improved performance.
    
    This function uses PennyLane's batching capabilities to evaluate
    multiple circuits more efficiently.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data of shape (batch_size, n_features).
    circuit_fn : Callable
        Function that defines the quantum circuit.
    params : torch.Tensor
        Circuit parameters.
    n_qubits : int
        Number of qubits in the circuit.
    batch_size : int
        Maximum number of circuits to batch together.
        
    Returns
    -------
    torch.Tensor
        Circuit output expectation values.
    """
    executor = BatchedCircuitExecutor(
        circuit_fn=circuit_fn,
        n_qubits=n_qubits,
        batch_size=batch_size
    )
    return executor.execute(X, params)


def adaptive_embedding(x: np.ndarray, n_qubits: int, 
                      strategy: str = "auto") -> None:
    """Adaptive data embedding strategy.
    
    This function selects the most appropriate embedding strategy based
    on the input data characteristics and qubit count.
    
    Parameters
    ----------
    x : np.ndarray
        Input data vector.
    n_qubits : int
        Number of qubits in the circuit.
    strategy : str
        Embedding strategy: "auto" (automatic selection),
        "angle", "amplitude", or "iqp" (IQP encoding).
    """
    # Check data dimension vs. available qubits
    data_dim = len(x)
    
    if strategy == "auto":
        # Automatically select embedding strategy
        if data_dim <= n_qubits:
            # For low-dimensional data, angle encoding is efficient
            for i in range(min(data_dim, n_qubits)):
                qml.RX(x[i % data_dim] * np.pi, wires=i)
                qml.RZ(x[i % data_dim] * np.pi / 2, wires=i)
        else:
            # For higher-dimensional data, use amplitude encoding
            # Pad or truncate x to match required dimension
            x_padded = np.zeros(2**n_qubits)
            x_padded[:min(len(x), 2**n_qubits)] = x[:min(len(x), 2**n_qubits)]
            # Normalize
            norm = np.linalg.norm(x_padded)
            if norm > 0:
                x_padded /= norm
            qml.AmplitudeEmbedding(x_padded, wires=range(n_qubits), normalize=True)
    
    elif strategy == "angle":
        # Standard angle encoding
        for i in range(n_qubits):
            qml.RX(x[i % data_dim] * np.pi, wires=i)
            qml.RZ(x[i % data_dim] * np.pi / 2, wires=i)
    
    elif strategy == "amplitude":
        # Standard amplitude encoding
        x_padded = np.zeros(2**n_qubits)
        x_padded[:min(len(x), 2**n_qubits)] = x[:min(len(x), 2**n_qubits)]
        norm = np.linalg.norm(x_padded)
        if norm > 0:
            x_padded /= norm
        qml.AmplitudeEmbedding(x_padded, wires=range(n_qubits), normalize=True)
    
    elif strategy == "iqp":
        # IQP-inspired encoding (useful for certain ML tasks)
        # First layer of Hadamards
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Phase rotations based on data
        for i in range(n_qubits):
            qml.PhaseShift(x[i % data_dim] * np.pi, wires=i)
        
        # Entangling layer
        for i in range(n_qubits):
            qml.CZ(wires=[i, (i + 1) % n_qubits])
            
        # Second layer of Hadamards
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
    
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")