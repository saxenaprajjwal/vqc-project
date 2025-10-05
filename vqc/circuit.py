"""Circuit construction primitives for variational quantum classifiers.

This module defines common building blocks for quantum circuits, including
various data embedding schemes, entanglement patterns, and ansatz layers.
These functions can be composed to build larger quantum models or used
directly within QNodes.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

__all__ = [
    "angle_embedding",
    "amplitude_embedding",
    "ring_entangler",
    "layer",
]


def angle_embedding(x: np.ndarray | qml.numpy.ndarray) -> None:
    """Encode classical data as rotation angles on each qubit.

    Each element of ``x`` controls an ``RX`` and ``RZ`` rotation. The input
    should have a length equal to the number of wires in the device.
    """
    for i, xi in enumerate(x):
        qml.RX(xi * np.pi, wires=i)
        qml.RZ(xi * np.pi / 2, wires=i)


def amplitude_embedding(x: np.ndarray | qml.numpy.ndarray) -> None:
    """Embed classical data into the amplitude of a quantum state.

    The input vector ``x`` must be normalised; it will be padded or trimmed to
    match the dimension of the Hilbert space (``2**n_qubits``). This embedding
    prepares a quantum state where the amplitudes correspond to the feature
    values.
    """
    # Flatten and normalise input
    x = np.array(x, dtype=float).flatten()
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError("Amplitude embedding: input vector has zero norm")
    x_norm = x / norm
    qml.AmplitudeEmbedding(x_norm, wires=range(int(np.log2(len(x_norm)))), normalize=True)


def ring_entangler(n_qubits: int) -> None:
    """Apply a ring of CNOT gates to entangle qubits.

    This pattern applies ``CNOT`` from qubit ``i`` to ``i+1``, wrapping around
    the last qubit to the first.
    """
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


def layer(weights: np.ndarray, n_qubits: int) -> None:
    """One layer of trainable single-qubit rotations followed by entanglement.

    Parameters
    ----------
    weights : array_like
        A two-dimensional array of shape ``(n_qubits, 3)`` where each row
        contains rotation parameters ``(rx, ry, rz)`` for a corresponding qubit.
    n_qubits : int
        Number of qubits.
    """
    for i in range(n_qubits):
        rx, ry, rz = weights[i]
        qml.RX(rx, wires=i)
        qml.RY(ry, wires=i)
        qml.RZ(rz, wires=i)
    ring_entangler(n_qubits)
