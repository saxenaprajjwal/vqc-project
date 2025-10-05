"""Variational Quantum Classifier model.

This module defines a hybrid quantum-classical model for classification tasks.
It uses a variational quantum circuit as a feature extractor followed by a
classical linear layer for final classification.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pennylane as qml
import torch
from torch import nn

from .circuit import amplitude_embedding, angle_embedding, layer


class VariationalQuantumClassifier(nn.Module):
    """Variational quantum classifier with trainable circuit parameters.

    This model combines a parameterized quantum circuit (PQC) with a classical
    linear layer. The PQC embeds data into a quantum state, applies a
    variational circuit with trainable parameters, and measures expectation
    values. These expectations are fed to a linear layer to produce
    classification logits.

    Parameters
    ----------
    n_qubits : int
        The number of qubits in the circuit. Should match data dimension.
    layers : int
        Number of trainable layers in the circuit.
    embedding : str or Callable
        Data embedding method. Built-in options are "angle" (default) or
        "amplitude". Alternatively, you can pass a custom callable that
        takes a feature tensor and embeds it into the circuit.
    entanglement : Callable[[int], None] or None
        Function that applies entangling gates between circuit layers.
        Takes the number of qubits as input. Defaults to no entanglement.
    shots : int or None
        Number of measurement shots. None or 0 for analytic expectation.
    readout_wires : Iterable[int] or None
        Indices of qubits to measure. If None, use the first two qubits
        (or just qubit 0 if there's only one qubit).
    device : str
        Name of the PennyLane device to use. Defaults to ``"default.qubit"``.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        embedding: str | Callable = "angle",
        entanglement: Callable[[int], None] | None = None,
        shots: int | None = None,
        readout_wires: Iterable[int] | None = None,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = None if shots in (0, None) else shots
        # Determine embedding function
        if isinstance(embedding, str):
            emb = embedding.lower()
            if emb == "angle":
                self.embedding_fn = angle_embedding
            elif emb == "amplitude":
                self.embedding_fn = amplitude_embedding
            else:
                raise ValueError(f"Unknown embedding type: {embedding}")
        elif callable(embedding):
            self.embedding_fn = embedding
        else:
            raise TypeError("embedding must be a string or callable")
        # Entanglement pattern
        self.entanglement_fn = entanglement or (lambda n: None)
        # Readout wires
        if readout_wires is None:
            # Use first wire if only one qubit, otherwise first two
            if n_qubits == 1:
                self.readout_wires = (0,)
            else:
                self.readout_wires = tuple(range(min(2, n_qubits)))
        else:
            self.readout_wires = tuple(readout_wires)
        # Quantum device
        self.dev = qml.device(device, wires=n_qubits, shots=self.shots)
        # Trainable circuit parameters initialised near zero
        init = 0.01 * torch.randn(layers, n_qubits, 3)
        self.params = nn.Parameter(init)
        # Linear head maps expectation values to logits
        self.head = nn.Linear(len(self.readout_wires), 1)

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the variational circuit to a single input sample.

        This private method is called within the QNode defined in ``forward``.
        It uses the embedding function, applies each trainable layer, and
        measures PauliZ on the readout wires.
        """
        # Embed data
        self.embedding_fn(x)
        # Apply layers
        for l in range(self.layers):
            layer(self.params[l], self.n_qubits)
            # Apply entanglement if provided
            self.entanglement_fn(self.n_qubits)
        # Measure expectations
        obs = [qml.expval(qml.PauliZ(w)) for w in self.readout_wires]
        return obs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for a batch of inputs.

        Each sample in the batch is passed through the quantum circuit via
        sequential evaluation. The resulting expectation values are stacked and
        passed through the classical linear head to produce logits.
        """
        # Define QNode inside forward to ensure differentiation works with current params
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def qnode(x: torch.Tensor) -> list:
            return self._circuit(x)

        outs: list[torch.Tensor] = []
        for i in range(X.shape[0]):
            expvals = qnode(X[i])
            # Convert the returned values to tensors
            if isinstance(expvals, list):
                # Handle case when we get a list of expectation values
                if len(expvals) == 1:
                    outs.append(torch.tensor([float(expvals[0])], device=X.device))
                else:
                    outs.append(torch.tensor([float(val) for val in expvals], device=X.device))
            elif not isinstance(expvals, torch.Tensor):
                # Convert scalar value to tensor
                outs.append(torch.tensor([float(expvals)], device=X.device))
            else:
                # Already a tensor
                outs.append(expvals)
        
        E = torch.stack(outs)
        logits = self.head(E)
        return logits.squeeze(-1)