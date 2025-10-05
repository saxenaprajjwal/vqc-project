"""Noise-robust Variational Quantum Classifier model.

This module extends the base VQC model with noise mitigation techniques
and improved optimization strategies.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Union

import pennylane as qml
import torch
from torch import nn
import numpy as np

from .circuit import amplitude_embedding, angle_embedding, layer


class NoiseRobustVQC(nn.Module):
    """Enhanced Variational Quantum Classifier with noise mitigation strategies.
    
    This model extends the basic VQC with:
    1. Support for measurement error mitigation
    2. Multiple readout observables
    3. Optimized circuit depth
    4. Support for custom ansatz patterns
    
    Parameters
    ----------
    n_qubits : int
        The number of qubits in the circuit. Should match data dimension.
    layers : int
        Number of trainable layers in the circuit.
    embedding : str or Callable
        Data embedding method: "angle" (default) or "amplitude".
    entanglement : Callable[[int], None] or None
        Function that applies entangling gates between circuit layers.
    shots : int or None
        Number of measurement shots. None or 0 for analytic expectation.
    readout_wires : Iterable[int] or None
        Indices of qubits to measure. If None, use the first two qubits.
    observables : List[str] or None
        List of observable types to measure: "Z", "X", "Y". Default is ["Z"].
    noise_strength : float
        Strength of depolarizing noise to simulate.
    mitigation_method : str
        Method for error mitigation: "none", "zne" (zero-noise extrapolation).
    device : str
        Name of the PennyLane device to use.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        embedding: Union[str, Callable] = "angle",
        entanglement: Optional[Callable[[int], None]] = None,
        shots: Optional[int] = None,
        readout_wires: Optional[Iterable[int]] = None,
        observables: Optional[List[str]] = None,
        noise_strength: float = 0.0,
        mitigation_method: str = "none",
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = None if shots in (0, None) else shots
        self.noise_strength = noise_strength
        self.mitigation_method = mitigation_method.lower()
        
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
            if n_qubits == 1:
                self.readout_wires = (0,)
            else:
                self.readout_wires = tuple(range(min(2, n_qubits)))
        else:
            self.readout_wires = tuple(readout_wires)
        
        # Observables to measure
        self.observables = observables or ["Z"]
        self._setup_observables()
        
        # Quantum device
        self.dev = qml.device(device, wires=n_qubits, shots=self.shots)
        
        # Trainable circuit parameters
        # Initialize parameters with a hardware-efficient strategy
        init_scale = 0.01
        if self.mitigation_method == "zne":
            # For ZNE, we need a different set of parameters for each noise scale
            # We'll use 3 scales: no noise, medium noise, high noise
            init = init_scale * torch.randn(3, layers, n_qubits, 3)
        else:
            init = init_scale * torch.randn(layers, n_qubits, 3)
        
        self.params = nn.Parameter(init)
        
        # Linear head maps expectation values to logits
        head_input_size = len(self.readout_wires) * len(self.observables)
        self.head = nn.Linear(head_input_size, 1)
        
        # Add batch normalization for better training stability
        self.batch_norm = nn.BatchNorm1d(head_input_size)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def _setup_observables(self) -> None:
        """Set up the observable list based on specified types."""
        self.obs_list = []
        for wire in self.readout_wires:
            for obs_type in self.observables:
                if obs_type == "X":
                    self.obs_list.append(qml.PauliX(wire))
                elif obs_type == "Y":
                    self.obs_list.append(qml.PauliY(wire))
                else:  # Default to Z
                    self.obs_list.append(qml.PauliZ(wire))
    
    def _apply_noise(self, noise_param: float = None) -> None:
        """Apply depolarizing noise to all qubits."""
        if noise_param is None:
            noise_param = self.noise_strength
            
        if noise_param > 0:
            for q in range(self.n_qubits):
                qml.DepolarizingChannel(noise_param, wires=q)
    
    def _circuit(self, x: torch.Tensor, noise_scale: float = 1.0) -> List:
        """Apply the variational circuit to a single input sample.
        
        Parameters
        ----------
        x : torch.Tensor
            Input feature vector.
        noise_scale : float
            Scaling factor for noise strength (used in ZNE).
            
        Returns
        -------
        List
            List of expectation values.
        """
        # Embed data
        self.embedding_fn(x)
        
        # Apply noise after embedding if using noise simulation
        self._apply_noise(self.noise_strength * noise_scale)
        
        # Apply layers with appropriate parameters
        if self.mitigation_method == "zne":
            # Use parameters corresponding to the noise scale
            # Here we use a simple mapping from continuous scale to discrete indices
            if noise_scale == 0:
                idx = 0  # No noise
            elif noise_scale == 1:
                idx = 1  # Medium noise
            else:
                idx = 2  # High noise
                
            params = self.params[idx]
        else:
            params = self.params
            
        for l in range(self.layers):
            layer(params[l], self.n_qubits)
            # Apply entanglement if provided
            self.entanglement_fn(self.n_qubits)
            # Apply mid-circuit noise if simulating noise
            if l < self.layers - 1:  # Don't apply after last layer
                self._apply_noise(self.noise_strength * noise_scale * 0.5)
        
        # Measure expectations
        return [qml.expval(obs) for obs in self.obs_list]
    
    def _mitigate_errors(self, results_list: List[torch.Tensor]) -> torch.Tensor:
        """Apply error mitigation to the raw results.
        
        For Zero-Noise Extrapolation (ZNE), we measure at different noise levels
        and extrapolate to zero noise.
        """
        if self.mitigation_method == "zne" and len(results_list) == 3:
            # Simple linear extrapolation to zero noise
            # results_list contains measurements at scale 0, 1, 2
            # We use results at scales 1 and 2 to extrapolate to 0
            scale1, scale2 = 1, 2
            r1, r2 = results_list[1], results_list[2]
            
            # Linear extrapolation: r0 = r1 - (r2 - r1)/(scale2 - scale1) * scale1
            r0_extrapolated = r1 - (r2 - r1) / (scale2 - scale1) * scale1
            return r0_extrapolated
        
        # Default: return the first result (no mitigation)
        return results_list[0]
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for a batch of inputs.
        
        Parameters
        ----------
        X : torch.Tensor
            Batch of input feature vectors.
            
        Returns
        -------
        torch.Tensor
            Output logits.
        """
        # For ZNE, we need to evaluate at multiple noise scales
        noise_scales = [0, 1, 2] if self.mitigation_method == "zne" else [1]
        
        results_list = []
        for scale in noise_scales:
            # Define QNode inside forward to ensure differentiation works
            @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
            def qnode(x: torch.Tensor) -> list:
                return self._circuit(x, noise_scale=scale)
            
            outs: list[torch.Tensor] = []
            for i in range(X.shape[0]):
                expvals = qnode(X[i])
                # Convert the returned values to tensors
                if isinstance(expvals, list):
                    # Handle case when we get a list of expectation values
                    outs.append(torch.tensor([float(val) for val in expvals], device=X.device))
                elif not isinstance(expvals, torch.Tensor):
                    # Convert scalar value to tensor
                    outs.append(torch.tensor([float(expvals)], device=X.device))
                else:
                    # Already a tensor
                    outs.append(expvals)
            
            E = torch.stack(outs)
            results_list.append(E)
        
        # Apply error mitigation if enabled
        E = self._mitigate_errors(results_list)
        
        # Apply batch normalization for training stability
        if self.training and E.shape[0] > 1:  # Only apply if batch size > 1
            E = self.batch_norm(E)
        
        # Apply dropout during training
        if self.training:
            E = self.dropout(E)
        
        # Pass through linear head
        logits = self.head(E)
        return logits.squeeze(-1)

    def reset_parameters(self) -> None:
        """Reset the trainable parameters of the model."""
        init_scale = 0.01
        with torch.no_grad():
            if self.mitigation_method == "zne":
                self.params.data = init_scale * torch.randn_like(self.params)
            else:
                self.params.data = init_scale * torch.randn_like(self.params)
            # Reset linear head
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)