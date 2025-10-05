"""Hyperparameter tuning for the Variational Quantum Classifier."""

import os
import itertools
import json
import numpy as np
import torch
from datetime import datetime
from vqc.data import generate_data, normalize_features, match_dimensions
from vqc.model import VariationalQuantumClassifier
from vqc.circuit import ring_entangler

# Custom entanglement patterns to test
def all_to_all_entangler(n_qubits):
    """Apply CNOT gates between all pairs of qubits."""
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qml.CNOT(wires=[i, j])

def line_entangler(n_qubits):
    """Apply CNOT gates between adjacent qubits in a line."""
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

def no_entanglement(n_qubits):
    """No entanglement between qubits."""
    pass

def run_experiment(params):
    """Run a single experiment with the given hyperparameters."""
    print(f"Running experiment with: {params}")
    
    # Set seeds
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    
    # Generate data
    X_train, y_train = generate_data(
        n=params["train_size"], 
        kind=params["dataset"], 
        noise=params["noise"], 
        seed=params["seed"]
    )
    X_test, y_test = generate_data(
        n=params["test_size"], 
        kind=params["dataset"], 
        noise=params["noise"], 
        seed=params["seed"] + 1
    )
    
    # Normalize and match dimensions
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    
    if X_train_norm.shape[1] != params["n_qubits"]:
        X_train_norm = match_dimensions(X_train_norm, params["n_qubits"])
        X_test_norm = match_dimensions(X_test_norm, params["n_qubits"])
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Select entanglement pattern
    if params["entanglement"] == "ring":
        entanglement_fn = ring_entangler
    elif params["entanglement"] == "all_to_all":
        entanglement_fn = all_to_all_entangler
    elif params["entanglement"] == "line":
        entanglement_fn = line_entangler
    else:
        entanglement_fn = no_entanglement
    
    # Create model
    model = VariationalQuantumClassifier(
        n_qubits=params["n_qubits"],
        layers=params["layers"],
        embedding=params["embedding"],
        shots=params["shots"],
        entanglement=entanglement_fn
    )
    
    # Early stopping parameters
    best_val_acc = 0
    best_epoch = 0
    patience = params.get("patience", 10)  # Number of epochs to wait for improvement
    
    # Create optimizer
    optimizer = getattr(torch.optim, params["optimizer"])(
        model.parameters(), lr=params["lr"]
    )
    
    # Training loop
    train_losses = []
    test_accs = []
    
    for epoch in range(params["epochs"]):
        # Shuffle data
        idx = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[idx]
        y_train_shuffled = y_train_tensor[idx]
        
        # Training
        model.train()
        total_loss = 0
        
        # Process in small batches
        batch_size = min(params.get("batch_size", 5), params["train_size"])
        for i in range(0, len(X_train_tensor), batch_size):
            end = min(i + batch_size, len(X_train_tensor))
            X_batch = X_train_shuffled[i:end]
            y_batch = y_train_shuffled[i:end]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
        
        avg_loss = total_loss / len(X_train_tensor)
        train_losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs >= 0.5).float()
            test_acc = (test_preds == y_test_tensor).float().mean().item()
            test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{params['epochs']}, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Early stopping check
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_tensor)
        train_probs = torch.sigmoid(train_logits)
        train_preds = (train_probs >= 0.5).float()
        train_acc = (train_preds == y_train_tensor).float().mean().item()
        
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).float()
        test_acc = (test_preds == y_test_tensor).float().mean().item()
    
    # Save results
    result = {
        "params": params,
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "test_accs": test_accs
    }
    
    return result


def grid_search():
    """Run grid search over hyperparameters."""
    # Define parameter grid (reduced for quick demonstration)
    param_grid = {
        "seed": [42],
        "n_qubits": [2, 3],
        "layers": [1, 2],
        "embedding": ["angle", "amplitude"],
        "entanglement": ["ring", "none"],
        "dataset": ["blobs"],
        "train_size": [20],
        "test_size": [10],
        "epochs": [10],
        "patience": [5],  # For early stopping
        "lr": [0.01, 0.05],
        "optimizer": ["Adam"],
        "shots": [None],  # None for analytic
        "batch_size": [5],
        "noise": [0.2]
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("experiments", f"grid_search_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameter grid for reference
    with open(os.path.join(output_dir, "param_grid.json"), "w") as f:
        json.dump(param_grid, f, indent=2)
    
    # Generate all parameter combinations
    # Start with smaller combinations to avoid combinatorial explosion
    param_keys = ["n_qubits", "layers", "embedding", "entanglement", "optimizer", "lr"]
    fixed_params = {k: v[0] for k, v in param_grid.items() if k not in param_keys}
    
    variable_params = []
    for k in param_keys:
        variable_params.append([(k, v) for v in param_grid[k]])
    
    # Generate combinations
    combinations = list(itertools.product(*variable_params))
    results = []
    
    for i, combination in enumerate(combinations):
        # Create parameter dictionary for this run
        params = fixed_params.copy()
        for key, value in combination:
            params[key] = value
        
        print(f"Running experiment {i+1}/{len(combinations)}")
        
        try:
            # Run experiment
            result = run_experiment(params)
            results.append(result)
            
            # Save individual result
            with open(os.path.join(output_dir, f"result_{i}.json"), "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = result.copy()
                serializable_result["train_losses"] = [float(x) for x in result["train_losses"]]
                serializable_result["test_accs"] = [float(x) for x in result["test_accs"]]
                json.dump(serializable_result, f, indent=2)
        
        except Exception as e:
            print(f"Error in experiment {i}: {str(e)}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x["final_test_acc"])
    print("\nBest configuration:")
    print(json.dumps(best_result["params"], indent=2))
    print(f"Test accuracy: {best_result['final_test_acc']:.4f}")
    
    # Save all results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        serialized_results = []
        for result in results:
            serialized_result = result.copy()
            serialized_result["train_losses"] = [float(x) for x in result["train_losses"]]
            serialized_result["test_accs"] = [float(x) for x in result["test_accs"]]
            serialized_results.append(serialized_result)
        json.dump(serialized_results, f, indent=2)
    
    return best_result


if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    best_result = grid_search()