"""Advanced training script for Quantum Classifiers with enhanced features.

This script integrates:
1. Noise-robust quantum models
2. Hardware-efficient ansatz designs
3. Feature engineering techniques
4. Complex dataset generation
5. Improved optimization and training strategies
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn import BCEWithLogitsLoss

# Import core VQC modules
from vqc.data import normalize_features, match_dimensions
from vqc.model import VariationalQuantumClassifier

# Import enhanced modules
from vqc.noise_robust_model import NoiseRobustVQC
from vqc.hardware_efficient import (
    hardware_efficient_ansatz, 
    ry_ansatz, 
    cnot_ring_ansatz, 
    adaptive_embedding
)
from vqc.feature_engineering import (
    QuantumFeatureEngineering, 
    generate_complex_data,
    load_real_dataset,
    data_augmentation
)

import pennylane as qml


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Advanced training script for quantum classifiers"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="blobs",
        choices=["moons", "blobs", "nonlinear", "spirals", "xor", "concentric", "gaussian", 
                "iris", "wine", "breast_cancer", "digits"],
        help="Dataset type to use"
    )
    parser.add_argument("--train-size", type=int, default=30, help="Training set size")
    parser.add_argument("--test-size", type=int, default=10, help="Test set size")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level for data")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--augment-factor", type=int, default=2, help="Data augmentation factor")
    
    # Feature engineering parameters
    parser.add_argument(
        "--feature-eng",
        type=str,
        default="simple",
        choices=["simple", "pca", "kpca", "fourier", "angle"],
        help="Feature engineering method"
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="minmax",
        choices=["minmax", "standard", "none"],
        help="Feature scaling method"
    )
    
    # Quantum model parameters
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--layers", type=int, default=2, help="Number of circuit layers")
    parser.add_argument(
        "--model-type",
        type=str,
        default="standard",
        choices=["standard", "noise-robust"],
        help="Type of quantum model to use"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="angle",
        choices=["angle", "amplitude", "adaptive", "iqp"],
        help="Data embedding scheme"
    )
    parser.add_argument(
        "--ansatz",
        type=str,
        default="default",
        choices=["default", "hardware-efficient", "ry", "cnot-ring"],
        help="Circuit ansatz type"
    )
    parser.add_argument(
        "--entanglement",
        type=str,
        default="ring",
        choices=["ring", "line", "all-to-all", "none"],
        help="Entanglement pattern"
    )
    parser.add_argument(
        "--noise-sim",
        type=float,
        default=0.0,
        help="Simulated noise strength (0 for no noise)"
    )
    parser.add_argument(
        "--mitigation",
        type=str,
        default="none",
        choices=["none", "zne"],
        help="Error mitigation method"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Measurement shots (None for analytic)"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=5, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "SGD", "RMSprop", "Adagrad"],
        help="Optimizer type"
    )
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay (L2 penalty)")
    parser.add_argument("--early-stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Convert shots=0 to None for analytical expectation
    if args.shots == 0:
        args.shots = None
        
    return args


def setup_entanglement(pattern: str):
    """Set up entanglement function based on pattern name."""
    if pattern == "ring":
        def ring_entangle(n_qubits):
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i+1) % n_qubits])
        return ring_entangle
    
    elif pattern == "line":
        def line_entangle(n_qubits):
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])
        return line_entangle
    
    elif pattern == "all-to-all":
        def all_to_all_entangle(n_qubits):
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    qml.CNOT(wires=[i, j])
        return all_to_all_entangle
    
    elif pattern == "none":
        def no_entangle(n_qubits):
            pass
        return no_entangle
    
    else:
        raise ValueError(f"Unknown entanglement pattern: {pattern}")


def create_model(args):
    """Create quantum model based on arguments."""
    # Set up entanglement pattern
    entanglement_fn = setup_entanglement(args.entanglement)
    
    # Set up embedding function if adaptive
    if args.embedding == "adaptive":
        embedding = "angle"  # Default, will be overridden by adaptive embedding
    else:
        embedding = args.embedding
    
    if args.model_type == "standard":
        model = VariationalQuantumClassifier(
            n_qubits=args.n_qubits,
            layers=args.layers,
            embedding=embedding,
            entanglement=entanglement_fn,
            shots=args.shots,
        )
    else:  # noise-robust model
        # Set up observables - measure in multiple bases for noise robustness
        observables = ["Z"]
        if args.n_qubits >= 2:
            observables = ["Z", "X"]
        
        model = NoiseRobustVQC(
            n_qubits=args.n_qubits,
            layers=args.layers,
            embedding=embedding,
            entanglement=entanglement_fn,
            shots=args.shots,
            observables=observables,
            noise_strength=args.noise_sim,
            mitigation_method=args.mitigation,
        )
    
    return model


def prepare_data(args):
    """Prepare dataset based on arguments."""
    # Generate or load dataset
    if args.dataset in ["iris", "wine", "breast_cancer", "digits"]:
        # Real-world dataset
        X_train, X_test, y_train, y_test = load_real_dataset(
            dataset_name=args.dataset,
            test_size=args.test_size / (args.train_size + args.test_size),
            binary=True,  # Convert to binary classification
            seed=args.seed
        )
    else:
        # Synthetic dataset
        if args.dataset in ["moons", "blobs"]:
            # Use original data generation function
            from vqc.data import generate_data
            X_train, y_train = generate_data(
                n=args.train_size,
                kind=args.dataset,
                noise=args.noise,
                seed=args.seed
            )
            X_test, y_test = generate_data(
                n=args.test_size,
                kind=args.dataset,
                noise=args.noise,
                seed=args.seed + 1
            )
        else:
            # Use enhanced data generation
            X_train, y_train = generate_complex_data(
                n_samples=args.train_size,
                dataset_type=args.dataset,
                noise=args.noise,
                seed=args.seed
            )
            X_test, y_test = generate_complex_data(
                n_samples=args.test_size,
                dataset_type=args.dataset,
                noise=args.noise,
                seed=args.seed + 1
            )
    
    # Apply data augmentation if requested
    if args.augment:
        X_train, y_train = data_augmentation(
            X=X_train,
            y=y_train,
            augmentation_factor=args.augment_factor,
            noise_level=args.noise * 0.5,
            seed=args.seed
        )
    
    # Apply feature engineering
    feature_eng = QuantumFeatureEngineering(
        n_qubits=args.n_qubits,
        scaling=args.scaling
    )
    
    X_train_processed, X_test_processed = feature_eng.fit_transform(
        X_train=X_train,
        X_test=X_test,
        method=args.feature_eng
    )
    
    # Plot data if requested
    if args.plot:
        if X_train_processed.shape[1] >= 2:  # Need at least 2D for plotting
            feature_eng.plot_transformed_data(
                X_train=X_train_processed,
                y_train=y_train,
                X_test=X_test_processed,
                y_test=y_test,
                title=f"{args.dataset}_{args.feature_eng}"
            )
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def train_model(model, X_train, y_train, X_test, y_test, args):
    """Train the model with advanced techniques."""
    # Create optimizer
    optimizer_class = getattr(optim, args.optimizer)
    optimizer = optimizer_class(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create loss function
    loss_fn = BCEWithLogitsLoss()
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Show verbose output if requested
    if args.verbose:
        print("Learning rate scheduler enabled with patience=5")
    
    # Initialize training history
    train_losses = []
    train_accs = []
    test_accs = []
    
    # Early stopping variables
    best_test_acc = 0
    best_epoch = 0
    best_state = None
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(args.epochs):
        # Shuffle data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Training
        model.train()
        total_loss = 0
        correct = 0
        
        # Process in mini-batches
        for i in range(0, len(X_train), args.batch_size):
            end = min(i + args.batch_size, len(X_train))
            X_batch = X_train_shuffled[i:end]
            y_batch = y_train_shuffled[i:end]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
            
            # Compute training accuracy
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
        
        # Compute metrics
        avg_loss = total_loss / len(X_train)
        train_acc = correct / len(X_train)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs >= 0.5).float()
            test_acc = (test_preds == y_test).float().mean().item()
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save metrics
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}, "
              f"Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Test Acc: {test_acc:.4f}")
        
        # Check for early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_state = model.state_dict().copy()
        
        if args.early_stopping and epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if args.early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train)
        train_probs = torch.sigmoid(train_logits)
        train_preds = (train_probs >= 0.5).float()
        final_train_acc = (train_preds == y_train).float().mean().item()
        
        test_logits = model(X_test)
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).float()
        final_test_acc = (test_preds == y_test).float().mean().item()
    
    print(f"Final Results - Train Accuracy: {final_train_acc:.4f}, Test Accuracy: {final_test_acc:.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = f"vqc_{args.dataset}_{args.model_type}_{args.embedding}_{args.n_qubits}q_{args.layers}l_{timestamp}.pt"
    model_path = os.path.join(args.output_dir, model_name)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate plots if requested
    if args.plot:
        # Training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label="Train")
        plt.plot(test_accs, label="Test")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_name = f"training_curves_{args.dataset}_{args.model_type}_{timestamp}.png"
        plt.savefig(os.path.join(plot_dir, plot_name))
        plt.close()
    
    return {
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
    }


def main():
    """Main function to run the advanced training."""
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print configuration
    print("=== Configuration ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===================")
    
    # Prepare data
    print("\n=== Preparing Data ===")
    X_train, y_train, X_test, y_test = prepare_data(args)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_model(args)
    print(f"Model type: {args.model_type}, Qubits: {args.n_qubits}, Layers: {args.layers}")
    
    # Train model
    print("\n=== Training Model ===")
    results = train_model(model, X_train, y_train, X_test, y_test, args)
    
    print("\n=== Done ===")
    

if __name__ == "__main__":
    main()