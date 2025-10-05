"""Training script for the Variational Quantum Classifier with fixed model."""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

from vqc.data import generate_data, normalize_features, match_dimensions
from vqc.model import VariationalQuantumClassifier


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a variational quantum classifier on synthetic data"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="blobs",
        choices=["moons", "blobs"],
        help="Type of dataset to generate",
    )
    parser.add_argument(
        "--n-qubits", type=int, default=2, help="Number of qubits (features)"
    )
    parser.add_argument("--layers", type=int, default=1, help="Number of circuit layers")
    parser.add_argument(
        "--embedding",
        type=str,
        default="angle",
        choices=["angle", "amplitude"],
        help="Data embedding scheme",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Measurement shots (None or 0 for analytic expectation)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--train-size", type=int, default=10, help="Training set size")
    parser.add_argument("--test-size", type=int, default=4, help="Test set size")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level for data")
    parser.add_argument(
        "--output-dir", type=str, default="artifacts", help="Output directory"
    )
    args = parser.parse_args()
    
    if args.shots == 0:
        args.shots = None
    
    return args


def train_model(args):
    print(f"Training VQC with {args.n_qubits} qubits, {args.layers} layers, {args.embedding} embedding...")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Generate data
    print(f"Generating {args.dataset} dataset with {args.train_size} training samples...")
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
    
    # Normalize and match dimensions
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    
    if X_train_norm.shape[1] != args.n_qubits:
        X_train_norm = match_dimensions(X_train_norm, args.n_qubits)
        X_test_norm = match_dimensions(X_test_norm, args.n_qubits)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create model
    model = VariationalQuantumClassifier(
        n_qubits=args.n_qubits,
        layers=args.layers,
        embedding=args.embedding,
        shots=args.shots,
    )
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEWithLogitsLoss()
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    
    batch_size = min(5, args.train_size)  # Keep batch size small
    
    for epoch in range(args.epochs):
        # Shuffle data
        idx = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[idx]
        y_train_shuffled = y_train_tensor[idx]
        
        # Training
        model.train()
        total_loss = 0
        
        # Process in small batches to avoid memory issues
        for i in range(0, len(X_train_tensor), batch_size):
            end = min(i + batch_size, len(X_train_tensor))
            X_batch = X_train_shuffled[i:end]
            y_batch = y_train_shuffled[i:end]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
        
        avg_loss = total_loss / len(X_train_tensor)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs >= 0.5).float()
            test_acc = (test_preds == y_test_tensor).float().mean().item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "vqc_model.pt"))
    print(f"Model saved to {os.path.join(args.output_dir, 'vqc_model.pt')}")
    
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
    
    print(f"Final Results - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


def main():
    args = parse_arguments()
    train_model(args)


if __name__ == "__main__":
    main()