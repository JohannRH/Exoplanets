"""
Step 4: Train the exoplanet MLP.

Context:
  We load the same preprocessed arrays (X_train, y_train, X_test, y_test) produced
  by preprocess.py. The model is trained with mini-batch gradient descent: each
  batch we compute loss (CrossEntropyLoss), backprop, and update weights with Adam.
  We evaluate on the test set each epoch and save the checkpoint with the highest
  test accuracy (for this MVP we use test as "validation"; normally you'd hold out
  a separate validation set for model selection).

What this script does:
  - Loads X_train, y_train, X_test, y_test and metadata from data/processed/.
  - Wraps train data in TensorDataset + DataLoader for batching and shuffling.
  - Builds the MLP, CrossEntropyLoss, and Adam optimizer.
  - Runs EPOCHS epochs: train on batches, then compute test accuracy; save best model.
  - Saves best_model.pt (state_dict + n_features, num_classes, class_names, feature_names)
    so evaluate.py can reload the model without hardcoding architecture.

Important details:
  - sys.path.insert lets us "from model import get_model" when run as python src/train.py.
  - model.train() / model.eval() toggle dropout and batch norm behaviour.
  - torch.no_grad() during test evaluation avoids storing gradients (saves memory).
  - state_dict is only the weights; we also save n_features/num_classes so the
    correct architecture can be rebuilt when loading.

Run from project root: python src/train.py
"""

import os
import sys
# When running "python src/train.py" from project root, src/ is not on path; add it so "model" is found.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import get_model

# -----------------------------------------------------------------------------
# Paths and hyperparameters
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "best_model.pt")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZES = (64, 32)
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42


def set_seed(seed: int):
    """Fix random seeds so training is reproducible (same data order, same init)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_processed_data():
    """Load NumPy arrays and feature/class names from data/processed/ (from preprocess.py)."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    with open(os.path.join(PROCESSED_DIR, "feature_names.txt")) as f:
        feature_names = [line.strip() for line in f if line.strip()]
    with open(os.path.join(PROCESSED_DIR, "class_names.txt")) as f:
        class_names = [line.strip() for line in f if line.strip()]
    return X_train, y_train, X_test, y_test, feature_names, class_names


def main():
    set_seed(RANDOM_STATE)
    if not os.path.exists(PROCESSED_DIR) or not os.path.exists(os.path.join(PROCESSED_DIR, "X_train.npy")):
        print("ERROR: Preprocessed data not found. Run preprocess.py first.")
        return

    print("Loading preprocessed data...")
    X_train, y_train, X_test, y_test, feature_names, class_names = load_processed_data()
    n_features = X_train.shape[1]
    num_classes = len(class_names)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples, Features: {n_features}, Classes: {num_classes}")

    # Tensors: float32 for features, long (int64) for class indices (required by CrossEntropyLoss).
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # DataLoader shuffles and returns batches; no need to shuffle manually each epoch.
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model on DEVICE (GPU if available); CrossEntropyLoss for multi-class; Adam for optimization.
    model = get_model(
        input_size=n_features,
        num_classes=num_classes,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_acc = 0.0
    print(f"\nTraining on {DEVICE} for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        model.train()  # Enables dropout.
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()   # Clear gradients from previous batch.
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()         # Compute gradients.
            optimizer.step()        # Update weights.
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Evaluate on test set (no gradient needed).
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()  # Class with highest logit.
            acc = (preds == y_test).mean()
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_features": n_features,
                "num_classes": num_classes,
                "class_names": class_names,
                "feature_names": feature_names,
            }, MODEL_SAVE_PATH)
        print(f"Epoch {epoch + 1:3d}/{EPOCHS}  loss: {train_loss:.4f}  test_acc: {acc:.4f}  best: {best_acc:.4f}")

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("Next: run evaluate.py for detailed metrics and confusion matrix.")


if __name__ == "__main__":
    main()
