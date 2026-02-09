"""
Step 4 (evaluation): Evaluate the trained exoplanet MLP on the test set.

Context:
  After training, we need to report how the model performs on unseen data (the
  test set). We load the best checkpoint saved by train.py, run forward passes
  on X_test, and compare predictions to y_test. Metrics (accuracy, precision,
  recall, F1) and a confusion matrix are printed and the confusion matrix is
  saved as a plot. This allows direct comparison with baseline_models.py output
  (same test set, same class names).

What this script does:
  - Loads best_model.pt (state_dict + n_features, num_classes, class_names).
  - Rebuilds the MLP with get_model(n_features, num_classes) and loads state_dict.
  - Loads X_test, y_test from data/processed/; runs model in eval mode (no dropout).
  - Computes accuracy, prints classification_report, and saves confusion matrix to outputs/.

Important details:
  - map_location=DEVICE so the checkpoint loads on CPU if no GPU (e.g. on another machine).
  - model.eval() disables dropout so predictions are deterministic.
  - argmax(dim=1) on logits gives the predicted class index (0, 1, or 2).
  - class_names from the checkpoint match the label encoding from preprocess.py.

Run from project root: python src/evaluate.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from model import get_model

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "best_model.pt")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_data():
    """
    Load the saved checkpoint and test data. Rebuilds the MLP using the
    architecture parameters stored in the checkpoint (n_features, num_classes)
    and loads the trained weights (model_state_dict). Returns model, test
    tensors, y_test as numpy, and class names for reporting.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        print("ERROR: No saved model found. Run train.py first.")
        return None, None, None, None

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    n_features = checkpoint["n_features"]
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]

    model = get_model(input_size=n_features, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    return model, X_test_t, y_test, class_names


def main():
    result = load_model_and_data()
    if result[0] is None:
        return
    model, X_test_t, y_test, class_names = result

    # Predict: one forward pass; argmax gives predicted class index per sample.
    with torch.no_grad():
        logits = model(X_test_t.to(DEVICE))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    accuracy = (y_pred == y_test).mean()
    print("=" * 50)
    print("Exoplanet classification — evaluation")
    print("=" * 50)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

    # classification_report: precision, recall, f1-score per class + macro/weighted averages.
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm)
    print()

    # Same plot style as baseline_models.py for easy visual comparison.
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax, label="Count")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.title("Confusion matrix — Exoplanet disposition")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=100)
    plt.close()
    print(f"Confusion matrix plot saved to outputs/confusion_matrix.png")

    # Short learning note
    print("\n--- Reading the confusion matrix ---")
    print("Rows = true label, Columns = predicted label.")
    print("Diagonal = correct predictions; off-diagonal = confusions.")
    print("E.g. high value in (FALSE POSITIVE, CANDIDATE) means false positives often predicted as candidates.")


if __name__ == "__main__":
    main()
