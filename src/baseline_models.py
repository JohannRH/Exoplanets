"""
Step 5 (baselines): Train and evaluate classical ML models on the same data as the MLP.

Context:
  Before claiming that the neural network is "good", we need a reference. Logistic
  Regression and Random Forest are standard baselines for tabular data. They are
  trained on the same X_train, y_train and evaluated on the same X_test, y_test
  from data/processed/, so metrics are directly comparable to the MLP. Class
  imbalance (e.g. fewer FALSE POSITIVES) is handled with class_weight='balanced',
  which up-weights minority classes in the loss.

What this script does:
  - Loads processed data (same as train.py / evaluate.py).
  - Fits Logistic Regression and Random Forest with class_weight='balanced'.
  - Computes accuracy, precision/recall/F1 (macro and weighted), confusion matrix.
  - Saves metrics as JSON (outputs/metrics/*.json) and confusion matrix plots
    (outputs/plots/*.png). Prints a summary table for quick comparison with the MLP.

Important details:
  - RANDOM_STATE=42 matches train.py and preprocess.py for reproducibility.
  - Macro averaging: one metric per class then average (treats classes equally).
  - Weighted averaging: average weighted by support (class size). Use macro when
    classes are imbalanced and you care about minority performance.
  - Per-class metrics in the JSON match the order in class_names.txt.

Run from project root: python src/baseline_models.py
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
RANDOM_STATE = 42  # Same as train.py for reproducibility


def load_processed_data():
    """
    Load the same train/test arrays and metadata used by the MLP.
    Ensures baselines are evaluated on identical data.
    """
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    with open(os.path.join(PROCESSED_DIR, "feature_names.txt")) as f:
        feature_names = [line.strip() for line in f if line.strip()]
    with open(os.path.join(PROCESSED_DIR, "class_names.txt")) as f:
        class_names = [line.strip() for line in f if line.strip()]
    return X_train, X_test, y_train, y_test, feature_names, class_names


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute metrics comparable to evaluate.py: accuracy, precision/recall/F1
    (macro and weighted), per-class metrics, and confusion matrix. Returns a
    dict (for JSON) and the confusion matrix array (for plotting).
    zero_division=0 avoids warnings when a class has no predictions.
    """
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }
    # Per-class metrics (same order as class_names)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["per_class"] = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1-score": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    metrics["confusion_matrix"] = cm.tolist()
    return metrics, cm


def save_confusion_matrix_plot(cm, class_names, model_name, save_path):
    """
    Save a confusion matrix plot in the same style as evaluate.py
    so baseline and MLP plots are visually comparable.
    """
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
    ax.set_title(f"Confusion matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def run_baselines():
    """Load data, train Logistic Regression and Random Forest, evaluate and save results."""
    if not os.path.exists(PROCESSED_DIR):
        print("ERROR: data/processed/ not found. Run preprocess.py first.")
        return

    print("Loading processed data (same as MLP)...")
    X_train, X_test, y_train, y_test, feature_names, class_names = load_processed_data()
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Classes: {class_names}\n")

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Model 1: Logistic Regression
    # class_weight='balanced' adjusts weights inversely to class frequency
    # so minority classes (e.g. FALSE POSITIVE) are not ignored.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  Logistic Regression")
    print("=" * 60)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics_lr, cm_lr = compute_metrics(y_test, y_pred_lr, class_names)
    print(classification_report(y_test, y_pred_lr, target_names=class_names, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_lr)
    with open(os.path.join(METRICS_DIR, "logistic_regression_metrics.json"), "w") as f:
        json.dump(metrics_lr, f, indent=2)
    save_confusion_matrix_plot(
        cm_lr, class_names, "Logistic Regression",
        os.path.join(PLOTS_DIR, "confusion_matrix_logistic_regression.png"),
    )
    print(f"Metrics saved to outputs/metrics/logistic_regression_metrics.json")
    print(f"Plot saved to outputs/plots/confusion_matrix_logistic_regression.png\n")

    # -------------------------------------------------------------------------
    # Model 2: Random Forest
    # class_weight='balanced' again for fair comparison and imbalance handling.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  Random Forest")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_rf, cm_rf = compute_metrics(y_test, y_pred_rf, class_names)
    print(classification_report(y_test, y_pred_rf, target_names=class_names, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_rf)
    with open(os.path.join(METRICS_DIR, "random_forest_metrics.json"), "w") as f:
        json.dump(metrics_rf, f, indent=2)
    save_confusion_matrix_plot(
        cm_rf, class_names, "Random Forest",
        os.path.join(PLOTS_DIR, "confusion_matrix_random_forest.png"),
    )
    print(f"Metrics saved to outputs/metrics/random_forest_metrics.json")
    print(f"Plot saved to outputs/plots/confusion_matrix_random_forest.png\n")

    # -------------------------------------------------------------------------
    # Summary for quick comparison
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  Summary (test set)")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 (macro)':>12} {'F1 (weighted)':>14}")
    print("-" * 65)
    print(f"{'Logistic Regression':<25} {metrics_lr['accuracy']:>10.4f} {metrics_lr['f1_macro']:>12.4f} {metrics_lr['f1_weighted']:>14.4f}")
    print(f"{'Random Forest':<25} {metrics_rf['accuracy']:>10.4f} {metrics_rf['f1_macro']:>12.4f} {metrics_rf['f1_weighted']:>14.4f}")
    print("\nCompare these to the MLP accuracy and classification report from evaluate.py.")


# -----------------------------------------------------------------------------
# Why baselines matter:
#   - They set a performance floor: if the MLP does not beat Logistic Regression
#     or Random Forest on the same data, the added complexity may not be justified.
#   - Logistic Regression is interpretable (coefficients per feature) and fast.
#   - Random Forest shows what tree-based ensembles achieve without deep learning.
#   Compare metrics in outputs/metrics/*.json with the MLP's evaluate.py output
#   (accuracy and classification report). Use F1 macro when classes are imbalanced.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_baselines()
