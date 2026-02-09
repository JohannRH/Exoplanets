"""
Step 2: Clean data and prepare features for the ML model.

Context:
  Raw KOI data has many columns (period, duration, depth, stellar params, etc.)
  and often missing values. For the MVP we use a small set of physical numerical
  features and drop rows with any missing value in those or the target. Labels
  are text (CONFIRMED, CANDIDATE, FALSE POSITIVE); we encode them to 0, 1, 2
  so PyTorch and sklearn can use them. Features are normalized so different
  scales (e.g. period in days vs depth in ppm) don't dominate the model.

What this script does:
  - Filters to the three disposition classes; drops rows with missing target or
    missing values in any selected feature.
  - Encodes labels with sklearn LabelEncoder (consistent mapping for train/test).
  - Splits into train/test with stratify so class proportions are similar.
  - Fits StandardScaler on train only, then transforms train and test (avoids
    data leakage: test data must not influence scaling).
  - Saves NumPy arrays (.npy), feature/class names (.txt), and fitted encoder
    and scaler (joblib) under data/processed/.

Important details:
  - stratify=y_encoded in train_test_split keeps class balance in both splits.
  - Scaler is fit only on X_train; X_test is transformed with that same fit.
  - class_names order matches the integer labels (encoder.classes_[i] -> i).

Run from project root: python src/preprocess.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "keplerdata.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Target: the column we want to predict (exoplanet disposition).
TARGET_COL = "koi_disposition"

# Features: physical quantities from the transit fit. We use the main value only
# (no _err1/_err2) to keep the model simple. All must exist in the CSV.
FEATURE_COLUMNS = [
    "koi_period",      # Orbital period [days]
    "koi_duration",    # Transit duration [hrs]
    "koi_depth",       # Transit depth [ppm]
    "koi_ror",         # Planet-star radius ratio
    "koi_prad",        # Planetary radius [Earth radii]
    "koi_teq",         # Equilibrium temperature [K]
    "koi_impact",      # Impact parameter
    "koi_score",       # Disposition score (often used in vetting)
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw(path: str) -> pd.DataFrame:
    """Load CSV skipping NASA comment lines (same logic as load_and_eda.py)."""
    df = pd.read_csv(path, comment="#", low_memory=False)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame):
    """
    Clean and prepare data for training. Returns dict with scaled arrays,
    encoder, scaler, and names for features/classes.
    """
    # Keep only the three disposition classes we model.
    valid = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
    df = df[df[TARGET_COL].isin(valid)].copy()
    print(f"After keeping only CONFIRMED/CANDIDATE/FALSE POSITIVE: {len(df)} rows.")

    # Use only feature columns that exist (CSV schema can vary).
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_cols = set(FEATURE_COLUMNS) - set(available)
    if missing_cols:
        print(f"Note: These feature columns were not found and are skipped: {missing_cols}")
    if not available:
        raise ValueError("No feature columns found in the CSV.")

    # Drop any row with missing target or missing value in any selected feature.
    use_cols = [TARGET_COL] + available
    df_clean = df[use_cols].dropna()
    print(f"After dropping rows with missing values in target or features: {len(df_clean)} rows.")

    y = df_clean[TARGET_COL]
    X = df_clean[available]

    # LabelEncoder assigns an integer to each class (order = first appearance in y).
    # e.g. CANDIDATE->0, CONFIRMED->1, FALSE POSITIVE->2. Same mapping for train and test.
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = list(encoder.classes_)
    print(f"Label encoding: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")

    # stratify=y_encoded ensures train and test have similar class ratios.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # StandardScaler: zero mean, unit variance. fit on train only; transform test with same params.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": available,
        "class_names": class_names,
        "encoder": encoder,
        "scaler": scaler,
    }


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return

    print("Loading raw data...")
    df = load_raw(DATA_PATH)
    result = preprocess(df)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save arrays as .npy — train.py and baseline_models.py load these directly.
    np.save(os.path.join(OUT_DIR, "X_train.npy"), result["X_train"])
    np.save(os.path.join(OUT_DIR, "X_test.npy"), result["X_test"])
    np.save(os.path.join(OUT_DIR, "y_train.npy"), result["y_train"])
    np.save(os.path.join(OUT_DIR, "y_test.npy"), result["y_test"])

    # Save feature names and class names as text (for evaluation script)
    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(result["feature_names"]))
    with open(os.path.join(OUT_DIR, "class_names.txt"), "w") as f:
        f.write("\n".join(result["class_names"]))

    # Save fitted preprocessing objects (encoder, scaler) so we can reuse them
    joblib.dump(result["encoder"], os.path.join(OUT_DIR, "label_encoder.joblib"))
    joblib.dump(result["scaler"], os.path.join(OUT_DIR, "scaler.joblib"))

    print(f"\nPreprocessed data saved to {OUT_DIR}")
    print(f"Train samples: {len(result['y_train'])}, Test samples: {len(result['y_test'])}")
    print(f"Features: {result['feature_names']}")
    print("Next: run train.py to train the model.")


if __name__ == "__main__":
    main()
