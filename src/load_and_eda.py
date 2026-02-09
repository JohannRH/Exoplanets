"""
Step 1: Load and explore the NASA Exoplanet Archive data.

Context:
  The NASA Exoplanet Archive provides a cumulative table of Kepler Objects of
  Interest (KOIs). Each row is a candidate with a disposition: CONFIRMED (planet),
  CANDIDATE (not yet confirmed), or FALSE POSITIVE (e.g. stellar eclipse, instrument).
  This script loads that CSV and produces exploratory plots so we can inspect
  the target distribution and feature behaviour before modelling.

What this script does:
  - Loads the cumulative KOI CSV, skipping the archive's comment header lines.
  - Prints dataset shape, target (koi_disposition) distribution, and basic info.
  - Saves EDA plots to outputs/: target distribution, feature histograms,
    and box plots of key features by disposition.

Important details:
  - The CSV has many lines starting with '#' (column descriptions). We use
    comment='#' in read_csv so the first non-comment line is the header;
    the number of comment lines can vary between archive exports.
  - low_memory=False lets pandas infer column types across the whole file.

Run from project root: python src/load_and_eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Configuration — paths are relative to the project root (parent of src/).
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "keplerdata.csv")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
def load_exoplanet_csv(path: str) -> pd.DataFrame:
    """
    Load the NASA Exoplanet Archive cumulative table from CSV.

    - comment='#' : pandas skips every line that starts with '#' (metadata).
      The first line that does NOT start with '#' is treated as the header.
    - dropna(how="all") : removes rows that are entirely NaN (e.g. blank line
      after comments). reset_index(drop=True) keeps row indices 0, 1, 2, ...
    """
    df = pd.read_csv(path, comment="#", low_memory=False)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def main():
    print("Loading exoplanet data...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("Please download the cumulative table from NASA Exoplanet Archive and place it there.")
        return

    df = load_exoplanet_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.\n")

    # -------------------------------------------------------------------------
    # Inspect target column (disposition)
    # -------------------------------------------------------------------------
    target_col = "koi_disposition"
    if target_col not in df.columns:
        print(f"ERROR: Target column '{target_col}' not found. Available: {list(df.columns[:10])}...")
        return

    print("Target distribution (koi_disposition):")
    print(df[target_col].value_counts(dropna=False))
    print()

    # -------------------------------------------------------------------------
    # Basic info
    # -------------------------------------------------------------------------
    print("Dataset info:")
    print(df.info())
    print("\nFirst few rows of key columns:")
    key_cols = [target_col, "koi_period", "koi_duration", "koi_prad", "koi_teq", "koi_score"]
    key_cols = [c for c in key_cols if c in df.columns]
    print(df[key_cols].head(10))

    # -------------------------------------------------------------------------
    # EDA plots (save to project root or a small outputs folder)
    # -------------------------------------------------------------------------
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Target distribution — how many CONFIRMED / CANDIDATE / FALSE POSITIVE.
    #    value_counts() gives counts per label; order depends on frequency.
    fig, ax = plt.subplots(figsize=(8, 4))
    df[target_col].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.set_title("Exoplanet disposition (target labels)")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_target_distribution.png"), dpi=100)
    plt.close()
    print(f"Saved EDA plot: outputs/eda_target_distribution.png")

    # 2) Histograms of numeric features we will use in preprocess.py.
    #    dropna() here only for plotting; preprocess.py will drop rows with any missing feature.
    feature_cols = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq", "koi_score"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    if feature_cols:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(feature_cols):
            ax = axes[i]
            df[col].dropna().hist(ax=ax, bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(col)
        plt.suptitle("Distributions of selected numeric features (non-missing only)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eda_feature_distributions.png"), dpi=100)
        plt.close()
        print(f"Saved EDA plot: outputs/eda_feature_distributions.png")

    # 3) Box plots: feature value vs disposition. If boxes overlap a lot,
    #    that feature alone may not separate classes well; the model uses all features together.
    plot_features = [c for c in ["koi_period", "koi_duration", "koi_prad"] if c in df.columns]
    if plot_features and target_col in df.columns:
        fig, axes = plt.subplots(1, len(plot_features), figsize=(4 * len(plot_features), 4))
        if len(plot_features) == 1:
            axes = [axes]
        for ax, col in zip(axes, plot_features):
            sub = df[[target_col, col]].dropna()
            sns.boxplot(data=sub, x=target_col, y=col, ax=ax)
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=15)
        plt.suptitle("Feature by disposition (EDA)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eda_features_by_disposition.png"), dpi=100)
        plt.close()
        print(f"Saved EDA plot: outputs/eda_features_by_disposition.png")

    print("\nEDA done. Next step: run preprocess.py to clean data and prepare train/test sets.")


if __name__ == "__main__":
    main()
