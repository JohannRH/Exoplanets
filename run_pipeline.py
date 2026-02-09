"""
Run the full exoplanet classification pipeline.

Order: load/EDA -> preprocess -> train -> evaluate -> baseline models.

Run from project root: python run_pipeline.py
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


def run(script_name: str, description: str) -> bool:
    """Run a Python script from src/ and return True if it succeeded."""
    path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        return False
    print("\n" + "=" * 60)
    print(f"  {description}")
    print("=" * 60)
    result = subprocess.run([sys.executable, path], cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    print("Exoplanet classification pipeline")
    if not run("load_and_eda.py", "Step 1: Load data and EDA"):
        return
    if not run("preprocess.py", "Step 2: Preprocess data"):
        return
    if not run("train.py", "Step 3: Train MLP"):
        return
    if not run("evaluate.py", "Step 4: Evaluate model"):
        return
    if not run("baseline_models.py", "Step 5: Baseline models (Logistic Regression + Random Forest)"):
        return
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
