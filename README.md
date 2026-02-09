# Exoplanets with AI

A beginner-friendly machine learning project that classifies exoplanet candidates from NASA Exoplanet Archive tabular data into **Confirmed**, **Candidate**, or **False Positive**.

## Goal

Train a simple PyTorch MLP to predict planet disposition using numerical physical parameters (orbital period, transit duration, planet radius, etc.) — no light curves or complex models.

## How it works

When you run `run_pipeline.py`, it runs **five steps in order**. Each step is a separate script; the console messages (“Step 1: …”, “Step 2: …”) are just labels for those scripts.

| Step | Script | What it does |
|------|--------|----------------|
| **1** | `load_and_eda.py` | Loads `data/keplerdata.csv`, prints basic stats, and saves EDA plots. |
| **2** | `preprocess.py` | Cleans data, selects features, encodes labels, splits train/test, normalizes. Saves everything under `data/processed/`. |
| **3** | `train.py` | Reads the processed arrays from `data/processed/`, trains the MLP, and saves the best model. |
| **4** | `evaluate.py` | Loads the saved model and test set, prints accuracy and classification report, and saves the confusion matrix plot. |
| **5** | `baseline_models.py` | Trains Logistic Regression and Random Forest on the same data; saves metrics (JSON) and confusion matrix plots. |

**Where do results go?**

- **`outputs/`** — Plots only: EDA (target distribution, feature distributions, box plots) and the final **confusion matrix**.
- **`data/processed/`** — Everything the model needs: train/test arrays (`.npy`), the **trained model** (`best_model.pt`), scaler, label encoder, and feature/class names. This is the “real” result of training; the plots in `outputs/` are for inspection.

So the result is **not** only in `outputs/`. The trained model and preprocessed data live in `data/processed/`.

**What happens if I run it again?**

Each run **overwrites** the previous one: new plots in `outputs/`, new arrays and a new `best_model.pt` in `data/processed/`. There is no checkpointing or “resume”; you always get a full, fresh pipeline run. That’s intentional for this MVP so the flow stays simple.

## Project structure

```
Exoplanet/
├── data/
│   ├── keplerdata.csv          # NASA Exoplanet Archive cumulative table (input)
│   └── processed/              # Created by pipeline: arrays, model, scaler, encoder
├── outputs/                    # Created by pipeline + baseline_models.py
│   ├── metrics/                # Baseline metrics (JSON), from baseline_models.py
│   ├── plots/                  # Baseline confusion matrices, from baseline_models.py
│   └── *.png                   # EDA + MLP confusion matrix
├── src/
│   ├── load_and_eda.py         # Step 1: Load data, exploratory analysis
│   ├── preprocess.py           # Step 2: Clean data, encode labels, normalize
│   ├── model.py                # MLP definition
│   ├── train.py                # Step 3: Training loop
│   ├── evaluate.py             # Step 4: Accuracy and confusion matrix
│   └── baseline_models.py      # Logistic Regression + Random Forest baselines
├── run_pipeline.py             # Runs all 5 steps in order
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment (recommended):**
   ```bash
   py -m venv venv
   venv\Scripts\activate   # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data:** Place `keplerdata.csv` in `data/` (download from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) — cumulative KOI table), or use your existing file.

## Usage

- **Run full pipeline** (EDA → preprocess → train → evaluate → baselines):
  ```bash
  py run_pipeline.py
  ```

- **Run steps separately** (for learning):
  ```bash
  python src/load_and_eda.py      # EDA only
  python src/preprocess.py        # Save cleaned train/test sets
  python src/train.py             # Train model (uses preprocessed data)
  python src/evaluate.py          # Evaluate saved model
  python src/baseline_models.py   # Logistic Regression + Random Forest baselines
  ```

## Tech stack

- **Python** — pandas, numpy, matplotlib, seaborn, scikit-learn, PyTorch
- **Model** — Simple MLP (no CNNs, no Transformers)
- **Scope** — Tabular data only; clarity over complexity

## Documentation (script-by-script)

Each script in `src/` has a module docstring at the top explaining **context**, **what it does**, and **important details**. Key ideas:

| Script | Purpose | Important concepts |
|--------|---------|--------------------|
| **load_and_eda.py** | Load NASA CSV and plot EDA | `comment='#'` in `read_csv` skips archive metadata lines; first non-comment line is the header. Plots go to `outputs/`. |
| **preprocess.py** | Clean data, encode labels, normalize, split | `stratify=y` in `train_test_split` keeps class balance. **Scaler is fit on train only**, then applied to test (avoids data leakage). Saves `.npy` arrays and `joblib` encoder/scaler to `data/processed/`. |
| **model.py** | PyTorch MLP definition | Input → Linear → ReLU → Dropout (×2) → Linear → logits. No softmax in forward; `CrossEntropyLoss` in train.py handles that. `get_model(input_size, num_classes)` builds the net. |
| **train.py** | Train the MLP and save best checkpoint | Loads data from `data/processed/`. `DataLoader` batches and shuffles. Each epoch: train on batches (loss, backward, step), then evaluate on test; save checkpoint when test accuracy improves. Checkpoint includes `state_dict` plus `n_features`, `num_classes`, `class_names` so evaluate.py can rebuild the model. |
| **evaluate.py** | Evaluate saved MLP on test set | Loads `best_model.pt`, rebuilds MLP with `get_model`, loads `state_dict`. Runs test set in `eval()` mode (no dropout). Prints accuracy and `classification_report`; saves confusion matrix to `outputs/confusion_matrix.png`. |
| **baseline_models.py** | Logistic Regression and Random Forest | Same train/test data as MLP. `class_weight='balanced'` for imbalanced classes. Saves JSON metrics to `outputs/metrics/` and confusion plots to `outputs/plots/`. Compare F1 macro/accuracy with the MLP. |

**Paths:** All scripts derive `PROJECT_ROOT` from `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (i.e. parent of `src/`), so paths work when run as `python src/script.py` from the project root.

**Random seeds:** `RANDOM_STATE=42` in preprocess, train, and baseline_models for reproducible splits and training.

## Learning focus

Code is commented to explain:

- How we handle NASA CSV comment headers and missing values
- Why we choose certain features and how we encode labels
- How a small MLP is defined and trained in PyTorch
- How to read a confusion matrix and accuracy
- Why baselines matter and how to compare them to the MLP

Enjoy exploring exoplanet classification.
