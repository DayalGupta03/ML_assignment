# BT2024167 Polynomial Regression Assignment

This repository contains reproducible training, validation, and inference code for both personalized problems in the Polynomial Regression assignment:

- **VAR1 — Power Plant Steam Turbine Optimization:** `x1`–`x6` predict Net Power Score `y`.
- **VAR2 — Subterranean Thermal Reservoir Mapping:** `x1`–`x3` predict Thermal Anomaly Score `y`.

Both workflows use `PolynomialFeatures` followed by `LinearRegression`. The scripts use only their matching training and test files; test targets are not used. CSVs are read in their original row order, and predictions are saved with one `y` column and no index.

## Repository contents

- `solve_var1.py` — runs VAR1 dataset inspection, 80/20 validation, degree 1–10 comparison, five-fold check, final fit on all training rows, and submission generation.
- `solve_var2.py` — runs the equivalent VAR2 workflow for degrees 1–20 and records numerical stability diagnostics.
- `inference.py` — refits the selected model on all training rows and regenerates one chosen prediction CSV without rerunning validation.
- `BT2024167/` — personalized training and test CSV files for both variants.
- `BT2024167_pred_var1.csv` and `BT2024167_pred_var2.csv` — generated submissions.
- `BT2024167_VAR1_Report.pdf` and `BT2024167_VAR2_Report.pdf` — concise report sections.
- `sample_submission.csv` — assignment-provided output format example.

## Setup

Use Python 3.11 or newer, then install the versions used for the experiments:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Reproduce training and validation

Run from the repository root:

```bash
python solve_var1.py
python solve_var2.py
```

Each script prints dataset diagnostics, validation metrics, five-fold robustness results for the three strongest validation candidates, and submission verification. It then writes its prediction CSV at the repository root.

## Regenerate predictions only

The selected degrees from the training-only validation and cross-validation experiments are fixed in `inference.py` (VAR1: degree 4; VAR2: degree 8). This command refits that selected polynomial-regression model on all rows in the corresponding training CSV and predicts the matching test CSV:

```bash
python inference.py --variant var1
python inference.py --variant var2
```

Optionally provide a different data directory and output path:

```bash
python inference.py --variant var2 --data-dir BT2024167 --output BT2024167_pred_var2.csv
```

## Recorded model selection results

Metrics below are from the reproducible 80/20 training-only split (`random_state=42`). They are validation results, not hidden-test scores.

| Variant | Selected degree | Validation MSE | Validation R² | Five-fold mean CV MSE | Five-fold mean CV R² |
|---|---:|---:|---:|---:|---:|
| VAR1 | 4 | 0.85152215 | 0.92367249 | 0.78515636 | 0.91834494 |
| VAR2 | 8 | 0.244339735 | 0.995118389 | 0.258073734 | 0.994009207 |

VAR2 degrees 15–20 showed rank deficiency and sharply worse validation scores; degree 8 was selected based on validation and five-fold results. No hidden test labels or test evaluation scores are included.
