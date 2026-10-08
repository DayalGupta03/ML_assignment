# BT2024167 Polynomial Regression Assignment

This repository contains the training, nested cross-validation, inference, figures, and report for both personalized polynomial-regression problems.

- **VAR1 — Power Plant Steam Turbine Optimization:** `x1`–`x6` predict Net Power Score `y`.
- **VAR2 — Subterranean Thermal Reservoir Mapping:** `x1`–`x3` predict Thermal Anomaly Score `y`.

Test files are used only as feature matrices for final prediction. They are never used to select models or hyperparameters.

## Setup

Use Python 3.11 or newer and install the tested package versions:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Full search and final fit

Run from the repository root:

```bash
python train_final.py
```

The script archives the previous submissions under `old_predictions/` and previous separate reports under `old_reports/`. It compares OLS, Ridge, Lasso, and ElasticNet using degree ranges 2–7 for VAR1 and 6–14 for VAR2. Regularized models are evaluated with and without `StandardScaler`. Five-fold outer CV (`random_state=7`) estimates generalization; three-fold inner CV selects alpha (and ElasticNet `l1_ratio`). A one-standard-error rule favors lower-degree and sparser configurations. The selected pipeline is refit on all training rows before test predictions are generated.

Outputs:

- `results/`: nested-CV tables, selected configurations, prediction comparison, and VAR2 high-degree stability diagnostics.
- `figures/`: 300-DPI CV, out-of-fold prediction, residual, sparsity, coefficient, and instability plots.
- `BT2024167_pred_var1.csv` and `BT2024167_pred_var2.csv`: one-column (`y`) predictions in test-row order.
- `BT2024167_Report.pdf`: combined report, no more than five pages.

## Recreate final predictions without rerunning the search

The selected configurations are hardcoded in `inference.py`:

```bash
python inference.py --variant var1
python inference.py --variant var2
```

Current selections: VAR1 degree-5 Lasso with `StandardScaler` (`alpha=0.006`); VAR2 degree-10 Ridge without scaling (`alpha=0.1`). Inference refits each exact pipeline on all training rows and predicts the matching test rows.

## Current nested-CV estimates

These are outer-fold cross-validation estimates, not hidden-test scores. The original plain-OLS baselines use degree 4 for VAR1 and degree 8 for VAR2, evaluated with the same nested-CV folds.

| Variant | Selected model | Degree | Alpha | Nested CV MSE (mean ± SD) | Nested CV R² mean | Original OLS baseline MSE / R² |
|---|---|---:|---:|---:|---:|---:|
| VAR1 | Scaled Lasso | 5 | 0.006 | 0.349092 ± 0.038405 | 0.963108 | 0.771742 / 0.918459 |
| VAR2 | Ridge | 10 | 0.1 | 0.232922 ± 0.013423 | 0.994256 | 0.255599 / 0.993656 |
