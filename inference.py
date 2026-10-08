"""Refit a selected VAR1/VAR2 polynomial model and write its predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CONFIG = {
    "var1": {"features": [f"x{i}" for i in range(1, 7)], "degree": 5,
             "model": "Lasso", "scaled": True, "alpha": 0.006, "l1_ratio": None},
    "var2": {"features": ["x1", "x2", "x3"], "degree": 10,
             "model": "Ridge", "scaled": False, "alpha": 0.1, "l1_ratio": None},
}


def make_estimator(config):
    family = config["model"]
    if family == "OLS":
        return LinearRegression()
    if family == "Ridge":
        return Ridge(alpha=config["alpha"], solver="lsqr", tol=1e-6)
    if family == "Lasso":
        return Lasso(alpha=config["alpha"], max_iter=12000, tol=1e-3,
                     warm_start=True, selection="cyclic")
    if family == "ElasticNet":
        return ElasticNet(alpha=config["alpha"], l1_ratio=config["l1_ratio"],
                          max_iter=12000, tol=1e-3, warm_start=True,
                          selection="cyclic")
    raise ValueError(f"Unknown model family: {family}")


def run_variant(variant: str, data_dir: Path = Path("BT2024167"), output: Path | None = None):
    if variant not in CONFIG:
        raise ValueError(f"Unknown variant: {variant}")
    config = CONFIG[variant]
    train_path = data_dir / f"BT2024167_train_{variant}.csv"
    test_path = data_dir / f"BT2024167_test_{variant}.csv"
    output_path = output or Path(f"BT2024167_pred_{variant}.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    features = config["features"]
    if train.columns.tolist() != features + ["y"]:
        raise ValueError(f"Unexpected training columns in {train_path}: {train.columns.tolist()}")
    if test.columns.tolist() != features:
        raise ValueError(f"Unexpected test columns in {test_path}: {test.columns.tolist()}")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Input data contains missing values")

    model = make_pipeline(PolynomialFeatures(degree=config["degree"], include_bias=False))
    if config["scaled"]:
        model.steps.append(("standardscaler", StandardScaler()))
    model.steps.append(("regressor", make_estimator(config)))
    model.fit(train[features], train["y"])
    predictions = model.predict(test[features])
    submission = pd.DataFrame({"y": predictions})
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    if len(submission) != len(test):
        raise AssertionError("Prediction count does not match test row count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Variant: {variant}; selected degree: {config['degree']}; model: {config['model']}; "
          f"scaled: {config['scaled']}; alpha: {config['alpha']}")
    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(f"Columns: {submission.columns.tolist()}; NaN count: {int(submission.isna().sum().sum())}")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CONFIG, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("BT2024167"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_variant(args.variant, args.data_dir, args.output)


if __name__ == "__main__":
    main()
