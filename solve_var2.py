"""Reproducible VAR2 polynomial-regression experiment and submission builder."""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "BT2024167" / "BT2024167_train_var2.csv"
TEST_PATH = ROOT / "BT2024167" / "BT2024167_test_var2.csv"
SUBMISSION_PATH = ROOT / "BT2024167_pred_var2.csv"
FEATURES = ["x1", "x2", "x3"]
TARGET = "y"
SEED = 42


def model_for(degree):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if train.columns.tolist() != FEATURES + [TARGET]:
        raise ValueError(f"Unexpected training columns: {train.columns.tolist()}")
    if test.columns.tolist() != FEATURES:
        raise ValueError(f"Unexpected test columns: {test.columns.tolist()}")
    if train.isna().sum().sum() or test.isna().sum().sum():
        raise ValueError("Missing values found")

    X, y = train[FEATURES], train[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    rows = []
    for degree in range(1, 21):
        started = time.perf_counter()
        caught = []
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            model = model_for(degree)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
        caught = [f"{type(w.message).__name__}: {w.message}" for w in warning_list]
        lin = model.named_steps["linearregression"]
        singular = np.asarray(lin.singular_)
        rows.append({
            "degree": degree,
            "polynomial_features": model.named_steps["polynomialfeatures"].n_output_features_,
            "validation_mse": mean_squared_error(y_val, pred),
            "validation_r2": r2_score(y_val, pred),
            "seconds": time.perf_counter() - started,
            "max_abs_coefficient": float(np.max(np.abs(lin.coef_))),
            "design_rank": int(lin.rank_),
            "design_columns": int(model.named_steps["polynomialfeatures"].n_output_features_),
            "smallest_to_largest_singular_ratio": float(singular[-1] / singular[0]) if len(singular) and singular[0] else 0.0,
            "max_abs_validation_prediction": float(np.max(np.abs(pred))),
            "warnings": " | ".join(caught),
        })
    results = pd.DataFrame(rows)
    print("DATASET_SUMMARY")
    print(json.dumps({
        "train_shape": list(train.shape), "test_shape": list(test.shape),
        "train_columns": train.columns.tolist(), "test_columns": test.columns.tolist(),
        "features": FEATURES, "target": TARGET,
        "train_dtypes": {k: str(v) for k, v in train.dtypes.items()},
        "test_dtypes": {k: str(v) for k, v in test.dtypes.items()},
        "train_missing": train.isna().sum().to_dict(),
        "test_missing": test.isna().sum().to_dict(),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "test_duplicate_rows": int(test.duplicated().sum()),
        "train_describe": train.describe().to_dict(),
        "test_describe": test.describe().to_dict(),
    }, indent=2))
    print("VALIDATION_RESULTS")
    print(results[["degree", "polynomial_features", "validation_mse", "validation_r2"]].to_string(index=False, float_format=lambda x: f"{x:.10g}"))
    print("STABILITY_DIAGNOSTICS")
    print(results[["degree", "seconds", "max_abs_coefficient", "design_rank", "design_columns", "smallest_to_largest_singular_ratio", "max_abs_validation_prediction", "warnings"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    # CV the three degrees with the lowest split-validation MSE.
    candidates = results.nsmallest(3, "validation_mse")["degree"].tolist()
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for degree in candidates:
        scores = cross_validate(
            model_for(degree), X, y, cv=cv,
            scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            n_jobs=1, return_train_score=True,
        )
        cv_rows.append({
            "degree": degree,
            "cv_mse_mean": -scores["test_mse"].mean(),
            "cv_mse_std": scores["test_mse"].std(ddof=1),
            "cv_r2_mean": scores["test_r2"].mean(),
            "cv_r2_std": scores["test_r2"].std(ddof=1),
            "train_mse_mean": -scores["train_mse"].mean(),
        })
    cv_results = pd.DataFrame(cv_rows).sort_values("cv_mse_mean")
    print("ROBUSTNESS_5FOLD")
    print(cv_results.to_string(index=False, float_format=lambda x: f"{x:.10g}"))

    # Select by split validation MSE; simplest degree breaks near-exact ties.
    best_mse = results["validation_mse"].min()
    selected = int(results[results["validation_mse"] <= best_mse + 1e-10].sort_values("degree").iloc[0]["degree"])
    final_model = model_for(selected)
    final_model.fit(X, y)
    predictions = final_model.predict(test[FEATURES])
    pd.DataFrame({TARGET: predictions}).to_csv(SUBMISSION_PATH, index=False)

    submission = pd.read_csv(SUBMISSION_PATH)
    if submission.columns.tolist() != [TARGET] or len(submission) != len(test):
        raise AssertionError("Submission shape or columns are incorrect")
    values = submission[TARGET].to_numpy()
    if not pd.api.types.is_numeric_dtype(submission[TARGET]) or not np.isfinite(values).all():
        raise AssertionError("Submission has non-numeric or non-finite predictions")
    print("SELECTION_AND_SUBMISSION")
    print(json.dumps({
        "selected_degree": selected,
        "validation_mse": float(results.loc[results.degree == selected, "validation_mse"].iloc[0]),
        "validation_r2": float(results.loc[results.degree == selected, "validation_r2"].iloc[0]),
        "cv_results_for_selected_degree": cv_results[cv_results.degree == selected].to_dict(orient="records"),
        "submission_file": str(SUBMISSION_PATH.resolve()),
        "submission_rows": len(submission), "submission_columns": submission.columns.tolist(),
        "nan_count": int(submission.isna().sum().sum()),
        "infinite_count": int(np.isinf(values).sum()),
        "first_10_predictions": values[:10].tolist(),
        "prediction_min": float(values.min()), "prediction_max": float(values.max()),
        "prediction_mean": float(values.mean()), "prediction_std": float(values.std(ddof=1)),
    }, indent=2))


if __name__ == "__main__":
    main()
