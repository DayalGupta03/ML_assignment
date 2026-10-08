"""Reproducible VAR1 polynomial-regression experiment and submission builder."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "BT2024167" / "BT2024167_train_var1.csv"
TEST_PATH = ROOT / "BT2024167" / "BT2024167_test_var1.csv"
SUBMISSION_PATH = ROOT / "legacy_ols" / "predictions" / "BT2024167_pred_var1_ols.csv"
FEATURES = [f"x{i}" for i in range(1, 7)]
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
    assert train.columns.tolist() == FEATURES + [TARGET]
    assert test.columns.tolist() == FEATURES
    assert train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0

    X, y = train[FEATURES], train[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    rows = []
    for degree in range(1, 11):
        started = time.perf_counter()
        model = model_for(degree)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rows.append({
            "degree": degree,
            "polynomial_features": model.named_steps["polynomialfeatures"].n_output_features_,
            "validation_mse": mean_squared_error(y_val, pred),
            "validation_r2": r2_score(y_val, pred),
            "seconds": time.perf_counter() - started,
        })
    results = pd.DataFrame(rows)
    print("DATASET_SUMMARY")
    print(json.dumps({
        "train_shape": list(train.shape), "test_shape": list(test.shape),
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
    print(results.to_string(index=False, float_format=lambda x: f"{x:.8g}"))

    # Additional robustness check for the three best split-validation degrees.
    contenders = results.nsmallest(3, "validation_mse")["degree"].tolist()
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for degree in contenders:
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
    print(cv_results.to_string(index=False, float_format=lambda x: f"{x:.8g}"))

    # Select by validation MSE. In a numerical tie within 1e-10, prefer the
    # simpler degree; preserve the validation R2 as a secondary diagnostic.
    best_mse = results["validation_mse"].min()
    tied = results[results["validation_mse"] <= best_mse + 1e-10]
    selected = int(tied.sort_values("degree").iloc[0]["degree"])

    final_model = model_for(selected)
    final_model.fit(X, y)
    predictions = final_model.predict(test[FEATURES])
    submission = pd.DataFrame({TARGET: predictions})
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    reread = pd.read_csv(SUBMISSION_PATH)
    assert reread.columns.tolist() == [TARGET]
    assert len(reread) == len(test)
    assert pd.api.types.is_numeric_dtype(reread[TARGET])
    assert np.isfinite(reread[TARGET].to_numpy()).all()
    assert reread[TARGET].notna().all()
    print("SELECTION", json.dumps({
        "selected_degree": selected,
        "validation_mse": float(results.loc[results.degree == selected, "validation_mse"].iloc[0]),
        "validation_r2": float(results.loc[results.degree == selected, "validation_r2"].iloc[0]),
        "test_predictions": len(reread),
        "output_file": str(SUBMISSION_PATH),
        "prediction_stats": reread[TARGET].describe().to_dict(),
        "first_10_predictions": reread.head(10)[TARGET].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
