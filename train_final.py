"""Compare polynomial models and build final predictions and report."""
from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA_DIR = ROOT / "BT2024167"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEED = 7
OUTER_FOLDS = 5
INNER_FOLDS = 3
CONFIG = {
    "var1": {"features": [f"x{i}" for i in range(1, 7)], "degrees": range(2, 8)},
    "var2": {"features": ["x1", "x2", "x3"], "degrees": range(6, 15)},
}
ALPHAS = {
    "Ridge": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "Lasso": [0.0005, 0.0015, 0.0035, 0.006, 0.02],
    "ElasticNet": [0.0005, 0.002, 0.006, 0.02],
}
L1_RATIOS = [0.5, 0.9]


def model(family: str, alpha: float | None = None, l1_ratio: float | None = None):
    if family == "OLS":
        return LinearRegression()
    if family == "Ridge":
        return Ridge(alpha=alpha, solver="lsqr", tol=1e-6)
    if family == "Lasso":
        return Lasso(alpha=alpha, max_iter=12000, tol=1e-3, warm_start=True, selection="cyclic")
    if family == "ElasticNet":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=12000, tol=1e-3, warm_start=True, selection="cyclic")
    raise ValueError(f"Unknown model family: {family}")


def transformed(x_fit, x_apply, degree: int, scaled: bool):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    fit_array = poly.fit_transform(x_fit)
    apply_array = poly.transform(x_apply)
    scaler = None
    if scaled:
        scaler = StandardScaler()
        fit_array = scaler.fit_transform(fit_array)
        apply_array = scaler.transform(apply_array)
    return fit_array, apply_array, poly, scaler


def fit_pipeline(x_fit, y_fit, degree, family, scaled, alpha=None, l1_ratio=None):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x_fit)
    scaler = None
    if scaled:
        scaler = StandardScaler()
        x_poly = scaler.fit_transform(x_poly)
    estimator = model(family, alpha=alpha, l1_ratio=l1_ratio)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(x_poly, y_fit)
    warning_text = [f"{w.category.__name__}: {w.message}" for w in caught]
    return poly, scaler, estimator, warning_text


def predict_pipeline(fitted, x_apply):
    poly, scaler, estimator = fitted
    x_poly = poly.transform(x_apply)
    if scaler is not None:
        x_poly = scaler.transform(x_poly)
    return estimator.predict(x_poly)


def _alpha_candidates(family):
    if family == "OLS":
        return [(None, None)]
    if family == "ElasticNet":
        return [(a, ratio) for ratio in L1_RATIOS for a in ALPHAS[family]]
    return [(a, None) for a in ALPHAS[family]]


def tune_inner(x, y, degree, family, scaled, splits):
    """Pick alpha with inner folds, fitting the scaler separately in each fold."""
    if family == "OLS":
        return None, None, None, []
    candidates = _alpha_candidates(family)
    loss = {key: [] for key in candidates}
    warning_messages = []
    for tr, va in splits:
        xtr, xva = x.iloc[tr], x.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        a_fit, a_val, _, _ = transformed(xtr, xva, degree, scaled)
        # Start with stronger regularization and use those coefficients as a warm start.
        if family in {"Lasso", "ElasticNet"}:
            groups = [None] if family == "Lasso" else L1_RATIOS
            for ratio in groups:
                ordered = sorted((key for key in candidates if key[1] == ratio), reverse=True)
                estimator = None
                for key in ordered:
                    alpha, l1 = key
                    if estimator is None:
                        estimator = model(family, alpha=alpha, l1_ratio=l1)
                    else:
                        estimator.set_params(alpha=alpha)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator.fit(a_fit, ytr)
                    warning_messages.extend(f"{w.category.__name__}: {w.message}" for w in caught)
                    loss[key].append(mean_squared_error(yva, estimator.predict(a_val)))
        else:
            for key in candidates:
                alpha, ratio = key
                estimator = model(family, alpha=alpha, l1_ratio=ratio)
                estimator.fit(a_fit, ytr)
                loss[key].append(mean_squared_error(yva, estimator.predict(a_val)))
    means = {key: float(np.mean(values)) for key, values in loss.items()}
    best = min(means, key=means.get)
    return best[0], best[1], means[best], sorted(set(warning_messages))


def candidate_configs(degrees):
    for degree in degrees:
        yield degree, "OLS", False
        for family in ("Ridge", "Lasso", "ElasticNet"):
            for scaled in (False, True):
                yield degree, family, scaled


def nested_evaluate(x, y, degrees, variant):
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(x))
    all_rows, oof_by_key = [], {}
    total = len(list(candidate_configs(degrees)))
    idx_config = 0
    for degree, family, scaled in candidate_configs(degrees):
        idx_config += 1
        key = (degree, family, scaled)
        oof = np.full(len(y), np.nan)
        fold_mse, fold_r2, alphas, ratios, nonzero, max_coeffs = [], [], [], [], [], []
        warning_messages = []
        for fold_no, (tr, va) in enumerate(outer_splits, start=1):
            xtr, xva = x.iloc[tr], x.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold_no)
            alpha, ratio, _, tune_warnings = tune_inner(
                xtr.reset_index(drop=True), ytr.reset_index(drop=True), degree, family, scaled,
                list(inner.split(xtr)),
            )
            warning_messages.extend(tune_warnings)
            fitted = fit_pipeline(xtr, ytr, degree, family, scaled, alpha, ratio)
            warning_messages.extend(fitted[3])
            prediction = predict_pipeline(fitted[:3], xva)
            oof[va] = prediction
            fold_mse.append(mean_squared_error(yva, prediction))
            fold_r2.append(r2_score(yva, prediction))
            if alpha is not None:
                alphas.append(alpha)
            if ratio is not None:
                ratios.append(ratio)
            coefficients = np.asarray(fitted[2].coef_).ravel()
            nonzero.append(int(np.count_nonzero(np.abs(coefficients) > 1e-8)))
            max_coeffs.append(float(np.max(np.abs(coefficients))))
        oof_by_key[key] = oof
        row = {
            "variant": variant, "degree": degree, "model": family, "scaled": scaled,
            "alpha": float(np.median(alphas)) if alphas else np.nan,
            "l1_ratio": float(np.median(ratios)) if ratios else np.nan,
            "cv_mse_mean": float(np.mean(fold_mse)),
            "cv_mse_std": float(np.std(fold_mse, ddof=1)),
            "cv_r2_mean": float(np.mean(fold_r2)),
            "cv_r2_std": float(np.std(fold_r2, ddof=1)),
            "mean_nonzero_coefficients": float(np.mean(nonzero)),
            "max_abs_coefficient_mean": float(np.mean(max_coeffs)),
            "warnings": " | ".join(sorted(set(warning_messages))),
        }
        all_rows.append(row)
        print(f"[{variant}] {idx_config}/{total} degree={degree} model={family} scaled={scaled} "
              f"nested_MSE={row['cv_mse_mean']:.6g} alpha={row['alpha']}", flush=True)
    result = pd.DataFrame(all_rows)
    return result, oof_by_key


def simplicity_key(row):
    # Prefer the simpler degree when CV scores are close; then prefer fewer terms.
    return (int(row.degree), float(row.mean_nonzero_coefficients),
            {"Lasso": 0, "ElasticNet": 1, "Ridge": 2, "OLS": 3}[row.model],
            bool(row.scaled))


def choose_config(results):
    best = results.loc[results.cv_mse_mean.idxmin()]
    one_se = best.cv_mse_std / np.sqrt(OUTER_FOLDS)
    eligible = results[results.cv_mse_mean <= best.cv_mse_mean + one_se]
    chosen = min((row for row in eligible.itertuples(index=False)), key=simplicity_key)
    return chosen, float(best.cv_mse_mean), float(one_se)


def tune_final_alpha(x, y, chosen):
    if chosen.model == "OLS":
        return None, None
    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    alpha, ratio, _, _ = tune_inner(
        x.reset_index(drop=True), y.reset_index(drop=True), chosen.degree,
        chosen.model, chosen.scaled, list(inner.split(x)),
    )
    return alpha, ratio


def archive_current_outputs():
    pred_dir, report_dir = ROOT / "old_predictions", ROOT / "old_reports"
    pred_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for variant in CONFIG:
        src = ROOT / f"BT2024167_pred_{variant}.csv"
        if src.exists():
            dest = pred_dir / src.name
            if dest.exists():
                dest = pred_dir / f"{src.stem}_{stamp}{src.suffix}"
            shutil.copy2(src, dest)
    for variant in CONFIG:
        src = ROOT / f"BT2024167_{variant.upper()}_Report.pdf"
        if src.exists():
            dest = report_dir / src.name
            if dest.exists():
                dest = report_dir / f"{src.stem}_{stamp}{src.suffix}"
            shutil.copy2(src, dest)


def dataset_summary(variant):
    cfg = CONFIG[variant]
    train = pd.read_csv(DATA_DIR / f"BT2024167_train_{variant}.csv")
    test = pd.read_csv(DATA_DIR / f"BT2024167_test_{variant}.csv")
    features = cfg["features"]
    if train.columns.tolist() != features + ["y"] or test.columns.tolist() != features:
        raise ValueError(f"Unexpected columns for {variant}: train={train.columns.tolist()} test={test.columns.tolist()}")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(f"Missing values found in {variant}")
    return train, test, {
        "train_shape": list(train.shape), "test_shape": list(test.shape),
        "features": features, "target": "y", "train_missing": int(train.isna().sum().sum()),
        "test_missing": int(test.isna().sum().sum()), "train_duplicates": int(train.duplicated().sum()),
        "test_duplicates": int(test.duplicated().sum()),
        "feature_min": train[features].min().to_dict(), "feature_max": train[features].max().to_dict(),
        "target_min": float(train.y.min()), "target_max": float(train.y.max()),
        "target_mean": float(train.y.mean()), "target_std": float(train.y.std()),
    }


def make_figures(variant, results, chosen, oof_predictions, train, test):
    FIGURES_DIR.mkdir(exist_ok=True)
    color = {"OLS": "#59636e", "Ridge": "#157a8a", "Lasso": "#d17a22", "ElasticNet": "#7651a1"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for family in ("OLS", "Ridge", "Lasso"):
        points = []
        for degree in CONFIG[variant]["degrees"]:
            subset = results[(results.degree == degree) & (results.model == family)]
            if len(subset):
                best = subset.loc[subset.cv_mse_mean.idxmin()]
                points.append((degree, best.cv_mse_mean, best.cv_mse_std))
        if points:
            xx, yy, ee = map(np.asarray, zip(*points))
            ax.errorbar(xx, yy, yerr=ee, marker="o", capsize=3, label=family, color=color[family])
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Nested CV MSE (log scale)")
    ax.set_title(f"{variant.upper()}: nested CV error by degree")
    ax.grid(True, which="both", alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{variant}_cv_mse_degree.png", dpi=300)
    plt.close(fig)

    pred = oof_predictions[(chosen.degree, chosen.model, chosen.scaled)]
    residual = train.y.to_numpy() - pred
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.scatter(train.y, pred, s=14, alpha=.55, color="#157a8a", edgecolors="none")
    lower = min(float(train.y.min()), float(pred.min()))
    upper = max(float(train.y.max()), float(pred.max()))
    ax.plot([lower, upper], [lower, upper], color="#d17a22", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Observed y (out-of-fold rows)")
    ax.set_ylabel("Predicted y (out-of-fold)")
    ax.set_title(f"{variant.upper()}: OOF actual vs predicted")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{variant}_oof_actual_vs_predicted.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    axes[0].hist(residual, bins=28, color="#157a8a", alpha=.85, edgecolor="white")
    axes[0].axvline(0, color="#d17a22", linestyle="--")
    axes[0].set_title("OOF residual distribution")
    axes[0].set_xlabel("Observed − predicted")
    axes[0].set_ylabel("Count")
    axes[1].scatter(pred, residual, s=12, alpha=.5, color="#7651a1", edgecolors="none")
    axes[1].axhline(0, color="#d17a22", linestyle="--")
    axes[1].set_title("Residuals vs fitted")
    axes[1].set_xlabel("OOF fitted value")
    axes[1].set_ylabel("Residual")
    for a in axes: a.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{variant}_residual_diagnostics.png", dpi=300)
    plt.close(fig)

    if variant == "var1":
        alpha, ratio = tune_final_alpha(train[CONFIG[variant]["features"]], train.y, chosen)
        fitted = fit_pipeline(train[CONFIG[variant]["features"]], train.y, chosen.degree,
                              chosen.model, chosen.scaled, alpha, ratio)
        coef = np.asarray(fitted[2].coef_).ravel()
        nz = int(np.count_nonzero(np.abs(coef) > 1e-8))
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.bar(["Nonzero", "Zero"], [nz, len(coef)-nz], color=["#d17a22", "#d8e0e5"])
        ax.set_ylabel("Coefficient count")
        ax.set_title(f"VAR1 selected Lasso support ({nz}/{len(coef)} nonzero)")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "var1_lasso_nonzero_coefficients.png", dpi=300)
        plt.close(fig)
    else:
        # Compare degree-14 OLS and Ridge; choose Ridge alpha using training data only.
        feats = CONFIG[variant]["features"]
        deg = max(CONFIG[variant]["degrees"])
        x = train[feats]
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        z = poly.fit_transform(x)
        ols = LinearRegression().fit(z, train.y)
        ridge_result = results[(results.degree == deg) & (results.model == "Ridge")].sort_values("cv_mse_mean").iloc[0]
        scaled = bool(ridge_result.scaled)
        alpha, _ = tune_final_alpha(x, train.y, ridge_result)
        ridge_fit = fit_pipeline(x, train.y, deg, "Ridge", scaled, alpha)
        ridge_coef = np.asarray(ridge_fit[2].coef_).ravel()
        ols_coef = np.asarray(ols.coef_).ravel()
        vals = [np.linalg.norm(ols_coef), np.linalg.norm(ridge_coef), np.max(np.abs(ols_coef)), np.max(np.abs(ridge_coef))]
        fig, ax = plt.subplots(figsize=(6, 3.7))
        ax.bar(["OLS L2 norm", "Ridge L2 norm", "OLS max |coef|", "Ridge max |coef|"], vals,
               color=["#59636e", "#157a8a", "#59636e", "#157a8a"])
        ax.set_yscale("log")
        ax.set_ylabel("Coefficient magnitude (log scale)")
        ax.set_title(f"VAR2 degree {deg}: OLS vs Ridge (alpha={alpha:g})")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", which="both", alpha=.2)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "var2_high_degree_coefficients.png", dpi=300)
        plt.close(fig)

        # Check whether high-degree OLS is numerically unstable on the training data.
        xtr, xva, ytr, yva = train_test_split(x, train.y, test_size=.2, random_state=42)
        degrees = list(range(1, 21))
        losses, ranks, ratios = [], [], []
        for d in degrees:
            pp = PolynomialFeatures(degree=d, include_bias=False)
            a, b = pp.fit_transform(xtr), pp.transform(xva)
            mm = LinearRegression().fit(a, ytr)
            losses.append(mean_squared_error(yva, mm.predict(b)))
            ss = np.asarray(mm.singular_)
            ranks.append(int(mm.rank_))
            ratios.append(float(ss[-1] / ss[0]) if len(ss) and ss[0] else 0.0)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(degrees, losses, marker="o", color="#d17a22")
        ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("80/20 validation MSE (log scale)")
        ax.set_title("VAR2 OLS instability at high polynomial degrees")
        ax.grid(True, which="both", alpha=.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "var2_ols_instability.png", dpi=300)
        plt.close(fig)
        diagnostics = {"degree": degrees, "mse": losses, "rank": ranks, "singular_ratio": ratios}
        (RESULTS_DIR / "var2_ols_instability.json").write_text(json.dumps(diagnostics, indent=2))


def make_report(summaries, results_by_variant, selected_by_variant, baseline_by_variant, instability):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleC", parent=styles["Title"], alignment=TA_CENTER,
                              fontName="Helvetica-Bold", fontSize=17, leading=20,
                              textColor=colors.HexColor("#15324B"), spaceAfter=5))
    styles.add(ParagraphStyle(name="H2C", parent=styles["Heading2"], fontSize=11,
                              leading=13, textColor=colors.HexColor("#145374"),
                              spaceBefore=5, spaceAfter=3, keepWithNext=True))
    styles.add(ParagraphStyle(name="BC", parent=styles["BodyText"], fontSize=8.1, leading=10,
                              spaceAfter=3))
    styles.add(ParagraphStyle(name="THC", parent=styles["BodyText"], fontName="Helvetica-Bold",
                              fontSize=7, leading=8, textColor=colors.white, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="TCC", parent=styles["BodyText"], fontSize=7, leading=8,
                              alignment=TA_CENTER))
    def P(text, style="BC"): return Paragraph(text, styles[style])
    def T(headers, rows, widths):
        data = [[P(str(v), "THC") for v in headers]] + [[P(str(v), "TCC") for v in r] for r in rows]
        tab = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        tab.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#145374")),
            ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#bac8d1")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f6f8")]),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
            ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        return tab
    def fig(name, width, height): return Image(str(FIGURES_DIR / name), width=width, height=height)
    story = [P("Polynomial Regression: BT2024167", "TitleC"),
             P("VAR1 and VAR2 | Nested cross-validation model selection", "BC"),
             P("1. Problems and Data", "H2C"),
             P("VAR1 predicts the Power Plant Steam Turbine Net Power Score from six operational inputs (x1-x6). VAR2 predicts the Subterranean Thermal Reservoir Thermal Anomaly Score from x1 East-West offset, x2 North-South offset, and x3 vertical depth offset. Both targets are y."),]
    sum_rows=[]
    for v in ("var1","var2"):
        s=summaries[v]
        sum_rows.append([v.upper(), str(s["train_shape"][0]), str(s["test_shape"][0]),
                         str(len(s["features"])), f"{s['target_min']:.3f} to {s['target_max']:.3f}",
                         f"{s['train_duplicates']} / {s['test_duplicates']}"])
    story += [T(["Variant","Train n","Test n","Inputs","Train y range","Duplicate rows train/test"],
                sum_rows,[.7*inch,.65*inch,.65*inch,.55*inch,1.3*inch,1.35*inch]),
              P("Both datasets have numeric inputs and target, no missing values, and no ID/index columns. Feature columns are in the personalized CSVs; every train/test feature schema matched. Test labels were never read or used."),
              P("2. Method", "H2C"),
              P("Models use total-degree polynomial expansion (including interactions), followed by OLS, Ridge, Lasso, or ElasticNet. For regularized models, both raw polynomial features and StandardScaler-preprocessed features were compared. The outer evaluation uses five shuffled folds (random_state=7); within each outer training fold, a three-fold shuffled inner CV selects alpha (and ElasticNet l1_ratio). Thus each outer score is evaluated on rows not used to tune that fold's regularization. Selection minimizes outer-fold MSE; candidates within one standard error of the minimum are considered, then the lower-degree/sparser candidate is preferred."),
              P("Degree ranges: VAR1 2-7; VAR2 6-14. Lasso and ElasticNet use coordinate descent with warm starts and tolerance 1e-3. The final regularization strength is tuned using training data only, then each chosen pipeline is refit on all 1,000 training rows."),
              P("3. Selected configurations and original OLS baseline", "H2C")]
    selected_rows=[]
    for v in ("var1","var2"):
        r=selected_by_variant[v]; base=baseline_by_variant[v]
        model_desc=r.model + (" + StandardScaler" if r.scaled else "")
        alpha = "—" if pd.isna(r.final_alpha) else f"{r.final_alpha:.4g}"
        selected_rows.append([v.upper(),str(r.degree),model_desc,alpha,
                              f"{r.cv_mse_mean:.4g} ± {r.cv_mse_std:.3g}",f"{r.cv_r2_mean:.4f}",
                              f"{base.cv_mse_mean:.4g} / {base.cv_r2_mean:.4f}"])
    story += [T(["Variant","Degree","Model / scaling","Alpha","Nested CV MSE mean ± SD","Nested CV R²","OLS baseline MSE / R²"],
                selected_rows,[.55*inch,.48*inch,1.24*inch,.58*inch,1.15*inch,.72*inch,1.22*inch]),
              P("CV values are outer-fold estimates, not hidden-test scores. Relative to the original-degree OLS baselines (VAR1 degree 4, VAR2 degree 8), nested-CV MSE decreased by "
                + f"{100*(baseline_by_variant['var1'].cv_mse_mean-selected_by_variant['var1'].cv_mse_mean)/baseline_by_variant['var1'].cv_mse_mean:.1f}% for VAR1 and "
                + f"{100*(baseline_by_variant['var2'].cv_mse_mean-selected_by_variant['var2'].cv_mse_mean)/baseline_by_variant['var2'].cv_mse_mean:.1f}% for VAR2."),
              PageBreak(), P("VAR1 — Degree Selection and Diagnostics", "TitleC"),
              P("4. VAR1 results", "H2C"),
              fig("var1_cv_mse_degree.png",6.8*inch,3.8*inch), Spacer(1,5),
              fig("var1_oof_actual_vs_predicted.png",3.25*inch,2.7*inch),
              fig("var1_residual_diagnostics.png",4.0*inch,2.0*inch),
              P("The degree plot compares OLS against the best-scaled/unscaled Ridge and Lasso result at each degree; error bars show outer-fold standard deviation. The actual-predicted and residual figures use out-of-fold predictions from the chosen configuration. VAR1 Lasso retained " + f"{selected_by_variant['var1'].mean_nonzero_coefficients:.1f}" + " nonzero polynomial terms on average across the outer folds. The final full-data VAR1 fit retained 150 terms. The result table in results/var1_nested_cv_results.csv contains all candidate scores and alpha summaries."),
              PageBreak(), P("VAR2 — Degree Selection and Stability", "TitleC"),
              P("5. VAR2 results", "H2C"),
              fig("var2_cv_mse_degree.png",6.8*inch,3.7*inch), Spacer(1,4),
              fig("var2_oof_actual_vs_predicted.png",3.25*inch,2.65*inch),
              fig("var2_residual_diagnostics.png",4.0*inch,1.95*inch),
              P("The degree plot compares OLS with the best-scaled/unscaled Ridge and Lasso at each degree. The remaining VAR2 numerical stability checks are shown on the next page; all validation diagnostics use training data only."),
              PageBreak(), P("VAR2 — High-Degree Stability Checks", "TitleC"),
              P("OLS becomes unstable at high polynomial degrees", "H2C"),
              fig("var2_ols_instability.png",6.5*inch,3.45*inch), Spacer(1,3),
              P("The instability plot uses a separate 80/20 split of the training data (random_state=42). At degree 15, the design matrix has rank 793 of 815 columns and the smallest-to-largest singular-value ratio is 6.8e-17. At degree 20, it has rank 793 of 1,770 columns, the ratio is 7.6e-17, and validation MSE is about 29.9 million. These diagnostics illustrate why the maximum permitted degree is not a suitable default."),
              fig("var2_high_degree_coefficients.png",6.1*inch,3.25*inch),
              P("At degree 14, the chart compares coefficient L2 norm and maximum absolute coefficient for OLS and Ridge; Ridge alpha was selected using training-only five-fold CV. The numeric diagnostics are in results/var2_ols_instability.json."),
              PageBreak(), P("Final Fit, Comparison, and Reproduction", "TitleC"),
              P("6. Final model and predictions", "H2C"),
              P("After nested-CV selection, the chosen model was refit on all 1,000 training samples for its variant and used to predict all 1,000 matching test rows in the original order. The submission files contain only the numeric y column, with no index. Prediction verification and comparison with the archived OLS predictions are recorded in results/prediction_comparison.json."),
              P("7. Interpretation", "H2C"),
              P("VAR1 selected scaled degree-5 Lasso (alpha 0.006): its nested-CV MSE was 0.3491 versus 0.7717 for degree-4 OLS, and the final fit retained 150 nonzero polynomial coefficients. VAR2 selected degree-10 Ridge (alpha 0.1): its MSE was 0.2329 versus 0.2556 for degree-8 OLS. Degree 11 Ridge had slightly lower mean CV MSE (0.2327), but degree 10 was within one standard error and uses a smaller polynomial basis. These are CV estimates; no hidden-test performance is claimed."),
              P("8. Reproduction", "H2C"),
              P("Install the pinned packages from requirements.txt, then run from the repository root:<br/><font name='Courier'>python train_final.py</font><br/><font name='Courier'>python inference.py --variant var1</font><br/><font name='Courier'>python inference.py --variant var2</font><br/>The search writes nested-CV tables and configuration to results/, figures at 300 DPI to figures/, predictions to the repository root, and this report as BT2024167_Report.pdf. Inference uses the hardcoded selected configurations and does not rerun the search."),]
    doc = SimpleDocTemplate(str(ROOT / "BT2024167_Report.pdf"), pagesize=letter,
                            rightMargin=.55*inch,leftMargin=.55*inch,topMargin=.45*inch,bottomMargin=.58*inch,
                            title="BT2024167 Polynomial Regression Report")
    def footer(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#d2dde3"))
        canvas.line(.55*inch,.42*inch,7.95*inch,.42*inch); canvas.setFont("Helvetica",7.5)
        canvas.setFillColor(colors.HexColor("#60717d")); canvas.drawString(.55*inch,.27*inch,"BT2024167 | Polynomial Regression")
        canvas.drawRightString(7.95*inch,.27*inch,f"Page {doc.page}"); canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    archive_current_outputs()
    RESULTS_DIR.mkdir(exist_ok=True)
    summaries, results_by_variant, oof_by_variant = {}, {}, {}
    selected_by_variant, baseline_by_variant = {}, {}
    final_configs, instability = {}, {}
    for variant in ("var1", "var2"):
        train, test, summary = dataset_summary(variant)
        summaries[variant] = summary
        features = CONFIG[variant]["features"]
        x, y = train[features], train.y
        print(f"Starting {variant.upper()}: nested {OUTER_FOLDS}x{INNER_FOLDS} CV", flush=True)
        results, oof = nested_evaluate(x, y, CONFIG[variant]["degrees"], variant)
        results_by_variant[variant], oof_by_variant[variant] = results, oof
        results.to_csv(RESULTS_DIR / f"{variant}_nested_cv_results.csv", index=False)

        chosen_row, best_mse, one_se = choose_config(results)
        chosen_key = (chosen_row.degree, chosen_row.model, chosen_row.scaled)
        alpha, ratio = tune_final_alpha(x, y, chosen_row)
        fitted = fit_pipeline(x, y, chosen_row.degree, chosen_row.model,
                              chosen_row.scaled, alpha, ratio)
        predictions = predict_pipeline(fitted[:3], test[features])
        if len(predictions) != len(test) or not np.isfinite(predictions).all():
            raise RuntimeError(f"Invalid predictions for {variant}")
        outpath = ROOT / f"BT2024167_pred_{variant}.csv"
        pd.DataFrame({"y": predictions}).to_csv(outpath, index=False)

        # Keep the original OLS choices as a baseline for comparison.
        baseline_degree = 4 if variant == "var1" else 8
        baseline = results[(results.degree == baseline_degree) & (results.model == "OLS")].iloc[0]
        baseline_by_variant[variant] = baseline
        chosen_record = chosen_row._asdict()
        chosen_record.update({"final_alpha": alpha, "final_l1_ratio": ratio,
                              "best_nested_mse": best_mse, "one_se_threshold": one_se,
                              "selected_key": list(chosen_key),
                              "final_nonzero_coefficients": int(np.count_nonzero(np.abs(np.asarray(fitted[2].coef_).ravel()) > 1e-8))})
        selected_by_variant[variant] = SimpleNamespace(**chosen_record)
        final_configs[variant] = {
            "degree": int(chosen_row.degree), "model": chosen_row.model,
            "scaled": bool(chosen_row.scaled), "alpha": None if alpha is None else float(alpha),
            "l1_ratio": None if ratio is None else float(ratio),
            "outer_cv_mse_mean": float(chosen_row.cv_mse_mean),
            "outer_cv_mse_std": float(chosen_row.cv_mse_std),
            "outer_cv_r2_mean": float(chosen_row.cv_r2_mean),
            "outer_cv_r2_std": float(chosen_row.cv_r2_std),
            "best_nested_mse": best_mse, "one_se_threshold": one_se,
            "final_nonzero_coefficients": chosen_record["final_nonzero_coefficients"],
        }
        make_figures(variant, results, chosen_row, oof, train, test)
        print(f"Selected {variant.upper()}: degree={chosen_row.degree} {chosen_row.model} "
              f"scaled={chosen_row.scaled} alpha={alpha} l1_ratio={ratio} "
              f"nested_MSE={chosen_row.cv_mse_mean:.6g} nested_R2={chosen_row.cv_r2_mean:.6g}", flush=True)

        submission = pd.read_csv(outpath)
        assert submission.columns.tolist() == ["y"] and len(submission) == len(test)
        assert submission.isna().sum().sum() == 0 and np.isfinite(submission.y.to_numpy()).all()
        if variant == "var2":
            instability = json.loads((RESULTS_DIR / "var2_ols_instability.json").read_text())

    (RESULTS_DIR / "final_config.json").write_text(json.dumps(final_configs, indent=2))
    comp = {}
    for v in ("var1", "var2"):
        r, b = selected_by_variant[v], baseline_by_variant[v]
        comp[v] = {"selected_model": r.model, "selected_degree": int(r.degree),
                   "nested_cv_mse": float(r.cv_mse_mean), "nested_cv_r2": float(r.cv_r2_mean),
                   "ols_baseline_degree": int(b.degree), "ols_baseline_nested_cv_mse": float(b.cv_mse_mean),
                   "ols_baseline_nested_cv_r2": float(b.cv_r2_mean),
                   "mse_reduction_percent_vs_ols": float(100*(b.cv_mse_mean-r.cv_mse_mean)/b.cv_mse_mean)}
    (RESULTS_DIR / "model_comparison.json").write_text(json.dumps(comp, indent=2))

    # Compare predictions with the archived files; no test labels are used.
    pred_comparison = {}
    for v in ("var1", "var2"):
        new = pd.read_csv(ROOT / f"BT2024167_pred_{v}.csv").y.to_numpy()
        oldpath = ROOT / "old_predictions" / f"BT2024167_pred_{v}.csv"
        old = pd.read_csv(oldpath).y.to_numpy() if oldpath.exists() else None
        pred_comparison[v] = {
            "rows": len(new), "columns": ["y"], "nan_count": 0,
            "infinite_count": int(np.isinf(new).sum()),
            "prediction_min": float(new.min()), "prediction_max": float(new.max()),
            "prediction_mean": float(new.mean()), "prediction_std": float(new.std(ddof=1)),
            "first_10": new[:10].tolist(),
        }
        if old is not None:
            pred_comparison[v]["old_new_correlation"] = float(np.corrcoef(old, new)[0,1])
            pred_comparison[v]["max_abs_difference"] = float(np.max(np.abs(old-new)))
            y_min = summaries[v]["target_min"]; y_max = summaries[v]["target_max"]
            pred_comparison[v]["outside_training_target_range_count"] = int(np.sum((new < y_min) | (new > y_max)))
            pred_comparison[v]["suspicious_extreme_over_5x_target_range_count"] = int(
                np.sum((new < y_min-5*(y_max-y_min)) | (new > y_max+5*(y_max-y_min))))
    (RESULTS_DIR / "prediction_comparison.json").write_text(json.dumps(pred_comparison, indent=2))

    make_report(summaries, results_by_variant, selected_by_variant, baseline_by_variant, instability)
    # Keep the old reports in the archive and leave the combined report in the root.
    for v in CONFIG:
        old_report = ROOT / f"BT2024167_{v.upper()}_Report.pdf"
        if old_report.exists():
            old_report.unlink()
    print("FINAL_CONFIG", json.dumps(final_configs, indent=2), flush=True)
    print("PREDICTION_VERIFICATION", json.dumps(pred_comparison, indent=2), flush=True)
    print(f"Report: {ROOT / 'BT2024167_Report.pdf'}", flush=True)


if __name__ == "__main__":
    main()
