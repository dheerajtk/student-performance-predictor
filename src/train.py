"""
train.py — Full ML training pipeline for Student Performance Predictor.

What this script does (in order):
  1. Loads and engineers features from student-mat.csv
  2. Splits into 80/20 train/test (stratified on pass_fail)
  3. Scales features with StandardScaler
  4. Trains 3 regression models  (predict G3, final grade 0-20)
  5. Trains 3 classification models (predict Pass/Fail)
  6. Evaluates every model and prints a comparison table
  7. Saves the best regression + best classification model as .pkl
  8. Saves the scaler, feature list, and a metadata JSON

Models used
-----------
  Regression:
    1. Linear Regression           -- baseline
    2. Ridge Regression (L2)       -- regularized, handles multicollinearity
    3. Polynomial Ridge (degree=2) -- captures non-linear interactions

  Classification:
    1. Logistic Regression (C=1)   -- baseline
    2. Logistic Regression (C=0.1) -- stronger regularization
    3. Polynomial Logistic (deg=2) -- captures non-linear interactions

  Note: RandomForest/GradientBoosting (.pyd Cython extensions) are blocked
  by the Windows Application Control policy on this machine. The three models
  above provide a valid comparison spanning: linear, regularized-linear,
  and polynomial-feature-expanded non-linear models.

Run:
    python -m src.train

Outputs (saved to models/):
    scaler.pkl
    best_regressor.pkl
    best_classifier.pkl
    feature_importance_reg.png
    feature_importance_clf.png
    confusion_matrix.png
    model_metadata.json
"""

import os
import sys
import json
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(_ROOT, "data", "student-mat.csv")
MODELS_DIR = os.path.join(_ROOT, "models")

FEATURE_COLS = ["studytime", "attendance_pct", "G1", "failures", "absences"]
TARGET_REG   = "G3"        # regression target  (0-20)
TARGET_CLF   = "pass_fail" # classification target (0=Fail, 1=Pass)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def _sep() -> None:
    print("-" * 60)


# --------------------------------------------------------------------------- #
# Step 1 – Load data
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    from src.load_data import load_dataset
    _header("Step 1 — Loading dataset")
    df = load_dataset(DATA_PATH)
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Pass    : {df[TARGET_CLF].sum()} ({df[TARGET_CLF].mean()*100:.1f}%)")
    print(f"  Fail    : {(df[TARGET_CLF]==0).sum()} ({(df[TARGET_CLF]==0).mean()*100:.1f}%)")
    return df


# --------------------------------------------------------------------------- #
# Step 2 – Split
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame):
    _header("Step 2 — Train/Test split (80/20, stratified on pass_fail)")

    X = df[FEATURE_COLS]
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = \
        train_test_split(
            X, y_reg, y_clf,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_clf,
        )

    print(f"  Train : {len(X_train)} samples  "
          f"(pass rate {y_clf_train.mean()*100:.1f}%)")
    print(f"  Test  : {len(X_test)}  samples  "
          f"(pass rate {y_clf_test.mean()*100:.1f}%)")

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


# --------------------------------------------------------------------------- #
# Step 3 – Scale
# --------------------------------------------------------------------------- #
def fit_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame):
    _header("Step 3 — Fitting StandardScaler on training set")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("  Feature means and std from training data:")
    for feat, mean, std in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
        print(f"    {feat:<18}  mean={mean:7.3f}  std={std:6.3f}")

    return scaler, X_train_s, X_test_s


# --------------------------------------------------------------------------- #
# Step 4 – Model definitions
# --------------------------------------------------------------------------- #
def _poly_ridge(degree: int = 2, alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=alpha)),
    ])


def _poly_logistic(degree: int = 2, C: float = 0.1) -> Pipeline:
    return Pipeline([
        ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
        ("log",   LogisticRegression(C=C, max_iter=2000,
                                     random_state=RANDOM_STATE)),
    ])


def get_regression_models() -> dict:
    return {
        "Linear Regression":          LinearRegression(),
        "Ridge Regression (L2)":      Ridge(alpha=1.0),
        "Polynomial Ridge (degree=2)": _poly_ridge(degree=2, alpha=1.0),
    }


def get_classification_models() -> dict:
    return {
        "Logistic Regression (C=1)":   LogisticRegression(
                                            C=1.0, max_iter=1000,
                                            random_state=RANDOM_STATE),
        "Logistic Regression (C=0.1)": LogisticRegression(
                                            C=0.1, max_iter=1000,
                                            random_state=RANDOM_STATE),
        "Polynomial Logistic (deg=2)": _poly_logistic(degree=2, C=0.1),
    }


# --------------------------------------------------------------------------- #
# Step 5a – Train & evaluate regression models
# --------------------------------------------------------------------------- #
def train_regression(models: dict, X_train, X_test, y_train, y_test):
    _header("Step 4a — Regression Models  (target: G3, 0-20)")
    print(f"  {'Model':<35} {'RMSE':>6}  {'MAE':>5}  {'R2':>6}  {'CV-R2':>7}")
    _sep()

    results, trained = {}, {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae    = float(mean_absolute_error(y_test, y_pred))
        r2     = float(r2_score(y_test, y_pred))

        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_r2 = float(cv.mean())

        results[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4),
                         "r2": round(r2, 4), "cv_r2": round(cv_r2, 4),
                         "fit_time_s": round(elapsed, 3)}
        trained[name] = model

        print(f"  {name:<35} {rmse:6.3f}  {mae:5.3f}  {r2:6.3f}  {cv_r2:7.3f}"
              f"  [{elapsed:.2f}s]")

    best = max(results, key=lambda n: results[n]["r2"])
    print(f"\n  Best regressor : {best}  (R2={results[best]['r2']:.4f})")
    return trained, results, best


# --------------------------------------------------------------------------- #
# Step 5b – Train & evaluate classification models
# --------------------------------------------------------------------------- #
def train_classification(models: dict, X_train, X_test, y_train, y_test):
    _header("Step 4b — Classification Models  (target: Pass/Fail)")
    print(f"  {'Model':<35} {'Acc':>5}  {'F1':>5}  {'AUC':>6}  {'CV-Acc':>7}")
    _sep()

    results, trained = {}, {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc    = float(accuracy_score(y_test, y_pred))
        f1     = float(f1_score(y_test, y_pred))
        auc    = float(roc_auc_score(y_test, y_prob))

        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_acc = float(cv.mean())

        results[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4),
                         "auc": round(auc, 4), "cv_acc": round(cv_acc, 4),
                         "fit_time_s": round(elapsed, 3)}
        trained[name] = model

        print(f"  {name:<35} {acc:5.3f}  {f1:5.3f}  {auc:6.3f}  {cv_acc:7.3f}"
              f"  [{elapsed:.2f}s]")

    best = max(results, key=lambda n: results[n]["auc"])
    print(f"\n  Best classifier: {best}  (AUC={results[best]['auc']:.4f})")

    _sep()
    print(f"\n  Classification Report -- {best}:")
    print(classification_report(y_test, trained[best].predict(X_test),
                                target_names=["Fail", "Pass"], digits=3))

    return trained, results, best


# --------------------------------------------------------------------------- #
# Step 6 – Feature importance
# --------------------------------------------------------------------------- #
def _extract_coef(model) -> np.ndarray:
    """
    Return per-original-feature importance from linear or pipeline model.
    For polynomial pipelines we sum |coef| mapped back to original features.
    """
    if isinstance(model, Pipeline):
        poly  = model.named_steps["poly"]
        final = model.named_steps.get("ridge") or model.named_steps.get("log")
        coefs = np.abs(final.coef_.ravel())
        # Map polynomial feature importances back to original features
        powers = poly.powers_         # shape (n_poly_feats, n_orig_feats)
        n_orig = len(FEATURE_COLS)
        orig_imp = np.zeros(n_orig)
        for coef, power in zip(coefs, powers):
            for j, p in enumerate(power):
                if p > 0:
                    orig_imp[j] += coef / p   # weight by degree
        total = orig_imp.sum() or 1.0
        return orig_imp / total
    else:
        coefs = np.abs(model.coef_.ravel())
        total = coefs.sum() or 1.0
        return coefs / total


def plot_feature_importance(model, name: str, suffix: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    importances = _extract_coef(model)
    indices  = np.argsort(importances)[::-1]
    features = [FEATURE_COLS[i] for i in indices]
    values   = importances[indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#3b82d4" if v == values.max() else "#93c5fd" for v in values]
    ax.barh(range(len(features)), values[::-1], color=colors[::-1],
            edgecolor="white")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features[::-1], fontsize=11)
    ax.set_xlabel("Relative Importance (|coef| normalised)", fontsize=10)
    ax.set_title(f"{name}\nFeature Importances", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(MODELS_DIR, f"feature_importance_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(path, _ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# Step 7 – Confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(model, X_test, y_test, name: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(path, _ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# Step 8 – Save artefacts
# --------------------------------------------------------------------------- #
def save_artefacts(scaler, best_reg_model, best_reg_name,
                   best_clf_model, best_clf_name,
                   reg_results, clf_results) -> None:
    _header("Step 6 — Saving artefacts")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for obj, fname in [
        (scaler,         "scaler.pkl"),
        (best_reg_model, "best_regressor.pkl"),
        (best_clf_model, "best_classifier.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        joblib.dump(obj, path)
        print(f"  Saved -> {os.path.relpath(path, _ROOT)}")

    meta = {
        "feature_cols":    FEATURE_COLS,
        "target_reg":      TARGET_REG,
        "target_clf":      TARGET_CLF,
        "pass_threshold":  10,
        "best_regressor":  {"name": best_reg_name,
                            "metrics": reg_results[best_reg_name]},
        "best_classifier": {"name": best_clf_name,
                            "metrics": clf_results[best_clf_name]},
        "all_regression_results":     reg_results,
        "all_classification_results": clf_results,
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved -> {os.path.relpath(meta_path, _ROOT)}")


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run_pipeline() -> None:
    df = load_data()

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = \
        split_data(df)

    scaler, X_train_s, X_test_s = fit_scaler(
        pd.DataFrame(X_train, columns=FEATURE_COLS),
        pd.DataFrame(X_test,  columns=FEATURE_COLS),
    )

    reg_models = get_regression_models()
    reg_trained, reg_results, best_reg = train_regression(
        reg_models, X_train_s, X_test_s, y_reg_train.values, y_reg_test.values
    )

    clf_models = get_classification_models()
    clf_trained, clf_results, best_clf = train_classification(
        clf_models, X_train_s, X_test_s, y_clf_train.values, y_clf_test.values
    )

    _header("Step 5 — Feature Importance Plots")
    plot_feature_importance(reg_trained[best_reg], best_reg, "reg")
    plot_feature_importance(clf_trained[best_clf], best_clf, "clf")
    plot_confusion_matrix(clf_trained[best_clf], X_test_s,
                          y_clf_test.values, best_clf)

    save_artefacts(
        scaler,
        reg_trained[best_reg], best_reg,
        clf_trained[best_clf], best_clf,
        reg_results, clf_results,
    )

    _header("Training Complete")
    print(f"  Best Regressor  : {best_reg}")
    r = reg_results[best_reg]
    print(f"    RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}  R2={r['r2']:.3f}  CV-R2={r['cv_r2']:.3f}")
    print(f"  Best Classifier : {best_clf}")
    c = clf_results[best_clf]
    print(f"    Acc={c['accuracy']:.3f}  F1={c['f1']:.3f}  AUC={c['auc']:.3f}  CV-Acc={c['cv_acc']:.3f}")
    print("\n  All artefacts saved to: models/")


if __name__ == "__main__":
    run_pipeline()
