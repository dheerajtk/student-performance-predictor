"""
predict.py — Prediction logic for Student Performance Predictor.

Loads the trained scaler + models once at import time.
Exposes two public functions:
    predict(inputs)  -> dict with grade, pass_fail, probability, explanation
    get_metadata()   -> dict with model names, metrics, feature importance
"""

import os
import json
import joblib
import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "models")

_SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
_REG_PATH      = os.path.join(MODELS_DIR, "best_regressor.pkl")
_CLF_PATH      = os.path.join(MODELS_DIR, "best_classifier.pkl")
_META_PATH     = os.path.join(MODELS_DIR, "model_metadata.json")

FEATURE_COLS   = ["studytime", "attendance_pct", "G1", "failures", "absences"]
PASS_THRESHOLD = 10

# Friendly display names for features
FEATURE_LABELS = {
    "studytime":      "Study Time",
    "attendance_pct": "Attendance %",
    "G1":             "Previous Marks (G1)",
    "failures":       "Previous Failures",
    "absences":       "Absences",
}

# --------------------------------------------------------------------------- #
# Input validation spec
# --------------------------------------------------------------------------- #
INPUT_SPEC = {
    "studytime":      {"type": int,   "min": 1,   "max": 4},
    "attendance_pct": {"type": float, "min": 0.0, "max": 100.0},
    "G1":             {"type": float, "min": 0.0, "max": 20.0},
    "failures":       {"type": int,   "min": 0,   "max": 4},
    "absences":       {"type": int,   "min": 0,   "max": 93},
}

# Studytime label map
STUDYTIME_LABELS = {
    1: "<2 hours/week",
    2: "2-5 hours/week",
    3: "5-10 hours/week",
    4: ">10 hours/week",
}

# --------------------------------------------------------------------------- #
# Model loading (lazy, cached)
# --------------------------------------------------------------------------- #
_scaler    = None
_regressor = None
_classifier = None
_metadata  = None


def _load_models():
    global _scaler, _regressor, _classifier, _metadata
    if _scaler is not None:
        return  # already loaded

    for path, label in [
        (_SCALER_PATH, "scaler"),
        (_REG_PATH,    "best_regressor"),
        (_CLF_PATH,    "best_classifier"),
        (_META_PATH,   "model_metadata"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model file not found: {path}\n"
                "Run  python -m src.train  first."
            )

    _scaler     = joblib.load(_SCALER_PATH)
    _regressor  = joblib.load(_REG_PATH)
    _classifier = joblib.load(_CLF_PATH)

    with open(_META_PATH) as f:
        _metadata = json.load(f)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_inputs(raw: dict) -> tuple[dict, list]:
    """
    Validate and coerce raw input dict.
    Returns (clean_dict, errors_list).
    errors_list is empty on success.
    """
    clean  = {}
    errors = []

    for col, spec in INPUT_SPEC.items():
        val = raw.get(col)
        if val is None or str(val).strip() == "":
            errors.append(f"'{FEATURE_LABELS[col]}' is required.")
            continue

        try:
            coerced = spec["type"](val)
        except (ValueError, TypeError):
            errors.append(
                f"'{FEATURE_LABELS[col]}' must be a number "
                f"(got '{val}')."
            )
            continue

        if coerced < spec["min"] or coerced > spec["max"]:
            errors.append(
                f"'{FEATURE_LABELS[col]}' must be between "
                f"{spec['min']} and {spec['max']} "
                f"(got {coerced})."
            )
            continue

        clean[col] = coerced

    return clean, errors


# --------------------------------------------------------------------------- #
# Grade interpretation helpers
# --------------------------------------------------------------------------- #
def _grade_label(g3: float) -> str:
    """Map a 0-20 G3 score to a Portuguese grade label."""
    if g3 >= 18:  return "Excellent (18-20)"
    if g3 >= 15:  return "Good (15-17)"
    if g3 >= 10:  return "Sufficient (10-14)"
    if g3 >= 5:   return "Poor (5-9)"
    return "Very Poor (0-4)"


def _confidence_label(prob: float) -> str:
    if prob >= 0.85:  return "Very High"
    if prob >= 0.70:  return "High"
    if prob >= 0.55:  return "Moderate"
    return "Low"


# --------------------------------------------------------------------------- #
# Plain-language explanation builder
# --------------------------------------------------------------------------- #
def _build_explanation(inputs: dict, predicted_g3: float,
                        pass_prob: float) -> list[str]:
    """Return a list of bullet-point strings explaining the prediction."""
    lines = []

    # G1 (strongest predictor)
    g1 = inputs["G1"]
    if g1 >= 15:
        lines.append(
            f"Your previous mark of {g1}/20 is strong — students with "
            "high G1 scores consistently achieve high final grades."
        )
    elif g1 >= 10:
        lines.append(
            f"Your previous mark of {g1}/20 is average. "
            "Improving your study habits could push you to a higher grade."
        )
    else:
        lines.append(
            f"Your previous mark of {g1}/20 is below average. "
            "This is the biggest risk factor for your final grade — "
            "seek extra support early."
        )

    # Failures
    f = inputs["failures"]
    if f == 0:
        lines.append("No previous failures — this is a positive sign.")
    elif f == 1:
        lines.append(
            "1 previous failure slightly increases risk. "
            "Stay consistent with your study schedule."
        )
    else:
        lines.append(
            f"{f} previous failures significantly increase your risk. "
            "Consider additional tutoring or office hours."
        )

    # Study time
    st = inputs["studytime"]
    st_label = STUDYTIME_LABELS.get(st, str(st))
    if st >= 3:
        lines.append(f"Studying {st_label} is above average — keep it up.")
    elif st == 2:
        lines.append(
            f"Studying {st_label} is typical. Increasing to 5+ hours/week "
            "is associated with better outcomes."
        )
    else:
        lines.append(
            f"Studying {st_label} is below average. "
            "Increasing study time is one of the easiest improvements to make."
        )

    # Attendance / absences
    att = inputs["attendance_pct"]
    ab  = inputs["absences"]
    if att >= 90:
        lines.append(f"Attendance of {att:.0f}% is excellent.")
    elif att >= 75:
        lines.append(
            f"Attendance of {att:.0f}% ({ab} absences) is acceptable "
            "but improving it may help."
        )
    else:
        lines.append(
            f"Attendance of {att:.0f}% ({ab} absences) is low. "
            "Frequent absences are strongly linked to poor outcomes."
        )

    return lines


# --------------------------------------------------------------------------- #
# Feature importance for display
# --------------------------------------------------------------------------- #
def _get_feature_importance_reg() -> list[dict]:
    """Return sorted feature importances from the best regressor."""
    from sklearn.pipeline import Pipeline
    model = _regressor
    if isinstance(model, Pipeline):
        poly  = model.named_steps["poly"]
        final = model.named_steps.get("ridge") or model.named_steps.get("log")
        coefs = np.abs(final.coef_.ravel())
        powers = poly.powers_
        orig_imp = np.zeros(len(FEATURE_COLS))
        for coef, power in zip(coefs, powers):
            for j, p in enumerate(power):
                if p > 0:
                    orig_imp[j] += coef / p
        total = orig_imp.sum() or 1.0
        importances = (orig_imp / total).tolist()
    else:
        coefs = np.abs(model.coef_.ravel())
        total = coefs.sum() or 1.0
        importances = (coefs / total).tolist()

    result = [
        {"feature": FEATURE_LABELS[f], "importance": round(importances[i], 4)}
        for i, f in enumerate(FEATURE_COLS)
    ]
    return sorted(result, key=lambda x: x["importance"], reverse=True)


def _get_feature_importance_clf() -> list[dict]:
    """Return sorted feature importances from the best classifier."""
    from sklearn.pipeline import Pipeline
    model = _classifier
    if isinstance(model, Pipeline):
        poly  = model.named_steps["poly"]
        final = model.named_steps.get("log")
        coefs = np.abs(final.coef_.ravel())
        powers = poly.powers_
        orig_imp = np.zeros(len(FEATURE_COLS))
        for coef, power in zip(coefs, powers):
            for j, p in enumerate(power):
                if p > 0:
                    orig_imp[j] += coef / p
        total = orig_imp.sum() or 1.0
        importances = (orig_imp / total).tolist()
    else:
        coefs = np.abs(model.coef_.ravel())
        total = coefs.sum() or 1.0
        importances = (coefs / total).tolist()

    result = [
        {"feature": FEATURE_LABELS[f], "importance": round(importances[i], 4)}
        for i, f in enumerate(FEATURE_COLS)
    ]
    return sorted(result, key=lambda x: x["importance"], reverse=True)


# --------------------------------------------------------------------------- #
# Main predict function
# --------------------------------------------------------------------------- #
def predict(inputs: dict) -> dict:
    """
    Run prediction on a validated input dict.

    Parameters
    ----------
    inputs : dict with keys studytime, attendance_pct, G1, failures, absences
             (already validated by validate_inputs)

    Returns
    -------
    dict:
        predicted_g3      float   predicted final grade (0-20)
        predicted_g3_int  int     rounded grade
        grade_label       str     e.g. "Sufficient (10-14)"
        pass_fail         int     1=Pass, 0=Fail
        pass_fail_label   str     "Pass" or "Fail"
        pass_probability  float   probability of passing (0-1)
        fail_probability  float   probability of failing (0-1)
        confidence_label  str     "High", "Moderate", etc.
        explanation       list    plain-language bullet points
        feature_importance_reg  list of {feature, importance}
        feature_importance_clf  list of {feature, importance}
        model_reg_name    str
        model_clf_name    str
        model_reg_metrics dict
        model_clf_metrics dict
    """
    _load_models()

    # Build feature vector -- DataFrame preserves column names the scaler
    # was fitted with, preventing a spurious UserWarning.
    import pandas as pd
    x_raw    = pd.DataFrame([[inputs[f] for f in FEATURE_COLS]],
                             columns=FEATURE_COLS)
    x_scaled = _scaler.transform(x_raw)

    # Regression
    predicted_g3 = float(np.clip(_regressor.predict(x_scaled)[0], 0, 20))

    # Classification
    pass_prob    = float(_classifier.predict_proba(x_scaled)[0][1])
    fail_prob    = 1.0 - pass_prob
    pass_fail    = int(pass_prob >= 0.5)

    return {
        "predicted_g3":      round(predicted_g3, 2),
        "predicted_g3_int":  int(round(predicted_g3)),
        "grade_label":       _grade_label(predicted_g3),
        "pass_fail":         pass_fail,
        "pass_fail_label":   "Pass" if pass_fail else "Fail",
        "pass_probability":  round(pass_prob, 4),
        "fail_probability":  round(fail_prob, 4),
        "confidence_label":  _confidence_label(max(pass_prob, fail_prob)),
        "explanation":       _build_explanation(inputs, predicted_g3, pass_prob),
        "feature_importance_reg": _get_feature_importance_reg(),
        "feature_importance_clf": _get_feature_importance_clf(),
        "model_reg_name":    _metadata["best_regressor"]["name"],
        "model_clf_name":    _metadata["best_classifier"]["name"],
        "model_reg_metrics": _metadata["best_regressor"]["metrics"],
        "model_clf_metrics": _metadata["best_classifier"]["metrics"],
    }


def get_metadata() -> dict:
    """Return model metadata (names + all metrics) for the /api/metadata route."""
    _load_models()
    return _metadata

