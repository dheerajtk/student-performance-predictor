"""
load_data.py — Dataset loading and feature engineering for Student Performance Predictor.

Dataset: UCI Student Performance (Math), student-mat.csv
Source : https://archive.ics.uci.edu/dataset/320/student+performance

Columns we use
--------------
studytime  : weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
absences   : number of school absences (0–93)
G1         : first period grade (0–20)  ← our "previous marks" feature
failures   : number of past class failures (0–4)
G3         : final grade (0–20)         ← TARGET

Derived column
--------------
attendance_pct : 100 - absences * 2.5, clipped to [0, 100].
    Rationale: The UCI dataset has no attendance column. Since 'absences'
    is the direct inverse of attendance, we derive a 0-100 percentage
    using this linear mapping. The user-supplied 'Attendance %' input
    is stored in this column so the model can use it consistently.
    This derivation is documented in the README and shown in the UI.

Target leakage note
-------------------
G2 (second period grade) is excluded as a feature because it correlates
~0.85 with G3 and would represent near-future knowledge unavailable at
the time of prediction. Only G1 (first period, "previous marks") is kept.
"""

import os
import pandas as pd
import numpy as np


# Columns expected in the raw CSV
_REQUIRED_COLS = {"studytime", "absences", "G1", "G3", "failures"}

# Studytime label map (for display / validation)
STUDYTIME_LABELS = {
    1: "<2 hours/week",
    2: "2–5 hours/week",
    3: "5–10 hours/week",
    4: ">10 hours/week",
}


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and perform a basic sanity-check."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Download student-mat.csv from:\n"
            "  https://archive.ics.uci.edu/dataset/320/student+performance\n"
            "and place it in the data/ folder."
        )

    # UCI file uses semicolons as delimiters
    df = pd.read_csv(csv_path, sep=";")

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"The CSV is missing expected columns: {missing}\n"
            "Make sure you are using 'student-mat.csv', not 'student-por.csv'."
        )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and engineer the five user-facing features plus the target.

    Returns a clean DataFrame with columns:
        studytime, attendance_pct, G1, failures, absences, G3, pass_fail
    """
    df = df.copy()

    # --- Derived attendance percentage ---
    # absences=0  → attendance_pct=100 %
    # absences=40 → attendance_pct=0   % (40 × 2.5 = 100)
    # absences>40 → clipped to 0 %
    df["attendance_pct"] = (100 - df["absences"] * 2.5).clip(lower=0, upper=100)

    # --- Binary pass/fail target (pass if G3 >= 10) ---
    df["pass_fail"] = (df["G3"] >= 10).astype(int)  # 1 = Pass, 0 = Fail

    # Keep only the columns we need
    keep = ["studytime", "attendance_pct", "G1", "failures", "absences", "G3", "pass_fail"]
    df = df[keep].copy()

    # Drop rows where the target is 0 AND G1 is also 0
    # (these are typically students who never submitted — 32 rows in the Math dataset)
    df = df[~((df["G3"] == 0) & (df["G1"] == 0))].reset_index(drop=True)

    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Convenience wrapper: load + engineer in one call."""
    raw = load_raw(csv_path)
    return engineer_features(raw)


# ---------------------------------------------------------------------------
# Quick verification (run this file directly: python -m src.load_data)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/student-mat.csv"
    df = load_dataset(path)

    print("=" * 55)
    print("  Student Performance Dataset — Loaded Successfully")
    print("=" * 55)
    print(f"  Rows   : {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print()
    print("--- First 5 rows ---")
    print(df.head().to_string(index=False))
    print()
    print("--- Descriptive statistics ---")
    print(df.describe().round(2).to_string())
    print()
    print("--- Pass / Fail distribution ---")
    counts = df["pass_fail"].value_counts().rename({1: "Pass", 0: "Fail"})
    print(counts.to_string())
    print()
    print("--- Missing values ---")
    print(df.isnull().sum().to_string())
    print()
    print("--- Study-time distribution ---")
    st = df["studytime"].value_counts().sort_index()
    for k, v in st.items():
        print(f"  {k} ({STUDYTIME_LABELS[k]}): {v} students")
