"""
eda.py — Exploratory Data Analysis for Student Performance Predictor.

Run:
    python -m src.eda

Outputs (saved to models/plots/):
    01_target_distribution.png   — G3 histogram + KDE
    02_passfail_pie.png          — Pass / Fail pie chart
    03_feature_distributions.png — 5-panel subplot for all features
    04_correlation_heatmap.png   — Pearson correlation matrix
    05_studytime_vs_g3.png       — Boxplot: study time vs final grade
    06_attendance_vs_g3.png      — Scatter: attendance % vs final grade
    07_failures_vs_g3.png        — Boxplot: previous failures vs final grade
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works in all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(_ROOT, "data", "student-mat.csv")
PLOTS_DIR  = os.path.join(_ROOT, "models", "plots")

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE    = sns.color_palette("muted")
ACCENT     = "#3b82d4"
PASS_COLOR = "#22c55e"
FAIL_COLOR = "#ef4444"
FIGSIZE_SM = (7, 4.5)
FIGSIZE_MD = (10, 6)
FIGSIZE_LG = (14, 8)

STUDYTIME_LABELS = {
    1: "<2 h",
    2: "2–5 h",
    3: "5–10 h",
    4: ">10 h",
}


def _save(fig: plt.Figure, name: str) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK Saved -> {os.path.relpath(path, _ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# Plot 1 — Target distribution (G3 histogram + KDE)
# --------------------------------------------------------------------------- #
def plot_target_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SM)

    sns.histplot(df["G3"], bins=21, color=ACCENT, edgecolor="white",
                 kde=True, line_kws={"linewidth": 2}, ax=ax)

    # Pass threshold line
    ax.axvline(10, color=FAIL_COLOR, linestyle="--", linewidth=1.8,
               label="Pass threshold (G3 = 10)")
    ax.set_xlabel("Final Grade (G3)", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_title("Distribution of Final Grades (G3)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    # Annotation: mean & median
    mean_g3   = df["G3"].mean()
    median_g3 = df["G3"].median()
    ax.axvline(mean_g3,   color="#f59e0b", linestyle="-.",
               linewidth=1.5, label=f"Mean {mean_g3:.1f}")
    ax.axvline(median_g3, color="#8b5cf6", linestyle=":",
               linewidth=1.5, label=f"Median {median_g3:.1f}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return _save(fig, "01_target_distribution.png")


# --------------------------------------------------------------------------- #
# Plot 2 — Pass / Fail pie chart
# --------------------------------------------------------------------------- #
def plot_passfail_pie(df: pd.DataFrame) -> str:
    counts = df["pass_fail"].value_counts().sort_index()
    labels = ["Fail (G3 <  10)", "Pass (G3 >= 10)"]
    colors = [FAIL_COLOR, PASS_COLOR]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")

    ax.set_title("Pass / Fail Distribution", fontsize=14, fontweight="bold", pad=18)
    fig.tight_layout()
    return _save(fig, "02_passfail_pie.png")


# --------------------------------------------------------------------------- #
# Plot 3 — Feature distributions (5-panel)
# --------------------------------------------------------------------------- #
def plot_feature_distributions(df: pd.DataFrame) -> str:
    features = [
        ("studytime",      "Study Time (1–4 scale)", True),
        ("attendance_pct", "Attendance %",            False),
        ("G1",             "Previous Marks (G1)",     False),
        ("failures",       "Previous Failures",       True),
        ("absences",       "Absences",                False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for idx, (col, label, is_discrete) in enumerate(features):
        ax = axes_flat[idx]
        if is_discrete:
            order = sorted(df[col].unique())
            sns.countplot(x=df[col], order=order, color=ACCENT, edgecolor="white", ax=ax)
        else:
            sns.histplot(df[col], bins=20, color=ACCENT, edgecolor="white",
                         kde=True, line_kws={"linewidth": 1.8}, ax=ax)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Distribution of {label}", fontsize=11, fontweight="bold")

    # Relabel study-time x-axis ticks
    st_ax = axes_flat[0]
    st_ax.set_xticks([0, 1, 2, 3])
    st_ax.set_xticklabels(["<2h", "2–5h", "5–10h", ">10h"], fontsize=9)

    # Hide the unused 6th panel
    axes_flat[5].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "03_feature_distributions.png")


# --------------------------------------------------------------------------- #
# Plot 4 — Correlation heatmap
# --------------------------------------------------------------------------- #
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    cols = ["studytime", "attendance_pct", "G1", "failures", "absences", "G3"]
    corr = df[cols].corr(method="pearson")

    # Rename for display
    rename = {
        "studytime":      "Study\nTime",
        "attendance_pct": "Attendance\n%",
        "G1":             "Prev.\nMarks (G1)",
        "failures":       "Prev.\nFailures",
        "absences":       "Absences",
        "G3":             "Final\nGrade (G3)",
    }
    corr.rename(index=rename, columns=rename, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # upper triangle

    sns.heatmap(
        corr, annot=True, fmt=".2f", mask=mask,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 10},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson Correlation Matrix\n(lower triangle)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Plot 5 — Study time vs G3 (boxplot)
# --------------------------------------------------------------------------- #
def plot_studytime_vs_g3(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SM)

    df_plot = df.copy()
    df_plot["studytime_label"] = df_plot["studytime"].map(STUDYTIME_LABELS)
    order = ["<2 h", "2–5 h", "5–10 h", ">10 h"]

    sns.boxplot(
        data=df_plot, x="studytime_label", y="G3",
        order=order, palette="Blues",
        linewidth=1.5, width=0.5,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
        ax=ax,
    )
    ax.axhline(10, color=FAIL_COLOR, linestyle="--", linewidth=1.5,
               label="Pass threshold (10)")
    ax.set_xlabel("Weekly Study Time", fontsize=12)
    ax.set_ylabel("Final Grade (G3)", fontsize=12)
    ax.set_title("Study Time vs Final Grade", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "05_studytime_vs_g3.png")


# --------------------------------------------------------------------------- #
# Plot 6 — Attendance % vs G3 (scatter + regression line)
# --------------------------------------------------------------------------- #
def plot_attendance_vs_g3(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SM)

    # Colour points by pass/fail
    colors = df["pass_fail"].map({1: PASS_COLOR, 0: FAIL_COLOR})
    ax.scatter(df["attendance_pct"], df["G3"],
               c=colors, alpha=0.45, s=35, edgecolors="none")

    # Regression line
    m, b = np.polyfit(df["attendance_pct"], df["G3"], 1)
    x_line = np.linspace(df["attendance_pct"].min(), df["attendance_pct"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="#1e40af", linewidth=2, label=f"Trend (slope={m:.2f})")

    ax.axhline(10, color=FAIL_COLOR, linestyle="--", linewidth=1.5,
               label="Pass threshold (10)", alpha=0.8)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PASS_COLOR, markersize=8, label="Pass"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FAIL_COLOR,  markersize=8, label="Fail"),
        Line2D([0], [0], color="#1e40af", linewidth=2, label=f"Trend (slope={m:.2f})"),
        Line2D([0], [0], color=FAIL_COLOR, linestyle="--", linewidth=1.5, label="Pass threshold"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_xlabel("Attendance % (derived from absences)", fontsize=12)
    ax.set_ylabel("Final Grade (G3)", fontsize=12)
    ax.set_title("Attendance % vs Final Grade", fontsize=14, fontweight="bold")
    ax.set_xlim(-2, 102)
    fig.tight_layout()
    return _save(fig, "06_attendance_vs_g3.png")


# --------------------------------------------------------------------------- #
# Plot 7 — Previous failures vs G3 (boxplot)
# --------------------------------------------------------------------------- #
def plot_failures_vs_g3(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SM)

    order = sorted(df["failures"].unique())
    sns.boxplot(
        data=df, x="failures", y="G3",
        order=order, palette="Oranges",
        linewidth=1.5, width=0.5,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
        ax=ax,
    )
    ax.axhline(10, color=FAIL_COLOR, linestyle="--", linewidth=1.5,
               label="Pass threshold (10)")
    ax.set_xlabel("Number of Previous Failures", fontsize=12)
    ax.set_ylabel("Final Grade (G3)", fontsize=12)
    ax.set_title("Previous Failures vs Final Grade", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "07_failures_vs_g3.png")


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #
def print_summary(df: pd.DataFrame) -> None:
    sep = "=" * 58
    print(sep)
    print("  EDA Summary - Student Performance Dataset")
    print(sep)
    print(f"  Total students     : {len(df)}")
    print(f"  Pass (G3 >= 10)     : {df['pass_fail'].sum()} "
          f"({df['pass_fail'].mean()*100:.1f}%)")
    print(f"  Fail (G3 <  10)     : {(df['pass_fail']==0).sum()} "
          f"({(df['pass_fail']==0).mean()*100:.1f}%)")
    print()
    print("  Feature statistics:")
    stats = df[["studytime","attendance_pct","G1","failures","absences","G3"]].describe()
    print(stats.round(2).to_string())
    print()

    print("  Pearson correlations with G3 (final grade):")
    corr_g3 = df[["studytime","attendance_pct","G1","failures","absences","G3"]]\
                  .corr()["G3"].drop("G3").sort_values(key=abs, ascending=False)
    for feat, val in corr_g3.items():
        bar = "#" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"    {feat:<18} {sign}{abs(val):.3f}  {bar}")
    print()

    print("  Key insights:")
    g1_corr = df["G1"].corr(df["G3"])
    print(f"    * G1 <-> G3 correlation  : {g1_corr:.3f}  (strong predictor)")
    fail_corr = df["failures"].corr(df["G3"])
    print(f"    * failures <-> G3        : {fail_corr:.3f}  (negative, expected)")
    att_corr = df["attendance_pct"].corr(df["G3"])
    print(f"    * attendance_pct <-> G3  : {att_corr:.3f}  (positive, expected)")
    abs_corr = df["absences"].corr(df["G3"])
    print(f"    * absences <-> G3        : {abs_corr:.3f}  (negative, expected)")
    st_corr = df["studytime"].corr(df["G3"])
    print(f"    * studytime <-> G3       : {st_corr:.3f}  (positive, expected)")
    print()
    print("  NOTE: G2 excluded to prevent target leakage (corr~0.90 with G3).")
    print(sep)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_eda(csv_path: str = DATA_PATH) -> None:
    # Import here to avoid circular at module level
    from src.load_data import load_dataset
    df = load_dataset(csv_path)

    print_summary(df)

    print("\nGenerating plots...")
    plot_target_distribution(df)
    plot_passfail_pie(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_studytime_vs_g3(df)
    plot_attendance_vs_g3(df)
    plot_failures_vs_g3(df)

    print(f"\nAll plots saved to: {os.path.relpath(PLOTS_DIR, _ROOT)}/")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    run_eda(path)


