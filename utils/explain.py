"""explain.py - SHAP-based model explainability."""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def compute_shap_values(pipeline, X_test: pd.DataFrame):
    """
    Compute SHAP values using LinearExplainer (fast, works with LogReg).
    Returns explainer + shap_values array.
    """
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    X_scaled = scaler.transform(X_test)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_test.columns)

    explainer = shap.LinearExplainer(clf, X_scaled_df, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled_df)

    # For binary classification, shap_values may be list → take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values, X_scaled_df


def plot_shap_bar(shap_values: np.ndarray, feature_names: list) -> plt.Figure:
    """Mean absolute SHAP feature importance bar chart."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    top_n = min(10, len(feature_names))
    idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    bars = ax.barh(
        [feature_names[i] for i in idx][::-1],
        mean_shap[idx][::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(shap_values: np.ndarray, X_scaled_df: pd.DataFrame) -> plt.Figure:
    """SHAP summary beeswarm plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        X_scaled_df,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    return fig


def plot_waterfall_single(shap_values: np.ndarray, X_scaled_df: pd.DataFrame,
                           feature_names: list, sample_idx: int = 0) -> plt.Figure:
    """Waterfall chart for single prediction explanation."""
    sv = shap_values[sample_idx]
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    top_n = min(8, len(feature_names))
    idx = sorted_idx[:top_n]

    names = [feature_names[i] for i in idx]
    vals = sv[idx]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=10)
    ax.set_title(f"Single Prediction Explanation (Sample #{sample_idx})", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def get_top_features(shap_values: np.ndarray, feature_names: list, top_n: int = 5) -> list:
    """Return top-n features sorted by mean |SHAP|."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1][:top_n]
    return [(feature_names[i], round(float(mean_shap[i]), 4)) for i in sorted_idx]
