"""
FairLens AI – Transparent Intelligence
Bias detection + model explainability dashboard.
Run: streamlit run app.py
"""

import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from utils.model import preprocess_data, train_model
from utils.bias import compute_fairness_metrics, bias_verdict
from utils.explain import (
    compute_shap_values,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_waterfall_single,
    get_top_features,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairLens AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (light, premium) ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
    .main { background: #f8f9fc; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #6366f1;
        margin-bottom: 12px;
    }
    .metric-card h3 { margin: 0 0 4px 0; font-size: 13px; color: #6b7280; font-weight: 500; }
    .metric-card p  { margin: 0; font-size: 26px; font-weight: 700; color: #111827; }

    /* Bias badge */
    .badge-low    { background:#d1fae5; color:#065f46; padding:6px 14px; border-radius:20px; font-weight:700; font-size:14px; }
    .badge-moderate { background:#fef3c7; color:#92400e; padding:6px 14px; border-radius:20px; font-weight:700; font-size:14px; }
    .badge-high   { background:#fee2e2; color:#991b1b; padding:6px 14px; border-radius:20px; font-weight:700; font-size:14px; }

    /* Section headers */
    .section-header {
        font-size: 18px; font-weight: 700; color: #1f2937;
        border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin: 24px 0 16px 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1e1b4b; }
    [data-testid="stSidebar"] * { color: #e0e7ff !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #a5b4fc !important; font-size: 13px; }

    /* Insight box */
    .insight-box {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 10px; padding: 16px 20px; margin-top: 12px;
        font-size: 14px; color: #1e40af; line-height: 1.6;
    }

    /* Progress bar override */
    .stProgress > div > div { background: #6366f1; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ FairLens AI")
    st.markdown("*Transparent Intelligence*")
    st.divider()

    st.markdown("### 📂 Data Upload")
    use_sample = st.checkbox("Use built-in sample dataset", value=True)

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.divider()
    st.markdown("### ⚙️ Configuration")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload CSV or use sample
    2. Select target + sensitive column
    3. Click Run Analysis
    4. Review bias scores & SHAP explanations
    """)
    st.markdown("---")
    st.caption("Built with Fairlearn + SHAP + scikit-learn")


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
     border-radius:16px;padding:32px 36px;margin-bottom:28px;color:white;">
  <h1 style="margin:0;font-size:32px;">⚖️ FairLens AI</h1>
  <p style="margin:8px 0 0 0;font-size:16px;opacity:0.9;">
    Detect bias in ML models · Explain predictions with SHAP · Build trustworthy AI
  </p>
</div>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_sample():
    return pd.read_csv("sample_data.csv")


df = None
if use_sample:
    df = load_sample()
    st.info("📋 Using **sample dataset** (adult income, 100 rows). Upload your own CSV to analyze custom data.")
elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    st.warning("👈 Enable sample dataset or upload a CSV to begin.")
    st.stop()


# ── Column selection ──────────────────────────────────────────────────────────
col_preview, col_config = st.columns([3, 1])
with col_preview:
    with st.expander("📊 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

with col_config:
    st.markdown("### 🎯 Column Setup")
    target_col = st.selectbox("Target column (what to predict)", df.columns.tolist(),
                               index=len(df.columns) - 1)
    sensitive_col = st.selectbox("Sensitive feature (e.g. gender, race)",
                                  [c for c in df.columns if c != target_col],
                                  index=0)
    n_unique = df[sensitive_col].nunique() if sensitive_col else 0
    if n_unique > 10:
        st.warning(f"26a0Fe0f '{sensitive_col}' has {n_unique} unique values. Choose a categorical column like gender or education.")
    sample_idx = st.number_input("Sample index for waterfall", min_value=0, max_value=20, value=0)


# ── Run analysis ──────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div class="insight-box">
    ✨ <b>Ready to analyze.</b> Configure columns above, then click <b>🚀 Run Analysis</b> in the sidebar.
    The system will train a Logistic Regression model, measure fairness using Fairlearn,
    and explain predictions with SHAP — all in seconds.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Pipeline ──────────────────────────────────────────────────────────────────
with st.spinner("🔄 Preprocessing data..."):
    try:
        X, y, sensitive, feature_names, encoders = preprocess_data(df, target_col, sensitive_col)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

progress = st.progress(0, text="Starting pipeline...")

with st.spinner("🧠 Training model..."):
    try:
        pipeline, y_pred, train_metrics = train_model(X, y)
        progress.progress(33, text="Model trained ✓")
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

with st.spinner("⚖️ Computing fairness metrics..."):
    try:
        X_test = train_metrics["X_test"]
        y_test = train_metrics["y_test"]
        overall_metrics, by_group, mf = compute_fairness_metrics(y_test, y_pred, sensitive)
        verdict = bias_verdict(
            overall_metrics["demographic_parity_difference"],
            overall_metrics["equalized_odds_difference"]
        )
        progress.progress(66, text="Fairness metrics computed ✓")
    except Exception as e:
        st.error(f"Bias detection failed: {e}")
        st.stop()

with st.spinner("🔍 Computing SHAP explanations..."):
    try:
        explainer, shap_values, X_scaled_df = compute_shap_values(pipeline, X_test)
        top_features = get_top_features(shap_values, feature_names)
        progress.progress(100, text="Complete ✓")
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        st.stop()

progress.empty()
st.success("✅ Analysis complete!")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 – MODEL PERFORMANCE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
accuracy = overall_metrics["overall_accuracy"]
precision = train_metrics["report"].get("weighted avg", {}).get("precision", 0)
recall = train_metrics["report"].get("weighted avg", {}).get("recall", 0)
f1 = train_metrics["report"].get("weighted avg", {}).get("f1-score", 0)

for col, label, val, color in zip(
    [c1, c2, c3, c4],
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [accuracy, precision, recall, f1],
    ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981"],
):
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
      <h3>{label}</h3>
      <p>{val:.1%}</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 – BIAS DETECTION
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⚖️ Bias Detection</div>', unsafe_allow_html=True)

badge_class = f"badge-{verdict['level'].lower()}"
st.markdown(
    f"**Overall Bias Level:** &nbsp;<span class='{badge_class}'>{verdict['level']} BIAS</span>",
    unsafe_allow_html=True
)
st.markdown(f"""<div class="insight-box">💡 {verdict['message']}</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)
metrics_display = [
    ("Demographic Parity Diff", overall_metrics["demographic_parity_difference"], "#e74c3c"),
    ("Demographic Parity Ratio", overall_metrics["demographic_parity_ratio"], "#f39c12"),
    ("Equalized Odds Diff", overall_metrics["equalized_odds_difference"], "#e74c3c"),
    ("Equalized Odds Ratio", overall_metrics["equalized_odds_ratio"], "#f39c12"),
]
for col, (label, val, color) in zip([b1, b2, b3, b4], metrics_display):
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
      <h3>{label}</h3>
      <p>{val:.4f}</p>
    </div>
    """, unsafe_allow_html=True)


# Per-group breakdown charts
groups = by_group.index.tolist()
n_groups = len(groups)

# Warn if sensitive column has too many unique values (e.g. age used as-is)
if n_groups > 10:
    st.warning(
        f"⚠️ **'{sensitive_col}'** has **{n_groups} unique values** — charts may be crowded. "
        f"For cleaner results, pick a categorical column (e.g. gender, race, education) "
        f"instead of a continuous one like age."
    )

st.markdown("**Per-Group Breakdown**")
chart_col1, chart_col2 = st.columns(2)

# Shared palette — cycle if more groups than colors
PALETTE = ["#6366f1","#8b5cf6","#06b6d4","#10b981","#f59e0b",
           "#ef4444","#ec4899","#84cc16","#f97316","#14b8a6"]
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(n_groups)]

with chart_col1:
    acc_vals = by_group["accuracy"].values
    fig_h = max(3.5, n_groups * 0.35)   # taller when many groups
    fig, ax = plt.subplots(figsize=(6, fig_h))
    y_pos = range(n_groups)
    bars = ax.barh([str(g) for g in groups], acc_vals,
                   color=bar_colors, edgecolor="white", linewidth=1)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(f"Accuracy by {sensitive_col}", fontsize=12, fontweight="bold")
    ax.axvline(accuracy, color="gray", linestyle="--", linewidth=1,
               label=f"Overall: {accuracy:.2f}")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, acc_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    sel_rates = by_group["selection_rate"].values
    if n_groups <= 6:
        # Pie only looks good with few slices
        fig, ax = plt.subplots(figsize=(5, fig_h))
        ax.pie(
            sel_rates,
            labels=[str(g) for g in groups],
            autopct="%1.1f%%",
            colors=bar_colors,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title(f"Selection Rate by {sensitive_col}", fontsize=12, fontweight="bold")
    else:
        # Horizontal bar — readable at any group count
        fig, ax = plt.subplots(figsize=(6, fig_h))
        ax.barh([str(g) for g in groups], sel_rates,
                color=bar_colors, edgecolor="white", linewidth=1)
        ax.set_xlim(0, max(sel_rates.max() * 1.2, 0.1))
        ax.set_xlabel("Selection Rate", fontsize=11)
        ax.set_title(f"Selection Rate by {sensitive_col}", fontsize=12, fontweight="bold")
        for i, v in enumerate(sel_rates):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# FPR / FNR by group
st.markdown("**False Positive & Negative Rates by Group**")
fig, axes = plt.subplots(1, 2, figsize=(11, fig_h))
for ax, metric, title, color in zip(
    axes,
    ["false_positive_rate", "false_negative_rate"],
    ["False Positive Rate", "False Negative Rate"],
    ["#ef4444", "#f97316"],
):
    vals = by_group[metric].values
    ax.barh([str(g) for g in groups], vals, color=color, alpha=0.85, edgecolor="white")
    ax.set_xlim(0, 1.15)
    ax.set_title(f"{title} by {sensitive_col}", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Raw group table
with st.expander("📋 Full Per-Group Metrics Table"):
    st.dataframe(by_group.style.format("{:.4f}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 – SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔍 Model Explainability (SHAP)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
🧠 <b>How to read SHAP charts:</b> SHAP (SHapley Additive exPlanations) shows how much each feature
pushes the prediction higher (positive) or lower (negative). Features at the top have the greatest
impact on model decisions.
</div>
""", unsafe_allow_html=True)

# Top features text
st.markdown("**Top Influential Features:**")
feat_cols = st.columns(min(5, len(top_features)))
for col, (feat, val) in zip(feat_cols, top_features):
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:#10b981">
      <h3>{feat}</h3>
      <p style="font-size:18px;">{val:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

shap_col1, shap_col2 = st.columns(2)

with shap_col1:
    st.markdown("**Feature Importance (Mean |SHAP|)**")
    fig_bar = plot_shap_bar(shap_values, feature_names)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

with shap_col2:
    st.markdown("**SHAP Summary Beeswarm**")
    try:
        fig_bee = plot_shap_beeswarm(shap_values, X_scaled_df)
        st.pyplot(fig_bee)
        plt.close(fig_bee)
    except Exception:
        st.info("Beeswarm plot unavailable for this dataset size.")

st.markdown(f"**Single Prediction Waterfall (Sample #{int(sample_idx)})**")
try:
    n_samples = len(shap_values)
    idx_safe = min(int(sample_idx), n_samples - 1)
    fig_wf = plot_waterfall_single(shap_values, X_scaled_df, feature_names, idx_safe)
    st.pyplot(fig_wf)
    plt.close(fig_wf)
except Exception as e:
    st.warning(f"Waterfall plot error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 – FULL REPORT DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📥 Export</div>', unsafe_allow_html=True)

report_lines = [
    "# FairLens AI – Analysis Report",
    "",
    "## Model Performance",
    f"- Accuracy:  {accuracy:.4f}",
    f"- Precision: {precision:.4f}",
    f"- Recall:    {recall:.4f}",
    f"- F1-Score:  {f1:.4f}",
    "",
    "## Bias Detection",
    f"- Bias Level: {verdict['level']}",
    f"- Demographic Parity Difference: {overall_metrics['demographic_parity_difference']:.4f}",
    f"- Demographic Parity Ratio:      {overall_metrics['demographic_parity_ratio']:.4f}",
    f"- Equalized Odds Difference:     {overall_metrics['equalized_odds_difference']:.4f}",
    f"- Equalized Odds Ratio:          {overall_metrics['equalized_odds_ratio']:.4f}",
    "",
    "## Verdict",
    verdict["message"],
    "",
    "## Top Features (SHAP)",
]
for feat, val in top_features:
    report_lines.append(f"- {feat}: {val:.4f}")

report_lines += ["", "## Per-Group Metrics", by_group.to_string()]

report_text = "\n".join(report_lines)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "📄 Download Report (.txt)",
        data=report_text,
        file_name="fairlens_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
with dl2:
    csv_bytes = by_group.to_csv().encode()
    st.download_button(
        "📊 Download Group Metrics (.csv)",
        data=csv_bytes,
        file_name="fairlens_group_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.caption("⚖️ FairLens AI · Built with Streamlit, Fairlearn, SHAP, scikit-learn")
