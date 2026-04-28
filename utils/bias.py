"""bias.py - Fairness metrics via Fairlearn."""

import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score


def compute_fairness_metrics(y_test: pd.Series, y_pred: np.ndarray, sensitive: pd.Series):
    """
    Compute demographic parity + equalized odds metrics.
    Returns dict of scalar metrics + MetricFrame.
    """
    # Align index
    sensitive_aligned = sensitive.loc[y_test.index]

    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_aligned)
    dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive_aligned)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_aligned)
    eor = equalized_odds_ratio(y_test, y_pred, sensitive_features=sensitive_aligned)

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_aligned,
    )

    # Per-group breakdown
    by_group = mf.by_group.copy()

    overall_metrics = {
        "demographic_parity_difference": dpd,
        "demographic_parity_ratio": dpr,
        "equalized_odds_difference": eod,
        "equalized_odds_ratio": eor,
        "overall_accuracy": accuracy_score(y_test, y_pred),
    }

    return overall_metrics, by_group, mf


def bias_verdict(dpd: float, eod: float) -> dict:
    """
    Return plain-language verdict + severity.
    Threshold: |dpd| or |eod| > 0.1 = biased.
    """
    dpd_abs = abs(dpd)
    eod_abs = abs(eod)

    if dpd_abs < 0.05 and eod_abs < 0.05:
        level = "LOW"
        color = "green"
        msg = (
            "Model shows minimal bias. Predictions are fairly distributed "
            "across sensitive groups. Both demographic parity and equalized "
            "odds differences are below the 5% threshold."
        )
    elif dpd_abs < 0.10 and eod_abs < 0.10:
        level = "MODERATE"
        color = "orange"
        msg = (
            "Model shows moderate bias. Some disparity exists between groups. "
            "Consider rebalancing training data or applying fairness constraints "
            "before deploying in high-stakes decisions."
        )
    else:
        level = "HIGH"
        color = "red"
        msg = (
            "Model shows significant bias. Predictions differ substantially "
            "across sensitive groups. Strong action recommended: audit training "
            "data, apply fairness-aware algorithms, or use post-processing "
            "methods to mitigate disparate impact."
        )

    return {"level": level, "color": color, "message": msg, "dpd": dpd_abs, "eod": eod_abs}
