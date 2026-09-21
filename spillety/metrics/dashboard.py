import numpy as np
from sklearn.metrics import average_precision_score

from spillety.models.calibration import brier_score, ece_score, reliability_stats

__all__ = ["brier_score", "ece_bootstrap_ci", "ece_score", "pr_auc_score", "reliability_table"]


def ece_bootstrap_ci(y_true, y_prob, n_bins=10, n_boot=1000, random_state=72) -> dict:
    """
    ## ECE with bootstrap 95% CI (percentile method, §7.6.4)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Bin count M.
    n_boot : int
        Bootstrap replications.
    random_state : int
        Seed for resampling.

    Returns
    ----------
    dict
        `{"ece": float, "ci_lower": float, "ci_upper": float, "n_boot": int}`.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    ece_vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ece_vals[b] = ece_score(y_true[idx], y_prob[idx], n_bins=n_bins)
    ece_point = ece_score(y_true, y_prob, n_bins=n_bins)
    return {
        "ece": float(ece_point),
        "ci_lower": float(np.percentile(ece_vals, 2.5)),
        "ci_upper": float(np.percentile(ece_vals, 97.5)),
        "n_boot": n_boot,
    }


def pr_auc_score(y_true, y_score):
    """
    ## Area under the precision-recall curve (§7.7.1)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score : np.ndarray
        Risk scores.

    Returns
    ----------
    float
        PR-AUC, 0.0 when the labels are single-class.
    """
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, np.asarray(y_score, dtype=float)))


def reliability_table(y_true, y_prob, n_bins=10):
    """
    ## Per-bin accuracy vs confidence rows for the reliability diagram (§7.6.3)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Bin count M.

    Returns
    ----------
    list[dict]
        One row per bin with `bin`, `lo`, `hi`, `acc`, `conf`, `count`.
    """
    stats = reliability_stats(y_true, y_prob, n_bins=n_bins)
    return [
        {
            "bin": m,
            "lo": stats["lo"][m],
            "hi": stats["hi"][m],
            "acc": stats["acc"][m],
            "conf": stats["conf"][m],
            "count": stats["count"][m],
        }
        for m in range(n_bins)
    ]
