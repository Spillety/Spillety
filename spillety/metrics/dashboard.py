import numpy as np
from sklearn.metrics import average_precision_score

from spillety.models.calibration import brier_score, ece_score, reliability_stats

__all__ = ["brier_score", "ece_score", "pr_auc_score", "reliability_table"]


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
