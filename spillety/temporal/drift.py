import numpy as np
from scipy.stats import ks_2samp

__all__ = ["cohen_d", "is_drift", "ks_stats", "median_distance", "tau_d"]


def median_distance(z_new, z_old) -> float:
    """
    ## Median Euclidean distance from a new anchor to old anchors (§8.3.1)

    Parameters
    ----------
    z_new : np.ndarray
        Single embedding vector, shape (d,).
    z_old : np.ndarray
        Old anchor embeddings, shape (n, d).

    Returns
    ----------
    float
        Median of ||z_new − z_old||_2 over old anchors.
    """
    diff = np.asarray(z_old, dtype=float) - np.asarray(z_new, dtype=float)
    return float(np.median(np.linalg.norm(diff, axis=1)))


def tau_d(history, strict: bool = False) -> float:
    """
    ## Drift threshold as a quantile of historical median distances (§8.3.2)

    Parameters
    ----------
    history : np.ndarray
        Historical d_median values.
    strict : bool
        False → Q0.95 default; True → Q0.99 when retrain is expensive.

    Returns
    ----------
    float
        Quantile threshold τd.
    """
    return float(np.quantile(np.asarray(history, dtype=float), 0.99 if strict else 0.95))


def cohen_d(a, b) -> float:
    """
    ## Standardized effect size between two samples (§8.3.4)

    Parameters
    ----------
    a, b : np.ndarray
        Train and test samples of one feature.

    Returns
    ----------
    float
        (mean(a) − mean(b)) / pooled std; 0.0 on zero variance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.std(a, ddof=1) if len(a) > 1 else 0.0
    sb = np.std(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / max(1, len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled != 0 else 0.0


def ks_stats(a, b) -> tuple:
    """
    ## Two-sample KS statistic and p-value (§8.3.4)

    Parameters
    ----------
    a, b : np.ndarray
        Train and test samples of one feature.

    Returns
    ----------
    tuple
        (D, p) from the two-sample Kolmogorov-Smirnov test.
    """
    res = ks_2samp(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(res.statistic), float(res.pvalue)


def is_drift(p: float, d: float, cohen: float) -> bool:
    """
    ## Joint drift rule: significance plus effect size (§8.3.4)

    Parameters
    ----------
    p : float
        KS p-value.
    d : float
        KS statistic D.
    cohen : float
        Cohen d effect size.

    Returns
    ----------
    bool
        True iff p < 0.05 and D > 0.10 and |d_Cohen| > 0.30.
    """
    return bool(p < 0.05 and d > 0.10 and abs(cohen) > 0.30)
