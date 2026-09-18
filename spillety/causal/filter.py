import numpy as np

from spillety.causal.validate import partial_corr_pvalue


def _sensitivity(r: float) -> tuple[float, float]:
    rr = 1.0 + 4.0 * abs(float(r))
    e_value = rr + np.sqrt(max(rr * (rr - 1.0), 0.0))
    gamma = 1.0 + 2.0 * abs(float(r))
    return float(e_value), float(gamma)


def causal_filter(
    anchors: np.ndarray,
    wallet: np.ndarray,
    confounders: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
) -> dict:
    """
    ## Drop anchors with no causal path beyond confounders (§6.4.2)

    Parameters
    ----------
    anchors : np.ndarray
        Per-anchor samples, shape (n, k).
    wallet : np.ndarray
        Wallet samples, shape (n,).
    confounders : np.ndarray | None
        Observed confounders, shape (n, p).
    alpha : float
        Exclusion level: anchor dropped when a⊥w|C not rejected.

    Returns
    ----------
    dict
        `mask` (k,) kept anchors, `pass_rate`, per-anchor `p_values`,
        `partial_corr`, plus `e_value`/`gamma` (k,) feeding §7.3.6.
    """
    # ponytail: asymptotic Fisher-z gate; bootstrap-stable variant lives
    # in validate.ci_test and stays opt-in for cost (§6.3.1 follow-up).
    a = np.asarray(anchors, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    w = np.asarray(wallet, dtype=float).ravel()
    k = a.shape[1]
    mask = np.empty(k, dtype=bool)
    p_values = np.empty(k)
    partial = np.empty(k)
    e_value = np.empty(k)
    gamma = np.empty(k)
    for j in range(k):
        r, p = partial_corr_pvalue(a[:, j], w, confounders)
        partial[j], p_values[j] = r, p
        mask[j] = bool(p < alpha)
        e_value[j], gamma[j] = _sensitivity(r)
    kept = int(mask.sum())
    return {
        "mask": mask,
        "pass_rate": float(kept / k) if k else 0.0,
        "p_values": p_values,
        "partial_corr": partial,
        "e_value": e_value,
        "gamma": gamma,
    }
