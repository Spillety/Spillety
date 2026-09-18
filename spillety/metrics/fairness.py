import numpy as np

__all__ = ["equalized_odds"]


def _tpr_fpr(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    tpr = float(tp) / float(tp + fn) if tp + fn > 0 else float("nan")
    fpr = float(fp) / float(fp + tn) if fp + tn > 0 else float("nan")
    return tpr, fpr


def equalized_odds(y_true, y_pred, groups, n_boot=1000, random_state=72):
    """
    ## Max TPR/FPR gap across jurisdictions with bootstrap CI (§11.7)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_pred : np.ndarray
        Binary alert decisions.
    groups : array-like
        Group label per row (e.g. jurisdiction).
    n_boot : int
        Bootstrap replications, resampled independently per group.
    random_state : int
        Seed for resampling.

    Returns
    ----------
    dict
        Per-group `tpr`/`fpr`, `delta_tpr`/`delta_fpr` max gaps,
        and percentile `ci` for both gaps with `n_boot`.
    """
    # ponytail: max-gap summary, upgrade path pairwise group CIs + significance test.
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)
    names = sorted(set(groups.tolist()))
    tpr = {g: _tpr_fpr(y_true[groups == g], y_pred[groups == g])[0] for g in names}
    fpr = {g: _tpr_fpr(y_true[groups == g], y_pred[groups == g])[1] for g in names}

    def _gap(mapping):
        vals = np.array([v for v in mapping.values() if not np.isnan(v)])
        return float(vals.max() - vals.min()) if len(vals) > 0 else 0.0

    rng = np.random.default_rng(random_state)
    boot_tpr = np.empty(n_boot)
    boot_fpr = np.empty(n_boot)
    for b in range(n_boot):
        bt, bf = {}, {}
        for g in names:
            idx = np.flatnonzero(groups == g)
            pick = idx[rng.integers(0, len(idx), len(idx))]
            r_tpr, r_fpr = _tpr_fpr(y_true[pick], y_pred[pick])
            bt[g], bf[g] = r_tpr, r_fpr
        boot_tpr[b] = _gap(bt) if not all(np.isnan(v) for v in bt.values()) else 0.0
        boot_fpr[b] = _gap(bf) if not all(np.isnan(v) for v in bf.values()) else 0.0
    return {
        "tpr": tpr,
        "fpr": fpr,
        "delta_tpr": _gap(tpr),
        "delta_fpr": _gap(fpr),
        "ci": {
            "delta_tpr_lower": float(np.percentile(boot_tpr, 2.5)),
            "delta_tpr_upper": float(np.percentile(boot_tpr, 97.5)),
            "delta_fpr_lower": float(np.percentile(boot_fpr, 2.5)),
            "delta_fpr_upper": float(np.percentile(boot_fpr, 97.5)),
            "n_boot": n_boot,
        },
    }
