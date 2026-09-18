import math

import numpy as np

__all__ = [
    "classify_regime",
    "hnsw_action",
    "mann_kendall",
    "regime_action",
    "retrain_gate",
    "walk_forward_retrain",
]

SILHOUETTE_LOW = 0.20
RECALL_FLOOR = 0.88
MAX_INSERTS = 20000
WALK_FORWARD_DROP = 0.10


def classify_regime(ks_drift: bool, silhouette: float) -> str:
    """
    ## Drift / novel-pattern / noise / normal decision table (§8.3.4)

    Parameters
    ----------
    ks_drift : bool
        Joint KS rule from `is_drift` (p + D + |d_Cohen|).
    silhouette : float
        Silhouette of new anchors against old clusters.

    Returns
    ----------
    str
        One of `drift`, `novel`, `noise`, `normal`.
    """
    low = silhouette < SILHOUETTE_LOW
    if ks_drift and low:
        return "drift"
    if low:
        return "novel"
    if ks_drift:
        return "noise"
    return "normal"


def regime_action(regime: str) -> str:
    """
    ## Operational action for a classified regime (§8.3.4)

    Parameters
    ----------
    regime : str
        One of `drift`, `novel`, `noise`, `normal`.

    Returns
    ----------
    str
        `retrain`, `review`, `ignore` or `incremental`.
    """
    return {"drift": "retrain", "novel": "review", "noise": "ignore", "normal": "incremental"}[regime]


def hnsw_action(recall_at_10: float, n_inserts: int) -> str:
    """
    ## Incremental HNSW update vs full rebuild (§8.4)

    Parameters
    ----------
    recall_at_10 : float
        Current recall@10 of the index.
    n_inserts : int
        Inserts since the last rebuild.

    Returns
    ----------
    str
        `rebuild` iff recall@10 < 0.88 or inserts ≥ 20k, else `incremental`.
    """
    if recall_at_10 < RECALL_FLOOR or n_inserts >= MAX_INSERTS:
        return "rebuild"
    return "incremental"


def mann_kendall(x) -> tuple:
    """
    ## Mann-Kendall trend test with tie correction (§8.2)

    Parameters
    ----------
    x : np.ndarray
        Ordered series (e.g. walk-forward PR-AUC per fold).

    Returns
    ----------
    tuple
        (S, p): Kendall score and two-sided normal-approximation p-value.
    """
    # ponytail: normal approximation, exact distribution only matters at n<10 — add when small-sample folds gate retrain.
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    tie = sum(t * (t - 1) * (2 * t + 5) for t in counts if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var == 0:
        return int(s), 1.0
    z = (s - 1) / math.sqrt(var) if s > 0 else ((s + 1) / math.sqrt(var) if s < 0 else 0.0)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return int(s), float(min(max(p, 0.0), 1.0))


def walk_forward_retrain(pr_first: float, pr_last: float) -> bool:
    """
    ## Walk-forward degradation gate: relative PR-AUC drop above 10% (§8.2)

    Parameters
    ----------
    pr_first : float
        PR-AUC of the earliest walk-forward fold.
    pr_last : float
        PR-AUC of the latest walk-forward fold.

    Returns
    ----------
    bool
        True iff (first − last) / first > 0.10.
    """
    if pr_first <= 0:
        return False
    return bool((pr_first - pr_last) / pr_first > WALK_FORWARD_DROP)


def retrain_gate(pr_first: float, pr_last: float, mk_s: int, mk_p: float) -> bool:
    """
    ## Retrain trigger: walk-forward drop or significant downward trend (§8.2)

    Parameters
    ----------
    pr_first : float
        PR-AUC of the earliest walk-forward fold.
    pr_last : float
        PR-AUC of the latest walk-forward fold.
    mk_s : int
        Mann-Kendall S score of the PR-AUC series.
    mk_p : float
        Mann-Kendall two-sided p-value.

    Returns
    ----------
    bool
        True iff walk-forward drop fires or (p < 0.05 with S < 0).
    """
    return bool(walk_forward_retrain(pr_first, pr_last) or (mk_p < 0.05 and mk_s < 0))
