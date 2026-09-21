import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

__all__ = ["choose_dbscan_params", "cluster_dbscan", "select_eps"]


def select_eps(embeddings: np.ndarray, min_pts: int, q: float = 0.95) -> float:
    """
    ## eps = k-distance quantile (§5.9)

    Parameters
    ----------
    embeddings : np.ndarray
        Point cloud (n, d).
    min_pts : int
        DBSCAN min_samples = k for k-distance.
    q : float
        Quantile in (0, 1]; default 0.95.

    Returns
    ----------
    float
        Distance threshold for DBSCAN.eps.
    """
    X = np.asarray(embeddings, dtype=float)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D")
    n = X.shape[0]
    if n == 0:
        raise ValueError("embeddings is empty")
    k = min(min_pts, n)
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = distances[:, -1]
    k_dist.sort()
    idx = min(int(np.ceil(q * n)) - 1, n - 1)
    return float(k_dist[idx])


def _split_large_cluster(X: np.ndarray, labels: np.ndarray, cid: int, max_size: int, eps: float, min_pts: int, depth: int = 0) -> np.ndarray:
    """
    Recursively split a cluster larger than max_size by re-clustering its points with tighter eps.
    """
    if depth > 10:  # Prevent infinite recursion
        return labels

    mask = labels == cid
    sub_X = X[mask]
    if sub_X.shape[0] <= max_size:
        return labels

    # Compute a tighter eps from the sub-cluster's own k-distance
    k = min(min_pts, sub_X.shape[0])
    if k < 2:
        return labels
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(sub_X)
    distances, _ = nbrs.kneighbors(sub_X)
    k_dist = distances[:, -1]
    k_dist.sort()
    idx = min(int(np.ceil(0.5 * sub_X.shape[0])) - 1, sub_X.shape[0] - 1)
    sub_eps = max(float(k_dist[idx]), 1e-6)

    sub_labels = DBSCAN(eps=sub_eps, min_samples=min_pts, metric="euclidean").fit_predict(sub_X)

    # Check if any resulting cluster is still too large
    largest_sub = 0
    for c in np.unique(sub_labels):
        if c != -1:
            sz = (sub_labels == c).sum()
            largest_sub = max(largest_sub, sz)
    if largest_sub > max_size:
        # Force KMeans split
        n_splits = int(np.ceil(sub_X.shape[0] / max_size))
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_splits, n_init=10, random_state=42 + depth).fit(sub_X)
        sub_labels = km.labels_

    # Remap sub_labels to new global cluster IDs
    new_labels = labels.copy()
    next_id = int(labels.max()) + 1 if labels.size > 0 else 0
    # Build assignment array for masked positions
    assign = np.full(mask.sum(), -1, dtype=int)
    for sub_cid in np.unique(sub_labels):
        if sub_cid == -1:
            assign[sub_labels == sub_cid] = -1
        else:
            assign[sub_labels == sub_cid] = next_id
            next_id += 1
    new_labels[mask] = assign

    return new_labels


def cluster_dbscan(embeddings: np.ndarray, eps: float, min_pts: int, max_size: int) -> np.ndarray:
    """
    ## DBSCAN with max cluster size cap (§5.9)

    Parameters
    ----------
    embeddings : np.ndarray
        Point cloud (n, d).
    eps : float
        Neighborhood radius.
    min_pts : int
        Minimum points to form a dense region.
    max_size : int
        Cap on cluster size; larger clusters are recursively split by density.

    Returns
    ----------
    np.ndarray
        Cluster labels; -1 for noise. No merge of noise points.
    """
    X = np.asarray(embeddings, dtype=float)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D")
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    labels = DBSCAN(eps=eps, min_samples=min_pts, metric="euclidean").fit_predict(X)

    # Iteratively split clusters that exceed max_size until all are within limit
    max_iter = 20
    for _ in range(max_iter):
        oversized = [cid for cid in np.unique(labels) if cid != -1 and (labels == cid).sum() > max_size]
        if not oversized:
            break
        for cid in oversized:
            labels = _split_large_cluster(X, labels, cid, max_size, eps, min_pts)

    return labels


def choose_dbscan_params(
    embeddings: np.ndarray,
    labels_val: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> dict:
    """
    ## Cost-optimal (eps, min_pts) on validation (§5.9)

    Parameters
    ----------
    embeddings : np.ndarray
        Validation embeddings (n, d).
    labels_val : np.ndarray
        Ground-truth cluster IDs for validation (same n).
    cost_fp : float
        Cost of false merge (pair-wise).
    cost_fn : float
        Cost of false split (pair-wise).

    Returns
    ----------
    dict
        {
            "eps": float,
            "min_pts": int,
            "cost": float,
            "cost_breakdown": {"fp": int, "fn": int, "cost_fp": float, "cost_fn": float},
            "n_clusters": int,
            "noise_frac": float,
        }
    """
    X = np.asarray(embeddings, dtype=float)
    y_true = np.asarray(labels_val, dtype=int)
    if X.shape[0] != y_true.shape[0]:
        raise ValueError("embeddings and labels_val length mismatch")
    if X.shape[0] == 0:
        raise ValueError("embeddings is empty")

    n = X.shape[0]
    # Candidate min_pts grid: 3 to min(50, n//2)
    min_pts_candidates = [m for m in [3, 5, 10, 20, 30, 40, 50] if m <= n // 2 and m >= 3]
    if not min_pts_candidates:
        min_pts_candidates = [min(3, max(1, n // 2))]

    # For each min_pts, get eps candidates via quantiles
    eps_quantiles = [0.80, 0.85, 0.90, 0.95, 0.98]
    best = None
    best_cost = np.inf

    for min_pts in min_pts_candidates:
        for q in eps_quantiles:
            eps = select_eps(X, min_pts, q=q)
            labels = cluster_dbscan(X, eps=eps, min_pts=min_pts, max_size=n)
            # Pairwise costs from cluster labels
            pred_same = np.zeros((n, n), dtype=bool)
            for cid in np.unique(labels):
                if cid == -1:
                    continue
                idx = np.where(labels == cid)[0]
                if len(idx) > 1:
                    pred_same[np.ix_(idx, idx)] = True
            true_same = np.zeros((n, n), dtype=bool)
            for cid in np.unique(y_true):
                idx = np.where(y_true == cid)[0]
                if len(idx) > 1:
                    true_same[np.ix_(idx, idx)] = True

            fp = int((pred_same & ~true_same).sum() // 2)  # upper triangle
            fn = int((~pred_same & true_same).sum() // 2)
            cost = float(cost_fp * fp + cost_fn * fn)

            if cost < best_cost:
                best_cost = cost
                n_clusters = len([c for c in np.unique(labels) if c != -1])
                noise_frac = float((labels == -1).sum()) / n
                best = {
                    "eps": float(eps),
                    "min_pts": int(min_pts),
                    "cost": cost,
                    "cost_breakdown": {
                        "fp": fp,
                        "fn": fn,
                        "cost_fp": float(cost_fp * fp),
                        "cost_fn": float(cost_fn * fn),
                    },
                    "n_clusters": n_clusters,
                    "noise_frac": noise_frac,
                }

    if best is None:
        # Fallback
        eps = select_eps(X, min_pts_candidates[0], q=0.95)
        labels = cluster_dbscan(X, eps=eps, min_pts=min_pts_candidates[0], max_size=n)
        n_clusters = len([c for c in np.unique(labels) if c != -1])
        noise_frac = float((labels == -1).sum()) / n
        pred_same = np.zeros((n, n), dtype=bool)
        for cid in np.unique(labels):
            if cid == -1:
                continue
            idx = np.where(labels == cid)[0]
            if len(idx) > 1:
                pred_same[np.ix_(idx, idx)] = True
        true_same = np.zeros((n, n), dtype=bool)
        for cid in np.unique(y_true):
            idx = np.where(y_true == cid)[0]
            if len(idx) > 1:
                true_same[np.ix_(idx, idx)] = True
        fp = int((pred_same & ~true_same).sum() // 2)
        fn = int((~pred_same & true_same).sum() // 2)
        best = {
            "eps": float(eps),
            "min_pts": int(min_pts_candidates[0]),
            "cost": float(cost_fp * fp + cost_fn * fn),
            "cost_breakdown": {"fp": fp, "fn": fn, "cost_fp": float(cost_fp * fp), "cost_fn": float(cost_fn * fn)},
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
        }

    return best