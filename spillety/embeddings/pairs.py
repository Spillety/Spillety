import numpy as np


def _rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng


def temporal_anchor_split(
    anchor_indices: np.ndarray,
    times: np.ndarray,
    era_split: int = 30,
) -> dict[str, np.ndarray]:
    """
    ## Split anchor indices into train/valid/test by temporal era (§4.3.5)

    Parameters
    ----------
    anchor_indices : np.ndarray
        Integer indices of anchor nodes.
    times : np.ndarray
        Timestep per node; used to assign era.
    era_split : int
        Boundary between train and valid eras (default 30).
        Valid = era_split+1..era_split+10, test = era_split+11..

    Returns
    ----------
    dict[str, np.ndarray]
        Keys "train", "valid", "test" with arrays of anchor indices
        falling into each temporal era.
    """
    train_mask = times[anchor_indices] <= era_split
    valid_mask = (times[anchor_indices] > era_split) & (times[anchor_indices] <= era_split + 10)
    test_mask = times[anchor_indices] > era_split + 10
    return {
        "train": anchor_indices[train_mask],
        "valid": anchor_indices[valid_mask],
        "test": anchor_indices[test_mask],
    }


def positive_pairs(
    edges: np.ndarray,
    labels: np.ndarray | None = None,
    times: np.ndarray | None = None,
    eps: int = 1,
    era_split: int | None = None,
) -> np.ndarray:
    """
    ## Co-spending positives filtered by label agreement and Δt (§4.3.1)

    Parameters
    ----------
    edges : np.ndarray
        Integer edge list (E, 2), node ids index `labels`/`times`.
    labels : np.ndarray | None
        Cluster/label id per node; None skips the agreement filter.
    times : np.ndarray | None
        Timestep per node; None skips the proximity filter.
    eps : int
        Max |Δt| for the temporal filter.
    era_split : int | None
        If given, only keep edges where both endpoints are in the same era.

    Returns
    ----------
    np.ndarray
        Filtered pairs (P, 2).
    """
    keep = np.ones(len(edges), dtype=bool)
    if labels is not None:
        keep &= labels[edges[:, 0]] == labels[edges[:, 1]]
    if times is not None:
        keep &= np.abs(times[edges[:, 0]] - times[edges[:, 1]]) <= eps
    if era_split is not None:
        t0, t1 = times[edges[:, 0]], times[edges[:, 1]]
        same_era = ((t0 <= era_split) & (t1 <= era_split)) | \
                   ((t0 > era_split) & (t1 > era_split) & (t0 <= era_split + 10) & (t1 <= era_split + 10)) | \
                   ((t0 > era_split + 10) & (t1 > era_split + 10))
        keep &= same_era
    return edges[keep]


def sampling_probs(degrees: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """
    ## Degree-corrected negative sampling distribution (§4.3.2)

    Parameters
    ----------
    degrees : np.ndarray
        Node degrees, must be > 0.
    alpha : float
        Correction exponent in [0.5, 1].

    Returns
    ----------
    np.ndarray
        Probabilities p(v) ∝ 1/deg(v)^alpha summing to 1.
    """
    if np.any(degrees <= 0):
        raise ValueError("degrees must be > 0")
    w = 1.0 / np.power(degrees.astype(float), alpha)
    return w / w.sum()


def sample_negatives(
    n_nodes: int,
    degrees: np.ndarray,
    n: int,
    alpha: float = 0.75,
    rng: int | np.random.Generator | None = None,
    times: np.ndarray | None = None,
    era_split: int = 30,
) -> np.ndarray:
    """
    ## Random negative pairs from the degree-corrected distribution (§4.3.2)

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    degrees : np.ndarray
        Node degrees, length n_nodes.
    n : int
        Pairs to sample.
    alpha : float
        Correction exponent.
    rng : int | Generator | None
        Seed or generator.
    times : np.ndarray | None
        Timestep per node; if given, negatives are sampled within the same era.
    era_split : int
        Boundary for temporal era when times is provided.

    Returns
    ----------
    np.ndarray
        Sampled pairs (n, 2).
    """
    g = _rng(rng)
    p = sampling_probs(degrees, alpha)
    if times is not None:
        era = np.digitize(times, [era_split, era_split + 10])
        out = np.zeros((n, 2), dtype=int)
        for i in range(n):
            same = era == g.integers(0, 3)
            candidates = np.where(same)[0]
            if len(candidates) < 2:
                out[i] = g.choice(n_nodes, size=2, replace=True, p=p)
            else:
                idx = g.choice(len(candidates), size=2, replace=False)
                out[i] = candidates[idx]
        return out
    return g.choice(n_nodes, size=(n, 2), replace=True, p=p)


def hard_negatives(
    labels: np.ndarray,
    times: np.ndarray,
    n: int,
    eps: int = 1,
    use_curriculum: bool = False,
    rng: int | np.random.Generator | None = None,
    era_split: int = 30,
) -> np.ndarray:
    """
    ## Hard negatives: same window, different clusters (§4.3.2)

    Parameters
    ----------
    labels : np.ndarray
        Cluster/label id per node.
    times : np.ndarray
        Timestep per node.
    n : int
        Pairs to sample.
    eps : int
        Max |Δt|.
    use_curriculum : bool
        False (early training) returns empty; True enables late-stage sampling.
    rng : int | Generator | None
        Seed or generator.
    era_split : int
        Boundary for temporal era; hard negatives are drawn within the same era.

    Returns
    ----------
    np.ndarray
        Sampled pairs (M, 2), M <= n; empty when curriculum is off.
    """
    if not use_curriculum:
        return np.zeros((0, 2), dtype=int)
    g = _rng(rng)
    era = np.digitize(times, [era_split, era_split + 10])
    # ponytail: O(tries) rejection sampling; for dense graphs switch to bucketed candidates
    out = []
    tries = 0
    while len(out) < n and tries < 20 * n + 100:
        a, b = g.integers(0, len(labels), size=2)
        tries += 1
        if a != b and labels[a] != labels[b] and abs(int(times[a]) - int(times[b])) <= eps and era[a] == era[b]:
            out.append((a, b))
    return np.array(out, dtype=int).reshape(-1, 2)


def knn_hard_negatives(
    z: np.ndarray,
    n: int,
    k: int = 10,
    labels: np.ndarray | None = None,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    ## Hard negatives from the kNN neighborhood, not uniform sampling (§4.3.2)

    Parameters
    ----------
    z : np.ndarray
        L2-comparable embeddings (N, d).
    n : int
        Pairs to sample.
    k : int
        Neighborhood size; must satisfy 1 <= k < N.
    labels : np.ndarray | None
        Cluster id per node; same-label neighbors are excluded when given.
    rng : int | Generator | None
        Seed or generator.

    Returns
    ----------
    np.ndarray
        Sampled pairs (M, 2), M <= n; each partner lies in the
        anchor's k nearest neighbors.
    """
    g = _rng(rng)
    count = z.shape[0]
    if not 1 <= k < count:
        raise ValueError(f"k must satisfy 1 <= k < N={count}, got {k}")
    if n <= 0:
        return np.zeros((0, 2), dtype=int)
    # ponytail: O(N^2) full distance matrix; for N > 50k switch to HNSW retrieval.
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1)[:, :k]
    out = []
    for _ in range(n):
        a = int(g.integers(0, count))
        pool = order[a]
        if labels is not None:
            pool = pool[labels[pool] != labels[a]]
        if len(pool) == 0:
            continue
        out.append((a, int(pool[g.integers(0, len(pool))])))
    return np.array(out, dtype=int).reshape(-1, 2)
