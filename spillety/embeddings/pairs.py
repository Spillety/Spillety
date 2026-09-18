import numpy as np


def _rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng


def positive_pairs(
    edges: np.ndarray,
    labels: np.ndarray | None = None,
    times: np.ndarray | None = None,
    eps: int = 1,
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

    Returns
    ----------
    np.ndarray
        Sampled pairs (n, 2).
    """
    g = _rng(rng)
    p = sampling_probs(degrees, alpha)
    return g.choice(n_nodes, size=(n, 2), replace=True, p=p)


def hard_negatives(
    labels: np.ndarray,
    times: np.ndarray,
    n: int,
    eps: int = 1,
    use_curriculum: bool = False,
    rng: int | np.random.Generator | None = None,
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

    Returns
    ----------
    np.ndarray
        Sampled pairs (M, 2), M <= n; empty when curriculum is off.
    """
    if not use_curriculum:
        return np.zeros((0, 2), dtype=int)
    g = _rng(rng)
    # ponytail: O(tries) rejection sampling; for dense graphs switch to bucketed candidates
    out = []
    tries = 0
    while len(out) < n and tries < 20 * n + 100:
        a, b = g.integers(0, len(labels), size=2)
        tries += 1
        if a != b and labels[a] != labels[b] and abs(int(times[a]) - int(times[b])) <= eps:
            out.append((a, b))
    return np.array(out, dtype=int).reshape(-1, 2)
