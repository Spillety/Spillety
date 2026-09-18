import numpy as np
from scipy.stats import ks_2samp


def _rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng


def mask_features(
    x: np.ndarray, p: float = 0.1, rng: int | np.random.Generator | None = None
) -> np.ndarray:
    """
    ## Feature masking: zero each entry with probability p (§4.8.2)

    Parameters
    ----------
    x : np.ndarray
        Feature matrix (N, d).
    p : float
        Mask rate in [0, 1].
    rng : int | Generator | None
        Seed or generator.

    Returns
    ----------
    np.ndarray
        Masked copy of `x`.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return x * (_rng(rng).random(x.shape) >= p)


def drop_edges(
    edge_index: np.ndarray, p: float = 0.1, rng: int | np.random.Generator | None = None
) -> np.ndarray:
    """
    ## Edge dropping: keep each edge with probability 1 - p (§4.8.2)

    Parameters
    ----------
    edge_index : np.ndarray
        Edge list (2, E).
    p : float
        Drop rate in [0, 1].
    rng : int | Generator | None
        Seed or generator.

    Returns
    ----------
    np.ndarray
        Kept edges (2, E').
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return edge_index[:, _rng(rng).random(edge_index.shape[1]) >= p]


def perturb_features(
    x: np.ndarray, sigma: float = 0.01, rng: int | np.random.Generator | None = None
) -> np.ndarray:
    """
    ## Additive Gaussian noise on node features (§4.8.2)

    Parameters
    ----------
    x : np.ndarray
        Feature matrix (N, d).
    sigma : float
        Noise std, must be >= 0.
    rng : int | Generator | None
        Seed or generator.

    Returns
    ----------
    np.ndarray
        Noisy copy of `x`.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    return x + _rng(rng).normal(0.0, sigma, size=x.shape)


def ks_gate(x_ref: np.ndarray, x_new: np.ndarray, p_value: float = 0.05) -> tuple[bool, float]:
    """
    ## KS-gate: accept augmentation iff distributions agree (§4.6.3)

    Parameters
    ----------
    x_ref : np.ndarray
        Reference features.
    x_new : np.ndarray
        Augmented features.
    p_value : float
        Accept threshold on the two-sample KS p-value.

    Returns
    ----------
    tuple[bool, float]
        (accept, p); accept is True iff p >= threshold.
    """
    p = float(ks_2samp(np.ravel(x_ref), np.ravel(x_new)).pvalue)
    return p >= p_value, p
