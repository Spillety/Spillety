import numpy as np


def nt_xent(z: np.ndarray, tau: float = 0.1) -> float:
    """
    ## NT-Xent over a doubled batch (§4.2.2)

    Parameters
    ----------
    z : np.ndarray
        Array (2N, d); rows 2k/2k+1 form a positive pair.
    tau : float
        Temperature, must be > 0.

    Returns
    ----------
    float
        Mean loss over all 2N anchors.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if z.ndim != 2 or z.shape[0] % 2 != 0:
        raise ValueError(f"z must be (2N, d), got {z.shape}")
    n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    sim = (n @ n.T) / tau
    sim = sim - sim.max(axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(sim) * (1 - np.eye(sim.shape[0])), axis=1) + 1e-12)
    pos = np.arange(sim.shape[0]) ^ 1
    return float(np.mean(-(sim[np.arange(sim.shape[0]), pos] - logsumexp)))


def jaccard_index(a: set, b: set) -> float:
    """
    ## Jaccard overlap of two sanction lists (§4.4.3)

    Parameters
    ----------
    a, b : set
        List members as hashables.

    Returns
    ----------
    float
        |a∩b|/|a∪b|; 1.0 when both empty.
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pull_margin(m0: float, jaccard: float) -> float:
    """
    ## Weighted pull margin m_pull = m0 * (1 - J) (§4.4.3)

    Parameters
    ----------
    m0 : float
        Base margin, must be >= 0.
    jaccard : float
        List overlap in [0, 1].

    Returns
    ----------
    float
        Pull margin for this anchor pair.
    """
    if m0 < 0:
        raise ValueError(f"m0 must be >= 0, got {m0}")
    if not 0 <= jaccard <= 1:
        raise ValueError(f"jaccard must be in [0, 1], got {jaccard}")
    return m0 * (1 - jaccard)


def anchor_loss(
    za: np.ndarray,
    zn: np.ndarray,
    lam: float = 0.5,
    m0: float = 1.0,
    jaccard: float = 0.0,
    m_push: float = 1.0,
) -> float:
    """
    ## Anchor hinge loss: pull anchors, push non-anchors (§4.4.2)

    Parameters
    ----------
    za : np.ndarray
        Anchor embeddings (Na, d).
    zn : np.ndarray
        Non-anchor embeddings (Nn, d).
    lam : float
        Balance weight, must be >= 0.
    m0 : float
        Base pull margin; effective margin is m0 * (1 - jaccard).
    jaccard : float
        Overlap of the sanction lists the anchors come from.
    m_push : float
        Minimum anchor/non-anchor distance.

    Returns
    ----------
    float
        lam * (pull + push).
    """
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    m_pull = pull_margin(m0, jaccard)
    # ponytail: O(Na^2 + Na*Nn) pairwise distances; for Na > 20k switch to batched chunks
    d_aa = np.linalg.norm(za[:, None, :] - za[None, :, :], axis=-1)
    triu = d_aa[np.triu_indices(len(za), k=1)] if len(za) > 1 else np.zeros(0)
    pull = float(np.maximum(0.0, triu - m_pull).mean()) if triu.size else 0.0
    d_an = np.linalg.norm(za[:, None, :] - zn[None, :, :], axis=-1)
    push = float(np.maximum(0.0, m_push - d_an).mean())
    return lam * (pull + push)
