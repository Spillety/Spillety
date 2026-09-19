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


def hetero_contrastive_loss(
    z_addr: np.ndarray,
    z_tx: np.ndarray,
    pos: np.ndarray,
    tau: float = 0.1,
) -> float:
    """
    ## Cross-type NT-Xent over address→tx positive pairs (§4.2.2, §4.4.2)

    Parameters
    ----------
    z_addr : np.ndarray
        Address embeddings (Na, d).
    z_tx : np.ndarray
        Transaction embeddings (Nt, d).
    pos : np.ndarray
        Positive pairs (P, 2) of (addr_idx, tx_idx).
    tau : float
        Temperature, must be > 0.

    Returns
    ----------
    float
        Mean cross-entropy of each anchor address over all tx nodes.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if z_addr.ndim != 2 or z_tx.ndim != 2:
        raise ValueError("z_addr and z_tx must be 2D")
    if z_addr.shape[1] != z_tx.shape[1]:
        raise ValueError("z_addr and z_tx must share width d")
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be (P, 2), got {pos.shape}")
    if len(pos) == 0:
        raise ValueError("pos must be non-empty")
    if pos[:, 0].max() >= len(z_addr) or pos[:, 1].max() >= len(z_tx):
        raise ValueError("pos indices out of range")
    if pos.min() < 0:
        raise ValueError("pos indices must be >= 0")
    # ponytail: O(Na*Nt) full similarity; for Na*Nt > 1e8 switch to batched chunks.
    na = z_addr / (np.linalg.norm(z_addr, axis=1, keepdims=True) + 1e-12)
    nt = z_tx / (np.linalg.norm(z_tx, axis=1, keepdims=True) + 1e-12)
    sim = (na @ nt.T) / tau
    sim = sim - sim.max(axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(sim), axis=1) + 1e-12)
    return float(np.mean(-(sim[pos[:, 0], pos[:, 1]] - logsumexp[pos[:, 0]])))
