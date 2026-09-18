PQ_M = 16
PQ_K = 256
BYTES_PER_CODE = 1
RAW_BYTES_PER_DIM = 4

MIN_RECALL_AT_K = 0.93
MAX_PR_AUC_DROP = 0.01


def raw_bytes(n: int, dim: int) -> int:
    """
    ## Float32 footprint of n vectors (§3.5.4)

    Parameters
    ----------
    n : int
        Vector count.
    dim : int
        Embedding dim.

    Returns
    ----------
    int
        Bytes for float32 storage (n * dim * 4).
    """
    if n < 0 or dim < 1:
        raise ValueError(f"need n >= 0 and dim >= 1, got n={n} dim={dim}")
    return n * dim * RAW_BYTES_PER_DIM


def pq_bytes(n: int, m: int = PQ_M) -> int:
    """
    ## PQ footprint of n vectors (§3.5.4)

    Parameters
    ----------
    n : int
        Vector count.
    m : int
        Subvector count; k=256 codes fit one byte each.

    Returns
    ----------
    int
        Bytes for PQ codes (n * m).
    """
    # ponytail: counts codes only, codebooks (~m*k*d/m floats) are negligible at 80M scale.
    if n < 0 or m < 1:
        raise ValueError(f"need n >= 0 and m >= 1, got n={n} m={m}")
    return n * m * BYTES_PER_CODE


def compression_ratio(dim: int, m: int = PQ_M) -> float:
    """
    ## Float32-to-PQ footprint ratio (§3.5.4)

    Parameters
    ----------
    dim : int
        Embedding dim.
    m : int
        Subvector count.

    Returns
    ----------
    float
        raw_bytes / pq_bytes for any n (dim * 4 / m); 32.0 at dim=128, m=16.
    """
    if dim < 1 or m < 1:
        raise ValueError(f"need dim >= 1 and m >= 1, got dim={dim} m={m}")
    return (dim * RAW_BYTES_PER_DIM) / (m * BYTES_PER_CODE)


def should_use_pq(
    pr_auc_full: float,
    pr_auc_pq: float,
    recall_at_k: float,
    max_drop: float = MAX_PR_AUC_DROP,
    min_recall: float = MIN_RECALL_AT_K,
) -> bool:
    """
    ## PQ-vs-float32 selection rule (§3.5.4)

    Parameters
    ----------
    pr_auc_full : float
        Downstream PR-AUC with exact vectors.
    pr_auc_pq : float
        Downstream PR-AUC with PQ vectors.
    recall_at_k : float
        ANN recall@K of the PQ index vs brute force.
    max_drop : float
        Max tolerated PR-AUC drop (default 0.01).
    min_recall : float
        Min tolerated recall@K (default 0.93).

    Returns
    ----------
    bool
        True iff (pr_auc_full - pr_auc_pq) < max_drop and recall_at_k > min_recall.
    """
    # ponytail: no codebook training here (needs faiss); rule + footprint only.
    return (pr_auc_full - pr_auc_pq) < max_drop and recall_at_k > min_recall
