import numpy as np

M_GRID = (16, 24, 32)
EF_CONSTRUCTION_GRID = (64, 128, 256)
EF_SEARCH_GRID = (50, 100, 200)

DEFAULT_M = 24
DEFAULT_EF_CONSTRUCTION = 128
DEFAULT_EF_SEARCH = 100


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={a.ndim}")
    if a.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    return a


def brute_query(
    anchors: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    ## Exact top-K nearest anchors, ground truth for recall@K (§3.5.3)

    Parameters
    ----------
    anchors : np.ndarray
        Anchor embeddings (n_anchors, d).
    queries : np.ndarray
        Query embeddings (n_queries, d).
    k : int
        Number of neighbours, 1 <= k <= n_anchors.

    Returns
    ----------
    tuple[np.ndarray, np.ndarray]
        Euclidean distances (n_queries, k) and anchor indices (n_queries, k),
        both sorted by ascending distance. Distances feed
        `features.build.build_feature_matrix` as d_1..d_K.
    """
    a = _as_2d(anchors, "anchors")
    q = _as_2d(queries, "queries")
    if a.shape[1] != q.shape[1]:
        raise ValueError(f"dim mismatch: anchors {a.shape[1]} vs queries {q.shape[1]}")
    if not 1 <= k <= a.shape[0]:
        raise ValueError(f"k must be in [1, {a.shape[0]}], got {k}")
    # Squared L2 via (a-b)^2 expansion, clipped against fp error before sqrt.
    d2 = (q**2).sum(axis=1, keepdims=True) - 2.0 * q @ a.T + (a**2).sum(axis=1)
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1)
    top = np.take_along_axis(idx, order, axis=1)
    dist = np.sqrt(np.maximum(np.take_along_axis(d2, top, axis=1), 0.0))
    return dist, top


def recall_at_k(approx_idx: np.ndarray, exact_idx: np.ndarray, k: int) -> float:
    """
    ## ANN fidelity vs brute-force ground truth (§3.5.3)

    Parameters
    ----------
    approx_idx : np.ndarray
        Approximate top-K indices per query (n, K).
    exact_idx : np.ndarray
        Exact top-K indices per query (n, K) from `brute_query`.
    k : int
        Cutoff, 1 <= k <= K.

    Returns
    ----------
    float
        Mean overlap |approx top-k ∩ exact top-k| / k over queries.
    """
    a = np.asarray(approx_idx)
    e = np.asarray(exact_idx)
    if a.shape != e.shape or a.ndim != 2:
        raise ValueError("approx_idx and exact_idx must share one 2D shape")
    if not 1 <= k <= a.shape[1]:
        raise ValueError(f"k must be in [1, {a.shape[1]}], got {k}")
    hits = sum(len(set(a[i, :k]) & set(e[i, :k])) for i in range(a.shape[0]))
    return hits / (a.shape[0] * k)


def _require_hnswlib():
    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError(
            "hnswlib is required for HNSW retrieval; install it with "
            "'pip install spillety[hnsw]'"
        ) from exc
    return hnswlib


def build_index(
    vectors: np.ndarray,
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_EF_SEARCH,
):
    """
    ## Build HNSW anchor index (§3.5.3)

    Parameters
    ----------
    vectors : np.ndarray
        Anchor embeddings (n, d), float.
    m : int
        Graph connectivity, must be in {16, 24, 32}.
    ef_construction : int
        Build beam width, must be in {64, 128, 256}.
    ef_search : int
        Default query beam width, must be in {50, 100, 200}.

    Returns
    ----------
    hnswlib.Index
        Ready index with `ef_search` preset via `set_ef`.
    """
    if m not in M_GRID:
        raise ValueError(f"m must be in {M_GRID}, got {m}")
    if ef_construction not in EF_CONSTRUCTION_GRID:
        raise ValueError(f"ef_construction must be in {EF_CONSTRUCTION_GRID}, got {ef_construction}")
    if ef_search not in EF_SEARCH_GRID:
        raise ValueError(f"ef_search must be in {EF_SEARCH_GRID}, got {ef_search}")
    v = _as_2d(vectors, "vectors")
    hnswlib = _require_hnswlib()
    index = hnswlib.Index(space="l2", dim=v.shape[1])
    index.init_index(
        max_elements=v.shape[0], ef_construction=ef_construction, M=m
    )
    index.add_items(v.astype(np.float32))
    index.set_ef(ef_search)
    return index


def query_index(
    index, queries: np.ndarray, k: int, ef_search: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    ## Query HNSW index for K nearest anchors (§3.5.3)

    Parameters
    ----------
    index :
        Index from `build_index`.
    queries : np.ndarray
        Query embeddings (n, d).
    k : int
        Number of neighbours.
    ef_search : int | None
        Per-query beam width override, must be in {50, 100, 200}.

    Returns
    ----------
    tuple[np.ndarray, np.ndarray]
        Euclidean distances (n, k) and anchor indices (n, k), ascending.
        Distances feed `features.build.build_feature_matrix` as d_1..d_K.
    """
    if ef_search is not None and ef_search not in EF_SEARCH_GRID:
        raise ValueError(f"ef_search must be in {EF_SEARCH_GRID}, got {ef_search}")
    q = _as_2d(queries, "queries")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if ef_search is not None:
        index.set_ef(ef_search)
    labels, dist2 = index.knn_query(q.astype(np.float32), k=k)
    # hnswlib l2 space returns squared distances; sqrt restores Euclidean scale.
    return np.sqrt(np.maximum(dist2, 0.0)), labels
