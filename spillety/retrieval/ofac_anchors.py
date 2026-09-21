import csv
from dataclasses import dataclass, field

import numpy as np

from spillety.retrieval.hnsw import brute_query

_DEFAULT_CSV = "data/sanctions/addresses.csv"
_PROBE_K = 10


@dataclass
class AnchorPool:
    kind: str
    matrix: np.ndarray
    meta: dict = field(default_factory=dict)
    coverage: float = 0.0
    era: str | None = None


def load_ofac_pool(csv_path: str = _DEFAULT_CSV) -> list[dict]:
    """
    ## OFAC address pool from sanctions CSV (§D1)

    Parameters
    ----------
    csv_path : str
        Path to sanctions CSV; defaults to data/sanctions/addresses.csv.

    Returns
    ----------
    list[dict]
        [{address, source, currency}]; missing/empty/malformed file -> [].
    """
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "address" not in reader.fieldnames:
                return []
            pool = []
            for row in reader:
                address = (row.get("address") or "").strip()
                if not address:
                    continue
                pool.append(
                    {
                        "address": address,
                        "source": (row.get("source") or "").strip(),
                        "currency": (row.get("currency") or "").strip(),
                    }
                )
    except (OSError, csv.Error):
        return []
    return pool


def build_anchor_pool_temporal(
    embeddings,
    labels,
    times,
    ofac_matrix,
    era_split: int = 30,
) -> dict[str, AnchorPool]:
    """
    ## Build OFAC anchor pools per temporal era (§4.3.5)

    Parameters
    ----------
    embeddings : np.ndarray
        Anchor embeddings (n, d).
    labels : np.ndarray
        Binary labels (1 = illicit).
    times : np.ndarray
        Timestep per node.
    ofac_matrix : np.ndarray | None
        OFAC embeddings in the same dim.
    era_split : int
        Boundary between train and valid eras.

    Returns
    ----------
    dict[str, AnchorPool]
        Keys "train", "valid", "test" with separate pools per era.
    """
    from spillety.embeddings.pairs import temporal_anchor_split

    emb = np.asarray(embeddings, dtype=float)
    lab = np.asarray(labels)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError("embeddings must be non-empty 2D")
    if lab.shape != (emb.shape[0],):
        raise ValueError("labels must match embeddings rows")
    eras = temporal_anchor_split(np.arange(len(emb)), times, era_split=era_split)
    ofac = (
        np.asarray(ofac_matrix, dtype=float)
        if ofac_matrix is not None
        else np.zeros((0, emb.shape[1]))
    )
    pools: dict[str, AnchorPool] = {}
    for era_name, idx in eras.items():
        if len(idx) == 0:
            continue
        era_emb = emb[idx]
        era_lab = lab[idx]
        pool = build_anchor_pool(era_emb, era_lab, ofac if ofac.shape[0] > 0 else None)
        pool.era = era_name
        pools[era_name] = pool
    return pools


def build_anchor_pool(embeddings, labels, ofac_matrix_or_none) -> AnchorPool:
    """
    ## OFAC-first anchor pool with train-illicit fallback (§D1). Absence of an anchor proves nothing (K0).

    Parameters
    ----------
    embeddings, labels : np.ndarray
        Candidate anchor embeddings and binary labels (1 = illicit).
    ofac_matrix_or_none : np.ndarray | None
        OFAC embeddings in the same dim; None or empty -> train-illicit only.

    Returns
    ----------
    AnchorPool
        kind ∈ {ofac, train_illicit, mixed}; coverage = share of queries hitting OFAC top-K.
    """
    emb = np.asarray(embeddings, dtype=float)
    lab = np.asarray(labels)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError("embeddings must be non-empty 2D")
    if lab.shape != (emb.shape[0],):
        raise ValueError("labels must match embeddings rows")
    train = emb[lab == 1]
    ofac = (
        np.asarray(ofac_matrix_or_none, dtype=float)
        if ofac_matrix_or_none is not None
        else np.zeros((0, emb.shape[1]))
    )
    if ofac.size == 0:
        ofac = np.zeros((0, emb.shape[1]))
    if ofac.ndim != 2 or ofac.shape[1] != emb.shape[1]:
        raise ValueError("OFAC dim must match embeddings dim")
    n_ofac, n_train = ofac.shape[0], train.shape[0]
    if n_ofac and n_train:
        kind, matrix = "mixed", np.vstack([ofac, train])
    elif n_ofac:
        kind, matrix = "ofac", ofac
    elif n_train:
        kind, matrix = "train_illicit", train
    else:
        raise ValueError("no anchors: OFAC pool and train-illicit are both empty")
    # ponytail: brute-force probe is O(N*No*d); upgrade path — HNSW probe past 1e8 pairs.
    if n_ofac == 0:
        coverage = 0.0
    else:
        dist, _ = brute_query(ofac, emb, min(_PROBE_K, n_ofac))
        # Finite nearest distance means the query hit a non-empty OFAC top-K.
        coverage = float(np.isfinite(dist[:, 0]).mean())
    return AnchorPool(
        kind=kind, matrix=matrix, meta={"n_ofac": n_ofac, "n_train": n_train}, coverage=coverage
    )


def query_with_fallback(pool: AnchorPool, queries, k: int = 10):
    """
    ## OFAC-first top-K; fallback to train-illicit (§D1)

    Parameters
    ----------
    pool : AnchorPool
        Pool from build_anchor_pool; OFAC rows first.
    queries : np.ndarray
        Query embeddings (n_queries, d).
    k : int
        Neighbours per query, clamped to sub-pool size.

    Returns
    ----------
    tuple[np.ndarray, np.ndarray, str]
        Distances, indices global into pool.matrix, and used sub-pool kind.
    """
    q = np.asarray(queries, dtype=float)
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError("queries must be non-empty 2D")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n_ofac = int(pool.meta.get("n_ofac", 0))
    ofac_part, train_part = pool.matrix[:n_ofac], pool.matrix[n_ofac:]
    # ponytail: pool-level fallback only; upgrade path — per-query distance gate when calibration lands.
    if ofac_part.shape[0] > 0:
        dist, idx = brute_query(ofac_part, q, min(k, ofac_part.shape[0]))
        return dist, idx, "ofac"
    if train_part.shape[0] == 0:
        raise ValueError("no anchors: OFAC pool and train-illicit are both empty")
    dist, idx = brute_query(train_part, q, min(k, train_part.shape[0]))
    return dist, idx + n_ofac, "train_illicit"
