#!/usr/bin/env python3
"""## J3 HNSW retrieval on real PCA32 embeddings (§3.5)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_decision_path_v2 import N_PCA, SEED, load_enriched

from spillety.retrieval.hnsw import (
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_EF_SEARCH,
    DEFAULT_M,
    brute_query,
    build_index,
    query_index,
    recall_at_k,
)
from spillety.retrieval.pq import (
    PQ_M,
    compression_ratio,
    pq_bytes,
    raw_bytes,
    should_use_pq,
)

K = 10
N_QUERIES = 1000
TRAIN_END = 30


def pca_all_labeled(X_raw, steps):
    """## PCA32 fit on train only, transform of all labeled rows."""
    tr = steps <= TRAIN_END
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(X_raw[tr])
    return pca.transform(X_raw)


def sample_queries(Z, n, seed):
    """## Deterministic query subset of the anchor set."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(Z.shape[0], size=min(n, Z.shape[0]), replace=False)
    return Z[idx], idx


def exact_topk(Z, Q, k):
    """## Chunked brute-force ground truth for large anchor sets."""
    # ponytail: fixed 250-row chunks; streaming top-k left for 80M scale.
    dist_parts, idx_parts = [], []
    for s in range(0, len(Q), 250):
        d, i = brute_query(Z, Q[s : s + 250], k)
        dist_parts.append(d)
        idx_parts.append(i)
    return np.vstack(dist_parts), np.vstack(idx_parts)


def time_queries(index, Q, k):
    """## Per-query HNSW latencies in milliseconds."""
    lat = np.empty(len(Q))
    idx_out = np.empty((len(Q), k), dtype=np.int64)
    for i in range(len(Q)):
        t0 = time.perf_counter()
        _, idx = query_index(index, Q[i : i + 1], k)
        lat[i] = (time.perf_counter() - t0) * 1000.0
        idx_out[i] = idx[0]
    return lat, idx_out


def train_pq(Z, m, seed):
    """## Per-subspace KMeans codebooks and uint8 codes (m subspaces)."""
    # ponytail: sklearn KMeans instead of faiss; ADC search below.
    d = Z.shape[1]
    if d % m:
        raise ValueError(f"dim {d} not divisible by m={m}")
    sub = d // m
    books = np.empty((m, 256, sub))
    codes = np.empty((Z.shape[0], m), dtype=np.uint8)
    for j in range(m):
        km = KMeans(n_clusters=256, random_state=seed, n_init=3).fit(Z[:, j * sub : (j + 1) * sub])
        books[j] = km.cluster_centers_
        codes[:, j] = km.labels_
    return books, codes


def pq_topk(Q, codes, books, k):
    """## Asymmetric-distance top-K over PQ codes."""
    m, _, sub = books.shape
    approx = np.empty((len(Q), k), dtype=np.int64)
    for qi in range(len(Q)):
        lut = ((books - Q[qi].reshape(m, 1, sub)) ** 2).sum(axis=2)
        d2 = np.zeros(codes.shape[0])
        for j in range(m):
            d2 += lut[j][codes[:, j]]
        approx[qi] = np.argpartition(d2, k - 1)[:k][np.argsort(d2[np.argpartition(d2, k - 1)[:k]])]
    return approx


def main():
    """## Build HNSW + PQ on real embeddings, dump metrics and README."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/elliptic_raw")
    ap.add_argument("--out", default="models/retrieval_v1")
    ap.add_argument("--index", default="data/derived/hnsw_v1.bin")
    opt = ap.parse_args()
    out, idx_path = Path(opt.out), Path(opt.index)
    out.mkdir(parents=True, exist_ok=True)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    X_raw, _, _, _, steps = load_enriched(opt.root)
    Z = pca_all_labeled(X_raw, steps)
    n, dim = Z.shape

    t0 = time.perf_counter()
    index = build_index(Z, m=DEFAULT_M, ef_construction=DEFAULT_EF_CONSTRUCTION, ef_search=DEFAULT_EF_SEARCH)
    build_s = time.perf_counter() - t0
    index.save_index(str(idx_path))

    Q, _ = sample_queries(Z, N_QUERIES, SEED)
    _, exact_idx = exact_topk(Z, Q, K)
    lat, approx_idx = time_queries(index, Q, K)
    recall = recall_at_k(approx_idx, exact_idx, K)

    books, codes = train_pq(Z, PQ_M, SEED)
    pq_idx = pq_topk(Q, codes, books, K)
    pq_recall = recall_at_k(pq_idx, exact_idx, K)
    ratio = compression_ratio(dim, PQ_M)
    # ponytail: ΔPR-AUC needs GBDT retraining on PQ vectors (out of J3 scope);
    # conditional rule assumes zero PR-AUC drop, decision reduces to recall gate.
    use_pq = should_use_pq(0.0, 0.0, pq_recall)

    metrics = {
        "seed": SEED,
        "n_labeled": n,
        "dim": dim,
        "pca": {"n_components": N_PCA, "fit": "train steps<=30 only"},
        "hnsw": {"m": DEFAULT_M, "ef_construction": DEFAULT_EF_CONSTRUCTION, "ef_search": DEFAULT_EF_SEARCH},
        "queries": {"n": len(Q), "k": K},
        "recall_at_10": recall,
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean())},
        "build_s": build_s,
        "footprint": {
            "raw_bytes": raw_bytes(n, dim),
            "index_bytes": idx_path.stat().st_size,
            "pq_bytes": pq_bytes(n, PQ_M),
            "pq_ratio": ratio,
        },
        "pq": {
            "m": PQ_M,
            "recall_at_10": pq_recall,
            "agreement_top10": pq_recall,
            "pr_auc_assumption": "drop assumed 0.0 (no retraining in J3)",
            "should_use_pq_conditional": bool(use_pq),
            "recommendation": "use PQ" if use_pq else "keep float32",
        },
        "index_file": str(idx_path),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "README.md").write_text(
        "# Retrieval v1 — HNSW on real PCA32 embeddings (J3, §3.5)\n\n"
        f"Anchors: all {n} labeled txs → PCA32 (fit on train steps≤30 only, seed {SEED}).\n"
        f"HNSW M={DEFAULT_M} ef_c={DEFAULT_EF_CONSTRUCTION} ef_s={DEFAULT_EF_SEARCH}.\n\n"
        f"- recall@10 vs brute on {len(Q)} queries (seed {SEED}): {recall:.4f}\n"
        f"- query latency: p50 {metrics['latency_ms']['p50']:.3f} ms, "
        f"p99 {metrics['latency_ms']['p99']:.3f} ms\n"
        f"- footprint: raw {metrics['footprint']['raw_bytes'] / 1e6:.2f} MB, "
        f"index {metrics['footprint']['index_bytes'] / 1e6:.2f} MB "
        f"(`{idx_path}`, {metrics['footprint']['index_bytes']} bytes)\n\n"
        "## PQ (m=16, sklearn-KMeans codebooks)\n\n"
        f"- recall@10 after compression: {pq_recall:.4f} "
        f"(agreement top-10 = same metric)\n"
        f"- footprint ratio float32→PQ: {ratio:.1f}x "
        f"({metrics['footprint']['pq_bytes']} bytes codes only)\n"
        "- Assumption: ΔPR-AUC assumed 0.0 — honest measurement needs GBDT "
        "retraining on PQ vectors (out of J3 scope). Conditional recommendation "
        f"via `should_use_pq(0, 0, recall)`: **{metrics['pq']['recommendation']}**.\n"
    )
    print(json.dumps({k: metrics[k] for k in ("recall_at_10", "latency_ms", "footprint", "pq")}, indent=2))


if __name__ == "__main__":
    main()
