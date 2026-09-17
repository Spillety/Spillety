import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

_GLOBAL_INDEX = None


def estimate_memory(n, dim, bytes_per=4):
    return int(n * dim * bytes_per)


def build_index(embeddings, algorithm="auto", metric="euclidean", n_neighbors=10):
    global _GLOBAL_INDEX
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {X.shape}")
    # ponytail: kd_tree/ball_tree as HNSW proxy, upgrade to hnswlib M=24 ef=128 when N>100k
    # brute O(N*d) exact, kd_tree O(log N) proxy — both via sklearn, no hnswlib dependency
    n_neighbors = min(n_neighbors, len(X))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
    nn.fit(X)
    _GLOBAL_INDEX = nn
    return nn


def query(*args, k=10):
    if len(args) == 2 and isinstance(args[0], NearestNeighbors):
        index, queries = args[0], args[1]
    elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], int):
        # query(queries, k) with global index
        queries, k = args[0], args[1]
        index = _GLOBAL_INDEX
    elif len(args) == 1:
        queries = args[0]
        index = _GLOBAL_INDEX
    elif len(args) == 2:
        # fallback: first is queries array, second is k via positional
        if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            index = _GLOBAL_INDEX
            queries = args[0]
            k = 10
        else:
            # assume query(index, queries)
            index, queries = args
            if index is None:
                index = _GLOBAL_INDEX
    else:
        raise ValueError("query(index, queries, k) or query(queries, k) with built index")
    if index is None:
        raise ValueError("no index built: call build_index first")
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    k = min(k, len(index._fit_X))
    dist, idx = index.kneighbors(queries, n_neighbors=k)
    return dist, idx


def evaluate_recall(approx_idx, exact_idx, k=10):
    approx_idx = np.asarray(approx_idx)
    exact_idx = np.asarray(exact_idx)
    if approx_idx.shape[0] != exact_idx.shape[0]:
        raise ValueError("approx and exact must have same n_queries")
    k = min(k, approx_idx.shape[1], exact_idx.shape[1])
    n = approx_idx.shape[0]
    rec = 0.0
    for i in range(n):
        rec += len(set(approx_idx[i, :k]) & set(exact_idx[i, :k])) / k
    return float(rec / n) if n else 0.0


if __name__ == "__main__":
    from pathlib import Path

    # synthetic 5 vectors 3d → brute and kd_tree should give recall 1.0
    rng = np.random.default_rng(72)
    X = rng.standard_normal((5, 3)).astype(np.float32)
    queries = rng.standard_normal((2, 3)).astype(np.float32)

    brute = build_index(X, algorithm="brute", metric="euclidean", n_neighbors=3)
    d_b, idx_b = query(brute, queries, k=2)
    assert idx_b.shape == (2, 2), idx_b.shape
    assert d_b.shape == (2, 2)

    approx = build_index(X, algorithm="kd_tree", metric="euclidean", n_neighbors=3)
    # kd_tree as HNSW proxy — on 5 points should be exact
    d_a, idx_a = query(approx, queries, k=2)
    assert evaluate_recall(idx_a, idx_b, k=2) == 1.0

    mem = estimate_memory(5, 3)
    assert mem == 60, mem
    assert estimate_memory(80_000_000, 128) == 80_000_000 * 128 * 4

    # smoke on real edgelist: tabular embeddings 165d → scaled, anchors=train illicit
    root = Path("data/elliptic_raw")
    if not root.exists():
        root = Path(__file__).resolve().parents[2] / "data/elliptic_raw"
    if root.exists():
        from sklearn.preprocessing import StandardScaler

        from spillety.data.loader import load_elliptic, temporal_split

        _, _, _, merged = load_elliptic(root)
        df = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
        df["y"] = (df["class"].astype(str) == "1").astype(int)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        train_df, _, test_df = temporal_split(df, train_end=30, valid_end=40)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feat_cols].values).astype(np.float32)
        X_test = scaler.transform(test_df[feat_cols].values).astype(np.float32)
        y_train = train_df["y"].values
        X_anchors = X_train[y_train == 1]
        # cap for speed
        X_anchors = X_anchors[:2000]
        X_q = X_test[:200]

        brute2 = build_index(X_anchors, algorithm="brute", metric="euclidean", n_neighbors=10)
        d2, idx2 = query(brute2, X_q, k=10)
        approx2 = build_index(X_anchors, algorithm="auto", metric="euclidean", n_neighbors=10)
        d3, idx3 = query(approx2, X_q, k=10)
        rec10 = evaluate_recall(idx3, idx2, k=10)
        assert 0 <= rec10 <= 1
        # on this scale proxy is near-exact
        assert rec10 >= 0.9, f"recall@10 {rec10}"
        mem_mb = estimate_memory(len(X_anchors), X_anchors.shape[1]) / (1024**2)
        print(f"retrieval real smoke: anchors {X_anchors.shape} queries {X_q.shape} recall@10={rec10:.3f} mem={mem_mb:.2f}MB")
    print("retrieval smoke passed: recall exact on synthetic", evaluate_recall(idx_a, idx_b, k=2))
