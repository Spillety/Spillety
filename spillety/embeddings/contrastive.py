import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ponytail: GraphSAGE torch_geometric — 2-layer mean aggregator, hidden 128, NT-Xent tau=0.1 + anchor margin, HNSW M=24


def encode_pca(X, n_components=32, random_state=72):
    """
    GraphSAGE proxy: StandardScaler + PCA 165→32, L2-normalized for cosine.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n_comp = min(n_components, min(X.shape[0], X.shape[1]))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_comp, random_state=random_state)
    Z = pca.fit_transform(Xs)
    # pad if requested dim > feasible (e.g. 256 but feat 165) — keep compat
    if Z.shape[1] < n_components:
        pad = np.zeros((Z.shape[0], n_components - Z.shape[1]))
        Z = np.hstack([Z, pad])
    # L2 normalize for cosine (as in NT-Xent)
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    Zn = Z / norms
    return Zn, pca, scaler


def _l2_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def nt_xent_loss(embeddings, pos_pairs, tau=0.1):
    """
    NT-Xent proxy on normalized embeddings.
    embeddings: (N,d) L2-normalized, pos_pairs: array (P,2) indices, tau: temperature
    """
    # tau=0.1 sharpens distribution — hard negatives penalized stronger, optimum vs 0.05 too sharp and 0.2 too smooth (notebook 10 tradeoff)
    embeddings = np.asarray(embeddings)
    pos_pairs = np.asarray(pos_pairs)
    if len(pos_pairs) == 0:
        return 0.0
    # ensure normalized
    En = _l2_normalize(embeddings)
    # similarity matrix for pos anchors vs all embeddings (in-batch negatives)
    # simplified: denominator = sum_{k != i} exp(sim(i,k)/tau)
    sim_mat = En @ En.T  # cosine
    losses = []
    for i, j in pos_pairs:
        i, j = int(i), int(j)
        if i >= len(En) or j >= len(En):
            continue
        pos_sim = sim_mat[i, j] / tau
        # denominator over all k != i
        logits = sim_mat[i] / tau
        # exclude self
        mask = np.ones(len(En), dtype=bool)
        mask[i] = False
        logits_m = logits[mask]
        m = np.max(logits_m)
        logsumexp = m + np.log(np.sum(np.exp(logits_m - m)))
        # also include pos in denominator already via logits_m if j != i; if we excluded self only, pos is inside
        # loss = -log( exp(pos)/ sum exp )
        losses.append(-(pos_sim - logsumexp))
    return float(np.mean(losses)) if losses else 0.0


def batch_nt_xent(Z_a, Z_b, is_positive, tau=0.1):
    """Helper matching notebook 10 batch layout: Z_a/Z_b paired, is_positive mask."""
    Z_a = _l2_normalize(np.asarray(Z_a))
    Z_b = _l2_normalize(np.asarray(Z_b))
    sim_mat = Z_a @ Z_b.T
    losses = []
    for idx in range(len(Z_a)):
        if not is_positive[idx]:
            continue
        pos_sim = sim_mat[idx, idx]
        logits = sim_mat[idx] / tau
        m = np.max(logits)
        logsumexp = m + np.log(np.sum(np.exp(logits - m)))
        losses.append(-(pos_sim / tau - logsumexp))
    losses = np.array(losses)
    return float(np.mean(losses)) if len(losses) else 0.0, losses


def anchor_loss(anchors, nonanchors, margin=0.5, jaccard=0.3):
    """
    Anchor loss proxy: gap between anchor-anchor and anchor-nonanchor cosine distances.
    anchors: (Na,d) illicit, nonanchors: (Nn,d) licit, assumed L2-normalized or will be normalized.
    Returns dict with gap and loss.
    """
    anchors = _l2_normalize(np.asarray(anchors))
    nonanchors = _l2_normalize(np.asarray(nonanchors))
    if len(anchors) < 2 or len(nonanchors) == 0:
        return {"gap": 0.0, "loss": 0.0, "weighted_margin": 0.0}
    # weighted margin имитирует OFAC/EU overlap: w=0.3+0.7*J, J_mock=0.3
    w = 0.3 + 0.7 * float(jaccard)
    weighted_margin = margin * w
    # cosine distance = 1 - cos
    # sample to avoid O(N^2) blowup
    rng = np.random.default_rng(72)
    n_a = min(800, len(anchors))
    n_n = min(2000, len(nonanchors))
    idx_a = rng.choice(len(anchors), size=n_a, replace=False)
    idx_n = rng.choice(len(nonanchors), size=n_n, replace=False)
    Za = anchors[idx_a]
    Zn = nonanchors[idx_n]
    # pairwise cosine distance via dot (since normalized)
    # D_aa upper triangle mean
    sim_aa = Za @ Za.T
    dist_aa = 1 - sim_aa
    triu = dist_aa[np.triu_indices(n_a, k=1)]
    sim_an = Za @ Zn.T
    dist_an = 1 - sim_an
    mean_aa = float(triu.mean()) if len(triu) else 0.0
    mean_an = float(dist_an.mean())
    gap = mean_an - mean_aa
    # proxy loss: max(0, mean_aa - mean_an + weighted_margin)
    loss = float(max(0.0, mean_aa - mean_an + weighted_margin))
    return {"gap": gap, "loss": loss, "weighted_margin": weighted_margin, "mean_aa": mean_aa, "mean_an": mean_an}


def evaluate_silhouette(embeddings, labels, metric="cosine", sample_size=None):
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    if sample_size is not None and len(embeddings) > sample_size:
        rng = np.random.default_rng(72)
        idx = rng.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    # need at least 2 classes
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(embeddings, labels, metric=metric))


def evaluate_recall(embeddings_pool, labels_pool, embeddings_query, k=10, metric="cosine"):
    embeddings_pool = np.asarray(embeddings_pool)
    embeddings_query = np.asarray(embeddings_query)
    labels_pool = np.asarray(labels_pool)
    if len(embeddings_query) == 0 or len(embeddings_pool) == 0:
        return 0.0
    nn = NearestNeighbors(n_neighbors=min(k, len(embeddings_pool)), metric=metric)
    nn.fit(embeddings_pool)
    _, idx = nn.kneighbors(embeddings_query, n_neighbors=min(k, len(embeddings_pool)))
    hits = 0
    for row in idx:
        if np.any(labels_pool[row] == 1):
            hits += 1
    return hits / len(embeddings_query)


# alias for spec naming
evaluate_recall_at_k = evaluate_recall

if __name__ == "__main__":
    # synthetic smoke
    rng = np.random.default_rng(72)
    X_syn = rng.standard_normal((500, 165))
    # add illicit cluster shift
    X_syn[:50] += 2.0
    y_syn = np.array([1] * 50 + [0] * 450)
    Zn, pca, _ = encode_pca(X_syn, n_components=32)
    assert Zn.shape == (500, 32)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"synthetic encode_pca 32 var={var:.2%} Zn {Zn.shape}")
    # pos_pairs: illicit cluster positives
    pos_pairs = np.array([[i, i + 1] for i in range(0, 40, 2)])
    loss = nt_xent_loss(Zn, pos_pairs, tau=0.1)
    print(f"synthetic nt_xent tau=0.1 loss={loss:.4f}")
    for tau in [0.05, 0.1, 0.2]:
        l, _ = batch_nt_xent(Zn[pos_pairs[:, 0]], Zn[pos_pairs[:, 1]], np.ones(len(pos_pairs), dtype=bool), tau=tau)
        print(f"  tau={tau:.2f} batch loss={l:.4f}")
    anchors = Zn[y_syn == 1]
    nonanchors = Zn[y_syn == 0]
    a_res = anchor_loss(anchors, nonanchors)
    print(f"anchor gap={a_res['gap']:.4f} loss={a_res['loss']:.4f} margin={a_res['weighted_margin']:.3f}")
    sil = evaluate_silhouette(Zn, y_syn, metric="cosine")
    rec = evaluate_recall(Zn, y_syn, Zn[y_syn == 1][:10], k=10)
    print(f"silhouette={sil:.4f} recall@10={rec:.4f}")
    assert -1 <= sil <= 1
    assert 0 <= rec <= 1

    # smoke real if data present
    from pathlib import Path

    root = Path("data/elliptic_raw")
    if root.exists():
        from spillety.data.loader import load_elliptic, temporal_split
        from sklearn.preprocessing import StandardScaler

        features, classes, edgelist, merged = load_elliptic(root)
        df = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
        df["y"] = (df["class"].astype(str) == "1").astype(int)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        train_df, valid_df, test_df = temporal_split(df, train_end=30, valid_end=40)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feat_cols].values)
        X_test = scaler.transform(test_df[feat_cols].values)
        y_train = train_df["y"].values
        y_test = test_df["y"].values
        # proper shared fit: fit PCA on train only, transform test
        pca_real = PCA(n_components=32, random_state=72)
        Z_train_raw = pca_real.fit_transform(X_train)
        Z_test_raw = pca_real.transform(X_test)
        var32 = float(pca_real.explained_variance_ratio_.sum())
        print(f"real PCA32 var={var32:.2%} (target ~78% proxy 32)")
        # L2 normalize for cosine
        Zn_train = Z_train_raw / (np.linalg.norm(Z_train_raw, axis=1, keepdims=True) + 1e-9)
        Zn_test = Z_test_raw / (np.linalg.norm(Z_test_raw, axis=1, keepdims=True) + 1e-9)
        sil_real = evaluate_silhouette(np.vstack([Zn_train, Zn_test])[:5000], np.concatenate([y_train, y_test])[:5000], metric="cosine", sample_size=5000)
        print(f"real silhouette 5k={sil_real:.4f}")
        rec10 = evaluate_recall(Zn_train, y_train, Zn_test[y_test == 1][:100], k=10)
        print(f"real recall@10 illicit->train {rec10:.4f}")
    else:
        print("real smoke skipped: data/elliptic_raw not found")
