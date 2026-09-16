import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


def _build_context(edgelist, df=None, time_col="time_step", source_col="txId1", target_col="txId2"):
    G = nx.from_pandas_edgelist(edgelist, source=source_col, target=target_col, create_using=nx.Graph())
    if df is not None and isinstance(df, pd.DataFrame) and "txId" in df.columns:
        G.add_nodes_from(df["txId"].unique())
    # undirected edge lookup O(1)
    edge_set = set()
    for a, b in zip(edgelist[source_col].values, edgelist[target_col].values):
        edge_set.add((a, b))
        edge_set.add((b, a))

    if isinstance(df, pd.DataFrame) and time_col in df.columns:
        tx_time = dict(zip(df["txId"], df[time_col]))
    elif isinstance(df, dict):
        tx_time = df
    elif isinstance(df, pd.Series):
        tx_time = df.to_dict()
    else:
        tx_time = {}

    deg_map = dict(G.degree())
    max_deg = max(deg_map.values()) if deg_map else 1
    if max_deg == 0:
        max_deg = 1
    return G, edge_set, tx_time, deg_map, max_deg


def compute_signals(edgelist, df=None, time_col="time_step", pairs=None, source_col="txId1", target_col="txId2"):
    if isinstance(df, str) and pairs is None:
        time_col, df = df, None
    if isinstance(time_col, pd.DataFrame) and df is None:
        df, time_col = time_col, "time_step"
    G, edge_set, tx_time, deg_map, max_deg = _build_context(edgelist, df, time_col, source_col, target_col)

    if pairs is None:
        pairs = list(zip(edgelist[source_col].values, edgelist[target_col].values))

    if isinstance(pairs, pd.DataFrame):
        if {"tx_a", "tx_b"}.issubset(pairs.columns):
            pairs = list(zip(pairs["tx_a"].values, pairs["tx_b"].values))
        elif {source_col, target_col}.issubset(pairs.columns):
            pairs = list(zip(pairs[source_col].values, pairs[target_col].values))
        else:
            cols = pairs.columns.tolist()[:2]
            pairs = list(zip(pairs[cols[0]].values, pairs[cols[1]].values))

    rows = []
    for a, b in pairs:
        cioh = 1 if (a, b) in edge_set else 0
        ta, tb = tx_time.get(a), tx_time.get(b)
        temporal = 1 if (ta is not None and tb is not None and abs(int(ta) - int(tb)) <= 1) else 0
        da, db = deg_map.get(a, 0), deg_map.get(b, 0)
        # max_deg smoothing avoids div-by-zero and normalizes similarity to [0,1]
        deg_sim = 1 - abs(da - db) / max_deg
        rows.append((a, b, cioh, temporal, float(deg_sim)))
    return pd.DataFrame(rows, columns=["tx_a", "tx_b", "cioh", "temporal", "deg_sim"])


def fuse_signals(signals, labels=None):
    if isinstance(signals, pd.DataFrame):
        feat_cols = [c for c in ["cioh", "temporal", "deg_sim"] if c in signals.columns]
        if not feat_cols:
            feat_cols = signals.columns.tolist()[:3]
        X = signals[feat_cols].values.astype(float)
        if labels is None:
            for c in ["label", "y", "same_comp", "same_class"]:
                if c in signals.columns:
                    labels = signals[c].values
                    break
    else:
        X = np.asarray(signals, dtype=float)
    if labels is None:
        raise ValueError("labels required: pass labels array or DataFrame with label column")
    y = np.asarray(labels).astype(int)
    # ponytail: no copula, upgrade to Clayton/Gumbel via scipy.optimize MLE when tail dependence matters
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=72)
    clf.fit(X, y)
    return clf


def cluster_union_find(edgelist, tx_ids=None, source_col="txId1", target_col="txId2"):
    # ponytail: no copula, single Graph — for directed use weakly connected via Graph proxy
    G = nx.from_pandas_edgelist(edgelist, source=source_col, target=target_col, create_using=nx.Graph())
    if tx_ids is not None:
        if isinstance(tx_ids, pd.Series):
            tx_ids = tx_ids.unique()
        G.add_nodes_from(tx_ids)
    components = list(nx.connected_components(G))
    comp_map = {}
    for cid, comp in enumerate(components):
        for tx in comp:
            comp_map[tx] = cid
    return components, comp_map


def evaluate_clusters(y_true, y_pred=None, zero_division=0):
    if isinstance(y_true, pd.DataFrame) and y_pred is None:
        if {"y_true", "y_pred"}.issubset(y_true.columns):
            y_true_arr = y_true["y_true"].values
            y_pred_arr = y_true["y_pred"].values
        elif {"true", "pred"}.issubset(y_true.columns):
            y_true_arr = y_true["true"].values
            y_pred_arr = y_true["pred"].values
        else:
            cols = y_true.columns.tolist()
            y_true_arr = y_true[cols[0]].values
            y_pred_arr = y_true[cols[1]].values
        y_true, y_pred = y_true_arr, y_pred_arr
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_division)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_division)),
        "f1": float(f1_score(y_true, y_pred, zero_division=zero_division)),
    }


if __name__ == "__main__":
    from pathlib import Path

    # synthetic graph 5 nodes: edges 1-2, 2-3, 4-5 → 2 components + isolated handling
    edgelist = pd.DataFrame({"txId1": [1, 2, 4], "txId2": [2, 3, 5]})
    df_time = pd.DataFrame({"txId": [1, 2, 3, 4, 5], "time_step": [1, 1, 2, 10, 10]})

    # cluster check
    comps, cmap = cluster_union_find(edgelist, tx_ids=df_time["txId"])
    assert len(comps) == 2, len(comps)
    assert cmap[1] == cmap[2] == cmap[3]
    assert cmap[4] == cmap[5]
    assert cmap[1] != cmap[4]

    # signals: positive pair (1,2) edge+temporal close, negative (1,4) disconnected + time far
    pairs = [(1, 2), (1, 3), (1, 4), (4, 5), (2, 4)]
    sig = compute_signals(edgelist, df_time, time_col="time_step", pairs=pairs)
    assert sig.shape == (5, 5), sig.shape
    assert sig.loc[0, "cioh"] == 1  # 1-2 edge
    assert sig.loc[2, "cioh"] == 0  # 1-4 no edge
    assert sig.loc[0, "temporal"] == 1  # both time 1
    assert sig.loc[2, "temporal"] == 0  # 1 vs 10
    assert sig["deg_sim"].between(0, 1).all()

    # fusion: labels 1 if same component else 0 → logistic should fit
    labels = np.array([1 if cmap[a] == cmap[b] else 0 for a, b in pairs])
    clf = fuse_signals(sig, labels)
    proba = clf.predict_proba(sig[["cioh", "temporal", "deg_sim"]].values)[:, 1]
    assert proba.shape == labels.shape
    assert (proba >= 0).all() and (proba <= 1).all()

    # pair-level eval
    pred = (proba >= 0.5).astype(int)
    m = evaluate_clusters(labels, pred)
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"] <= 1

    # smoke on real edgelist if present
    root = Path("data/elliptic_raw")
    if not root.exists():
        root = Path(__file__).resolve().parents[2] / "data/elliptic_raw"
    if root.exists():
        from spillety.data.loader import load_elliptic

        features, classes, elist_real, merged = load_elliptic(root)
        # sample 200 random pairs for smoke
        rng = np.random.default_rng(72)
        all_tx = features["txId"].values
        samp_pairs = [tuple(rng.choice(all_tx, 2, replace=False)) for _ in range(200)]
        sig_real = compute_signals(elist_real, features, time_col="time_step", pairs=samp_pairs)
        assert sig_real.shape == (200, 5)
        assert sig_real["cioh"].isin([0, 1]).all()
        # quick fusion on synthetic labels from components (sanity: no crash)
        comps_r, cmap_r = cluster_union_find(elist_real, tx_ids=features["txId"])
        y_r = np.array([1 if cmap_r.get(a, -1) == cmap_r.get(b, -2) else 0 for a, b in samp_pairs])
        clf_r = fuse_signals(sig_real, y_r)
        assert clf_r.coef_.shape == (1, 3)
        # recall@ smoke via components vs random baseline: just check precision call
        m_r = evaluate_clusters(y_r, (clf_r.predict_proba(sig_real[["cioh", "temporal", "deg_sim"]].values)[:, 1] >= 0.5).astype(int))
        assert 0 <= m_r["f1"] <= 1
        print("real smoke passed:", sig_real.shape, "comps", len(comps_r), "f1", round(m_r["f1"], 3))
    print("entity resolution smoke passed:", sig.shape, "comps", len(comps))
