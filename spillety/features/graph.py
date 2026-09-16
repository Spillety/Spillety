import networkx as nx
import pandas as pd


def compute_graph_features(
    edgelist: pd.DataFrame,
    tx_ids: pd.Series | None = None,
    source_col: str = "txId1",
    target_col: str = "txId2",
) -> pd.DataFrame:
    """
    ## Ego-graph features from Elliptic edgelist (notebook 02)

    Parameters
    ----------
    edgelist : DataFrame
        Must contain `source_col` and `target_col` (default txId1/txId2).
    tx_ids : Series | None
        All txIds to include as nodes (isolated nodes would be lost from edgelist alone).
        If None, uses union of source+target.
    source_col, target_col : str
        Column names for edge endpoints.

    Returns
    ----------
    DataFrame
        Columns [txId, in_degree, out_degree, total_degree, pagerank, in_out_ratio]
        One row per txId, no NaNs.
    """
    G = nx.from_pandas_edgelist(edgelist, source=source_col, target=target_col, create_using=nx.DiGraph())

    if tx_ids is not None:
        # isolated txs (degree==0) must stay as nodes — otherwise left join produces NaN
        G.add_nodes_from(tx_ids.unique())
    else:
        # fallback: union of edgelist endpoints — isolated nodes are lost
        pass

    in_deg_map = dict(G.in_degree())
    out_deg_map = dict(G.out_degree())

    # PageRank is global (O(N+M) per iter) — power iteration, 100 iters enough for sparse BTC graph (density ~1e-5)
    # ponytail: for N>1M switch to nx.pagerank with personalization or approximate via sampling
    pagerank_map = nx.pagerank(G, alpha=0.85, max_iter=100)

    # isolated nodes have no entry in degree maps until add_nodes_from; still fillna 0 below as safety
    nodes = list(G.nodes())
    out = pd.DataFrame({"txId": nodes})
    out["in_degree"] = out["txId"].map(in_deg_map).fillna(0).astype(int)
    out["out_degree"] = out["txId"].map(out_deg_map).fillna(0).astype(int)
    out["total_degree"] = out["in_degree"] + out["out_degree"]
    # fillna 0: isolated nodes get 0 pagerank after mapping if graph has nodes not in map (edge case: empty graph)
    out["pagerank"] = out["txId"].map(pagerank_map).fillna(0.0)
    # +1 smoothing avoids div-by-zero and shrinks ratio for low-degree nodes (Laplace)
    out["in_out_ratio"] = (out["in_degree"] + 1) / (out["out_degree"] + 1)

    # PCA not needed here: 5 features only, degree and pagerank are on different scales but kept raw for tree models;
    # linear models should scale via StandardScaler downstream.
    # top 5% degree ≈ Exchange_Hot proxy — high-fan-out hubs correlate with exchange wallets in Elliptic EDA (heavy-tail)

    return out


def add_graph_features(
    df: pd.DataFrame,
    edgelist: pd.DataFrame,
    tx_col: str = "txId",
) -> pd.DataFrame:
    """
    ## Left join graph features onto transaction table

    Parameters
    ----------
    df : DataFrame
        Must contain `tx_col`.
    edgelist : DataFrame
        Edge list for graph construction.
    tx_col : str
        Join key.

    Returns
    ----------
    DataFrame
        Copy of `df` enriched with 5 graph columns, NaNs filled with 0.
    """
    g = compute_graph_features(edgelist, tx_ids=df[tx_col])
    out = df.merge(g, on="txId", how="left")
    # edge case: df has txIds not in G (should not happen via add_nodes_from, but safe for empty edgelist)
    for c in ["in_degree", "out_degree", "total_degree", "pagerank", "in_out_ratio"]:
        out[c] = out[c].fillna(0)
    return out


if __name__ == "__main__":
    from pathlib import Path

    from spillety.data.loader import load_elliptic

    root = Path("data/elliptic_raw")
    if not root.exists():
        # synthetic smoke test when data is missing (CI without archive)
        edgelist = pd.DataFrame({"txId1": [1, 1, 2], "txId2": [2, 3, 3]})
        df = pd.DataFrame({"txId": [1, 2, 3, 4], "time_step": [1, 1, 2, 2]})
        feat = compute_graph_features(edgelist, tx_ids=df["txId"])
        assert feat.shape == (4, 6), feat.shape
        assert not feat[["in_degree", "out_degree", "pagerank", "in_out_ratio"]].isna().any().any()
        enriched = add_graph_features(df, edgelist)
        assert enriched.shape[0] == df.shape[0]
        assert not enriched[["pagerank"]].isna().any().any()
        print("graph smoke test (synthetic) passed:", feat.shape)
    else:
        features, classes, edgelist, merged = load_elliptic(root)
        feat = compute_graph_features(edgelist, tx_ids=features["txId"])
        assert feat.shape[0] == features["txId"].nunique(), (feat.shape, features["txId"].nunique())
        assert not feat[["in_degree", "out_degree", "total_degree", "pagerank", "in_out_ratio"]].isna().any().any()
        enriched = add_graph_features(merged.head(1000), edgelist)
        assert enriched.shape[0] == 1000
        assert not enriched[["pagerank", "in_degree"]].isna().any().any()
        print("graph demo passed:", feat.shape, "pagerank sum", feat["pagerank"].sum())
