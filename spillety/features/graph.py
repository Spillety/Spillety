import networkx as nx
import pandas as pd


def compute_graph_features(
    edgelist: pd.DataFrame,
    tx_ids: pd.Series | None = None,
    source_col: str = "txId1",
    target_col: str = "txId2",
) -> pd.DataFrame:
    """
    ## Ego-graph features from edgelist (§7.3.3)

    Parameters
    ----------
    edgelist : DataFrame
        Must contain `source_col` and `target_col`.
    tx_ids : Series | None
        Nodes to include; isolated nodes are lost from edgelist alone.
    source_col, target_col : str
        Edge endpoint columns.

    Returns
    ----------
    DataFrame
        One row per node, no NaNs.
    """
    G = nx.from_pandas_edgelist(edgelist, source=source_col, target=target_col, create_using=nx.DiGraph())

    if tx_ids is not None:
        G.add_nodes_from(tx_ids.unique())

    in_deg_map = dict(G.in_degree())
    out_deg_map = dict(G.out_degree())

    # ponytail: for N>1M switch to approximate pagerank via sampling
    pagerank_map = nx.pagerank(G, alpha=0.85, max_iter=100)

    nodes = list(G.nodes())
    out = pd.DataFrame({"txId": nodes})
    out["in_degree"] = out["txId"].map(in_deg_map).fillna(0).astype(int)
    out["out_degree"] = out["txId"].map(out_deg_map).fillna(0).astype(int)
    out["total_degree"] = out["in_degree"] + out["out_degree"]
    out["pagerank"] = out["txId"].map(pagerank_map).fillna(0.0)
    out["in_out_ratio"] = (out["in_degree"] + 1) / (out["out_degree"] + 1)

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
        Copy of `df` with 5 graph columns, NaNs filled with 0.
    """
    g = compute_graph_features(edgelist, tx_ids=df[tx_col])
    out = df.merge(g, left_on=tx_col, right_on="txId", how="left")
    for c in ["in_degree", "out_degree", "total_degree", "pagerank", "in_out_ratio"]:
        out[c] = out[c].fillna(0)
    return out
