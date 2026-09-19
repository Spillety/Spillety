import numpy as np

EDGE_TYPES = ("co_spend", "addr_tx", "tx_addr")


def _as_flags(n, *seqs):
    cols = []
    for seq in seqs:
        if seq is None:
            cols.append([False] * n)
        else:
            vals = [bool(v) for v in seq]
            if len(vals) != n:
                raise ValueError(f"flag length {len(vals)} != transactions {n}")
            cols.append(vals)
    return list(zip(*cols)) if cols else [() for _ in range(n)]


def build_account_graph(
    transactions,
    *,
    is_coinjoin=None,
    is_exchange_hot=None,
    address_ids=None,
):
    """
    ## Heterogeneous address <-> tx graph with CIOH co-spend edges 

    Parameters
    ----------
    transactions : Sequence[Sequence[Hashable]]
        One entry per transaction, listing its input addresses.
    is_coinjoin : Sequence[bool] | None
        Flagged transactions contribute addr↔tx edges but no co-spend edges.
    is_exchange_hot : Sequence[bool] | None
        Same exclusion as `is_coinjoin`.
    address_ids : Sequence[Hashable] | None
        Fixed address ordering; defaults to sorted repr order.

    Returns
    ----------
    dict
        Keys `address_ids`, `n_addr`, `n_tx`, `edges` mapping each of
        `EDGE_TYPES` to an int64 array (2, E).
    
    See Also
    ----------
        §3.3, §5.2.1
    """
    n_tx = len(transactions)
    flags = _as_flags(n_tx, is_coinjoin, is_exchange_hot)
    addr_set = {a for tx in transactions for a in tx if a is not None}
    ids = list(address_ids) if address_ids is not None else sorted(addr_set, key=repr)
    index = {a: i for i, a in enumerate(ids)}
    co, at, ta = [], [], []
    for t, (tx, (cj, xh)) in enumerate(zip(transactions, flags)):
        members = [index[a] for a in tx if a is not None and a in index]
        for a in members:
            at.append((a, t))
            ta.append((t, a))
        if cj or xh or len(members) < 2:
            continue
        first = members[0]
        for other in members[1:]:
            co.append((first, other))
            co.append((other, first))
    # ponytail: co-spend star only; add change-heuristic edges when §5.2.2 lands.
    edges = {
        "co_spend": np.array(co, dtype=np.int64).T.reshape(2, -1),
        "addr_tx": np.array(at, dtype=np.int64).T.reshape(2, -1),
        "tx_addr": np.array(ta, dtype=np.int64).T.reshape(2, -1),
    }
    return {"address_ids": ids, "n_addr": len(ids), "n_tx": n_tx, "edges": edges}


def _norm_adj(torch, n, edge_index):
    adj = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel():
        adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj + torch.eye(n)
    deg = adj.sum(dim=1)
    d_inv = deg.pow(-0.5)
    d_inv[torch.isinf(d_inv)] = 0.0
    return d_inv[:, None] * adj * d_inv[None, :]


def encode_account_graph(
    x_addr,
    graph,
    hidden_dim=32,
    out_dim=16,
    seed=72,
):
    """
    ## Two-view GCN with attention fusion over address subgraphs (§3.3)

    Parameters
    ----------
    x_addr : array-like
        Address features (Na, in_dim), numpy array or CPU tensor.
    graph : dict
        Output of `build_account_graph`.
    hidden_dim : int
        Width of each view encoder.
    out_dim : int
        Fused embedding width.
    seed : int
        Seed for init; CPU only.

    Returns
    ----------
    tuple[torch.Tensor, torch.Tensor]
        Fused address embeddings (Na, out_dim) and attention weights
        (Na, 2) with rows summing to 1.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required for account-graph encode") from e
    torch.manual_seed(seed)
    xa = torch.as_tensor(np.asarray(x_addr), dtype=torch.float32)
    n_addr, n_tx = graph["n_addr"], graph["n_tx"]
    if xa.shape[0] != n_addr:
        raise ValueError(f"x_addr rows {xa.shape[0]} != n_addr {n_addr}")
    co = torch.as_tensor(graph["edges"]["co_spend"], dtype=torch.long)
    at = torch.as_tensor(graph["edges"]["addr_tx"], dtype=torch.long)
    a_hat = _norm_adj(torch, n_addr, co)
    w_co = torch.randn(xa.shape[1], hidden_dim) * (xa.shape[1] ** -0.5)
    h_co = torch.relu(a_hat @ xa @ w_co)
    if n_tx and at.numel():
        b = torch.zeros((n_addr, n_tx), dtype=torch.float32)
        b[at[0], at[1]] = 1.0
        deg_t = b.sum(dim=0, keepdim=True).clamp_min(1.0)
        h_tx = (b / deg_t).t() @ xa
        deg_a = b.sum(dim=1, keepdim=True).clamp_min(1.0)
        back = (b / deg_a) @ h_tx
    else:
        back = torch.zeros_like(xa)
    w_bi = torch.randn(xa.shape[1], hidden_dim) * (xa.shape[1] ** -0.5)
    h_bi = torch.relu(back @ w_bi)
    # ponytail: single-head additive attention; add multi-head when K1 quality demands.
    q = torch.randn(hidden_dim, 1) * (hidden_dim**-0.5)
    with torch.no_grad():
        s_co = torch.tanh(h_co) @ q
        s_bi = torch.tanh(h_bi) @ q
        attn = torch.softmax(torch.cat([s_co, s_bi], dim=1), dim=1)
        fused = attn[:, [0]] * h_co + attn[:, [1]] * h_bi
        w_out = torch.randn(hidden_dim, out_dim) * (hidden_dim**-0.5)
        return fused @ w_out, attn
