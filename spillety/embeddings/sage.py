from typing import Any

_INSTALL = 'pip install "spillety[graphsage]" (needs torch>=2.0, torch-geometric>=2.3)'

NEIGHBOR_SAMPLES = (25, 10)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError(f"torch is required for GraphSAGE encode; install via {_INSTALL}") from e
    try:
        from torch_geometric.nn import SAGEConv
        from torch_geometric.utils import to_undirected
    except ImportError as e:
        raise ImportError(
            f"torch-geometric is required for GraphSAGE encode; install via {_INSTALL}"
        ) from e
    return torch, SAGEConv, to_undirected


def _sample_layer_edges(
    torch: Any, edge_index: Any, n_nodes: int, fanout: int, gen: Any
) -> Any:
    dst = edge_index[1]
    order = torch.randperm(edge_index.shape[1], generator=gen)
    counts = torch.zeros(n_nodes, dtype=torch.long)
    keep = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for e in order.tolist():
        d = int(dst[e])
        if counts[d] < fanout:
            keep[e] = True
            counts[d] += 1
    return edge_index[:, keep]


def encode(
    x: Any,
    edge_index: Any,
    hidden_dim: int = 32,
    out_dim: int = 16,
    seed: int = 72,
) -> Any:
    """
    ## Two-layer GraphSAGE with mean aggregation (§3.3.2)

    Parameters
    ----------
    x : array-like
        Node features (N, in_dim), numpy array or CPU tensor.
    edge_index : array-like
        Edge list (2, E), numpy array or CPU tensor.
    hidden_dim : int
        First-layer width.
    out_dim : int
        Embedding width.
    seed : int
        Seed for neighbor sampling and init; CPU only.

    Returns
    ----------
    torch.Tensor
        Embeddings (N, out_dim) on CPU.
    """
    torch, SAGEConv, to_undirected = _require_torch()
    gen = torch.Generator().manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    ei = to_undirected(torch.as_tensor(edge_index, dtype=torch.long))
    n_nodes = xt.shape[0]
    torch.manual_seed(seed)
    conv1 = SAGEConv(xt.shape[1], hidden_dim, aggr="mean")
    conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
    e1 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[0], gen)
    e2 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[1], gen)
    with torch.no_grad():
        h = torch.relu(conv1(xt, e1))
        return conv2(h, e2)


def encode_gated(
    x: Any,
    edge_index: Any,
    hidden_dim: int = 32,
    out_dim: int = 16,
    seed: int = 72,
) -> Any:
    """
    ## Feature-gated GraphSAGE: residual pathway + gated fusion (FG-EGCN style)

    Adds a learned gate that balances raw node features against aggregated
    neighbourhood features. Prevents feature washout during message passing.

    g = sigmoid(W_r @ x_raw + W_a @ x_agg + b)
    out = g * project(x_raw) + (1 - g) * x_agg

    Parameters
    ----------
    x : array-like
        Node features (N, in_dim).
    edge_index : array-like
        Edge list (2, E).
    hidden_dim : int
        First-layer width.
    out_dim : int
        Embedding width.
    seed : int
        Seed; CPU only.

    Returns
    ----------
    torch.Tensor
        Gated embeddings (N, out_dim) on CPU.
    """
    torch, SAGEConv, to_undirected = _require_torch()
    gen = torch.Generator().manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    ei = to_undirected(torch.as_tensor(edge_index, dtype=torch.long))
    n_nodes = xt.shape[0]
    in_dim = xt.shape[1]
    torch.manual_seed(seed)

    conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
    conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")

    gate_w_raw = torch.randn(in_dim, out_dim) * (in_dim**-0.5)
    gate_w_agg = torch.randn(out_dim, out_dim) * (out_dim**-0.5)
    gate_b = torch.zeros(out_dim)
    proj_raw = torch.randn(in_dim, out_dim) * (in_dim**-0.5)

    e1 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[0], gen)
    e2 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[1], gen)

    with torch.no_grad():
        h = torch.relu(conv1(xt, e1))
        x_agg = conv2(h, e2)

        x_raw_proj = xt @ proj_raw
        gate = torch.sigmoid(xt @ gate_w_raw + x_agg @ gate_w_agg + gate_b)
        return gate * x_raw_proj + (1 - gate) * x_agg
