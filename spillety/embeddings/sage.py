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
    # Keep the first `fanout` shuffled edges per destination: uniform neighbor sample.
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
    # ponytail: full-tensor forward; for N > 1M switch to NeighborLoader minibatches
    conv1 = SAGEConv(xt.shape[1], hidden_dim, aggr="mean")
    conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
    e1 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[0], gen)
    e2 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[1], gen)
    with torch.no_grad():
        h = torch.relu(conv1(xt, e1))
        return conv2(h, e2)
