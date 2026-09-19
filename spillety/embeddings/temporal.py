from typing import Any


def _norm_adj(torch, n, edge_index):
    adj = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel():
        adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj + torch.eye(n)
    deg = adj.sum(dim=1)
    d_inv = deg.pow(-0.5)
    d_inv[torch.isinf(d_inv)] = 0.0
    return d_inv[:, None] * adj * d_inv[None, :]


class EvolveGCNEncoder:
    """
    ## EvolveGCN-style encoder: GRU-evolved GCN weights plus salient gate (§3.3)

    Parameters
    ----------
    in_dim : int
        Input feature width.
    hidden_dim : int
        GCN hidden width; also the GRU state width.
    out_dim : int
        Output embedding width.
    """

    def __init__(self, torch, in_dim, hidden_dim=32, out_dim=16):
        self._torch = torch
        scale_in = in_dim**-0.5
        scale_h = hidden_dim**-0.5
        self.w = torch.randn(in_dim, hidden_dim) * scale_in
        self.w_out = torch.randn(hidden_dim, out_dim) * scale_h
        self.proj = torch.randn(in_dim, hidden_dim) * scale_in
        self.gru = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.gate_w = torch.randn(out_dim) * (out_dim**-0.5)
        self.gate_b = torch.zeros(out_dim)

    def parameters(self):
        return [self.w, self.w_out, self.proj, self.gate_w, self.gate_b] + list(
            self.gru.parameters()
        )

    def step(self, x, edge_index):
        """
        ## Single temporal step: evolve weights, propagate, gate (§3.3)

        Parameters
        ----------
        x : torch.Tensor
            Node features (N, in_dim) at this step.
        edge_index : torch.Tensor
            Edge list (2, E) at this step.

        Returns
        ----------
        torch.Tensor
            Gated embeddings (N, out_dim).
        """
        torch = self._torch
        n = x.shape[0]
        # GRU evolves each row of the GCN weight matrix from the pooled graph summary.
        summary = x.mean(dim=0) @ self.proj
        gist = summary.unsqueeze(0).expand(self.w.shape[0], -1)
        self.w = self.gru(gist, self.w.detach())
        a_hat = _norm_adj(torch, n, edge_index)
        h = torch.relu(a_hat @ x @ self.w)
        z = h @ self.w_out
        gate = torch.sigmoid(z * self.gate_w + self.gate_b)
        return gate * z


def encode_temporal(
    xs,
    edge_indices,
    hidden_dim=32,
    out_dim=16,
    seed=72,
) -> Any:
    """
    ## Forward pass over snapshots with GRU-evolved GCN weights (§3.3)

    Parameters
    ----------
    xs : Sequence[array-like]
        Per-step node features, each (N, in_dim); shared node set.
    edge_indices : Sequence[array-like]
        Per-step edge lists, each (2, E_t).
    hidden_dim : int
        GCN hidden width.
    out_dim : int
        Output embedding width.
    seed : int
        Seed for init; CPU only.

    Returns
    ----------
    torch.Tensor
        Stacked embeddings (T, N, out_dim) on CPU.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required for temporal encode") from e
    if len(xs) != len(edge_indices):
        raise ValueError(f"len(xs) {len(xs)} != len(edges) {len(edge_indices)}")
    if not len(xs):
        raise ValueError("xs must be non-empty")
    torch.manual_seed(seed)
    in_dim = torch.as_tensor(xs[0], dtype=torch.float32).shape[1]
    enc = EvolveGCNEncoder(torch, in_dim, hidden_dim, out_dim)
    # ponytail: full-batch per step; add NeighborLoader windows when T*N exceeds RAM.
    with torch.no_grad():
        outs = []
        for x, ei in zip(xs, edge_indices):
            xt = torch.as_tensor(x, dtype=torch.float32)
            et = torch.as_tensor(ei, dtype=torch.long)
            if xt.shape[1] != in_dim:
                raise ValueError(f"in_dim mismatch: {xt.shape[1]} != {in_dim}")
            outs.append(enc.step(xt, et))
        return torch.stack(outs)
