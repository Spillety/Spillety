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
    with torch.no_grad():
        outs = []
        for x, ei in zip(xs, edge_indices):
            xt = torch.as_tensor(x, dtype=torch.float32)
            et = torch.as_tensor(ei, dtype=torch.long)
            if xt.shape[1] != in_dim:
                raise ValueError(f"in_dim mismatch: {xt.shape[1]} != {in_dim}")
            outs.append(enc.step(xt, et))
        return torch.stack(outs)


class DecoupledEvolveGCNEncoder:
    """
    ## Temporal/structural decoupled encoder with gated fusion

    Separates temporal feature evolution (GRU on node features) from
    structural propagation (GCN on graph). Fuses with a learned gate.

    Parameters
    ----------
    in_dim : int
        Input feature width.
    hidden_dim : int
        Hidden width for both temporal and structural branches.
    out_dim : int
        Output embedding width.
    """

    def __init__(self, torch, in_dim, hidden_dim=32, out_dim=16):
        self._torch = torch
        s = in_dim**-0.5
        sh = hidden_dim**-0.5

        # Temporal branch: GRU tracks feature evolution per node
        self.temporal_gru = torch.nn.GRUCell(in_dim, hidden_dim)
        self.temporal_proj = torch.randn(hidden_dim, out_dim) * sh

        # Structural branch: GCN aggregates neighbourhood
        self.struct_w = torch.randn(in_dim, hidden_dim) * s
        self.struct_proj = torch.randn(hidden_dim, out_dim) * sh

        # Gated fusion
        self.gate_w = torch.randn(out_dim * 2, out_dim) * (out_dim * 2) ** -0.5
        self.gate_b = torch.zeros(out_dim)

    def parameters(self):
        return (
            [self.temporal_proj, self.struct_w, self.struct_proj, self.gate_w, self.gate_b]
            + list(self.temporal_gru.parameters())
        )

    def step(self, x, edge_index, h_prev):
        """
        Single step: temporal + structural + gated fusion.

        Parameters
        ----------
        x : torch.Tensor (N, in_dim)
        edge_index : torch.Tensor (2, E)
        h_prev : torch.Tensor (N, hidden_dim)
            GRU hidden state from previous step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (fused_embedding (N, out_dim), new_h_prev (N, hidden_dim))
        """
        torch = self._torch
        n = x.shape[0]

        # Temporal: GRU tracks how features evolve per node
        h_temp = torch.relu(self.temporal_gru(x, h_prev))
        z_temp = h_temp @ self.temporal_proj

        # Structural: GCN aggregates neighbourhood at this snapshot
        a_hat = _norm_adj(torch, n, edge_index)
        h_struct = torch.relu(a_hat @ x @ self.struct_w)
        z_struct = h_struct @ self.struct_proj

        # Gated fusion
        cat = torch.cat([z_temp, z_struct], dim=-1)
        gate = torch.sigmoid(cat @ self.gate_w + self.gate_b)
        fused = gate * z_temp + (1 - gate) * z_struct
        return fused, h_temp


def encode_temporal_decoupled(
    xs,
    edge_indices,
    hidden_dim=32,
    out_dim=16,
    seed=72,
) -> Any:
    """
    ## Forward pass with decoupled temporal/structural encoding

    Parameters
    ----------
    xs : Sequence[array-like]
        Per-step node features, each (N, in_dim).
    edge_indices : Sequence[array-like]
        Per-step edge lists, each (2, E_t).
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output embedding width.
    seed : int
        Seed; CPU only.

    Returns
    ----------
    torch.Tensor
        Stacked embeddings (T, N, out_dim).
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
    enc = DecoupledEvolveGCNEncoder(torch, in_dim, hidden_dim, out_dim)
    with torch.no_grad():
        outs = []
        h = torch.zeros(torch.as_tensor(xs[0], dtype=torch.float32).shape[0], hidden_dim)
        for x, ei in zip(xs, edge_indices):
            xt = torch.as_tensor(x, dtype=torch.float32)
            et = torch.as_tensor(ei, dtype=torch.long)
            if xt.shape[1] != in_dim:
                raise ValueError(f"in_dim mismatch: {xt.shape[1]} != {in_dim}")
            z, h = enc.step(xt, et, h)
            outs.append(z)
        return torch.stack(outs)
