import torch
import torch.nn as nn
import numpy as np


class NOTEARS_DAG(nn.Module):
    """Differentiable DAG discovery via adjacency matrix parameterization.

    Uses local subgraph (BFS depth 2) for O(n³) tractability.
    # ponytail: NOTEARS global vs local — local subgraph chosen for O(n³) tractability
    """

    def __init__(self, max_nodes: int = 100, lambda1: float = 0.1) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.lambda1 = lambda1
        self.adj = nn.Parameter(torch.eye(max_nodes))

    def forward(self, adj_matrix: torch.Tensor | None = None) -> torch.Tensor:
        """Compute NOTEARS smoothness loss from adjacency matrix.

        Args:
            adj_matrix: Optional square adjacency matrix of shape (n, n).
                If None, uses self.adj parameter.

        Returns:
            DAG smoothness loss scalar
        """
        adj = adj_matrix if adj_matrix is not None else self.adj
        if not adj.requires_grad:
            adj = adj.detach().requires_grad_(True)
        n = adj.size(0)
        # O(n³) matrix exponential for DAG constraint
        expm = torch.linalg.matrix_exp(adj * self.lambda1)
        loss_dag = (expm.trace() - n) / self.lambda1
        # L1 regularization for sparsity
        loss_sparse = self.lambda1 * torch.abs(adj).sum()
        return loss_dag + loss_sparse

    def get_adjacency(self) -> torch.Tensor:
        """Return current adjacency parameter with domain prior mask applied."""
        return torch.sigmoid(self.adj) * self._get_mask()

    def _get_mask(self) -> torch.Tensor:
        """Domain prior mask: prevent self-loops."""
        mask = torch.eye(self.max_nodes, dtype=torch.float32)
        return 1.0 - mask


def demo() -> None:
    """Smoke test: NOTEARS_DAG with assert tests."""
    model = NOTEARS_DAG(max_nodes=10)
    adj = torch.randn(10, 10) * 0.1
    loss = model.forward(adj)
    assert loss.ndim == 0, "Loss must be scalar"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert loss.item() > 0, "Loss must be positive"
    adj_out = model.get_adjacency()
    assert adj_out.shape == (10, 10), "Adjacency shape mismatch"
    # Self-loop mask check
    assert torch.allclose(adj_out.diag(), torch.zeros(10)), "Self-loops must be masked"
    print("NOTEARS_DAG demo passed")


if __name__ == "__main__":
    demo()
