import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core.src.hyperbolic.lorentz_ops import LorentzOps


class HyperbolicMessagePassing(nn.Module):
    """Combines Lorentz ops + curvature for GNN message passing.

    forward(x, edges) -> Tensor.
    # ponytail: constraint strictness — curvature sigmoid-constrained to (-2.0, -0.1)
    """

    def __init__(self, embed_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.lorentz = LorentzOps(dim=embed_dim)
        self.linears = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)
        ])
        self.curvatures = nn.ParameterList([
            nn.Parameter(torch.tensor(-1.0, dtype=torch.float32)) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Hyperbolic message passing over graph edges.

        Args:
            x: Node features of shape (num_nodes, embed_dim)
            edges: Edge index tensor of shape (2, num_edges)

        Returns:
            Updated node features of shape (num_nodes, embed_dim)
        """
        h = x
        for i, linear in enumerate(self.linears):
            c = torch.sigmoid(self.curvatures[i]) * (-1.9) + (-0.1)
            # Aggregate messages from neighbors
            src = edges[0]
            dst = edges[1]
            aggregated = torch.zeros_like(h)
            aggregated = aggregated.scatter_add(0, src.unsqueeze(1).expand(-1, self.embed_dim), h[dst])
            # Transform and map via Lorentz expmap
            h_tangent = linear(h + aggregated)
            h = self.lorentz.expmap(h_tangent, c)
            # Gradient clipping for numerical stability
            h = torch.clamp(h, min=-1e5, max=1e5)
        return h


def demo() -> None:
    """Smoke test: HyperbolicMessagePassing with assert tests."""
    model = HyperbolicMessagePassing(embed_dim=16, num_layers=2)
    x = torch.randn(5, 16) * 0.1
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = model(x, edges)
    assert out.shape == (5, 16), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output must not contain NaN"
    assert not torch.isinf(out).any(), "Output must not contain Inf"
    print("HyperbolicMessagePassing demo passed")


if __name__ == "__main__":
    demo()
