import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core.hyperbolic.lorentz_ops import LorentzOps
from model_core.hyperbolic.poincare_ops import PoincareBallOps

Geometry = str


class HyperbolicMessagePassing(nn.Module):
    """Combines Lorentz/Poincare ops + curvature for GNN message passing.

    geometry: "lorentz" (legacy default, kept for ablation), "poincare"
    (canonical per temp.md) or "euclidean" (ablation baseline).
    forward(x, edges) -> Tensor.
    """

    VALID_GEOMETRIES = ("lorentz", "poincare", "euclidean")

    def __init__(self, embed_dim: int = 128, num_layers: int = 2, geometry: Geometry = "lorentz") -> None:
        super().__init__()
        if geometry not in self.VALID_GEOMETRIES:
            raise ValueError(f"geometry must be one of {self.VALID_GEOMETRIES}, got {geometry!r}")
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.geometry = geometry
        self.lorentz = LorentzOps(dim=embed_dim)
        self.poincare = PoincareBallOps(dim=embed_dim)
        self.linears = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)
        ])
        self.curvatures = nn.ParameterList([
            nn.Parameter(torch.tensor(-1.0, dtype=torch.float32)) for _ in range(num_layers)
        ])

    def _aggregate(self, h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        src, dst = edges[0], edges[1]
        return torch.zeros_like(h).scatter_add(0, src.unsqueeze(1).expand(-1, self.embed_dim), h[dst])

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if self.geometry == "euclidean":
            return self._forward_euclidean(x, edges)
        if self.geometry == "poincare":
            return self._forward_poincare(x, edges)
        return self._forward_lorentz(x, edges)

    def _forward_lorentz(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        h = x
        for i, linear in enumerate(self.linears):
            c = torch.sigmoid(self.curvatures[i]) * (-1.9) + (-0.1)
            h_tangent = linear(h + self._aggregate(h, edges))
            h = self.lorentz.expmap(h_tangent, c)
            # Gradient clipping for numerical stability
            h = torch.clamp(h, min=-1e5, max=1e5)
        return h

    def _forward_poincare(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        # Aggregate in tangent space at origin, project back into the ball
        h = self.poincare.expmap0(x)
        for i, linear in enumerate(self.linears):
            c = torch.sigmoid(self.curvatures[i]) * (-1.9) + (-0.1)
            tangent = self.poincare.logmap0(h, float(c))
            h = self.poincare.expmap0(linear(tangent + self._aggregate(tangent, edges)), float(c))
        return h

    def _forward_euclidean(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        h = x
        for linear in self.linears:
            h = linear(h + self._aggregate(h, edges))
            h = torch.clamp(h, min=-1e5, max=1e5)
        return h
