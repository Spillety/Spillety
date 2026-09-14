import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core.src.hyperbolic.lorentz_ops import LorentzOps


class HyperbolicLinear(nn.Module):
    """Hyperbolic linear layer (custom PyTorch).

    Input/output in Lorentz space. Uses LorentzOps internally.
    # ponytail: curvature bounds — sigmoid-constrained to (-2.0, -0.1)
    """

    def __init__(self, in_features: int = 128, out_features: int = 128, c_init: float = -1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.lorentz = LorentzOps(dim=in_features)
        # Learnable curvature for this layer
        self.curvature = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hyperbolic linear transformation.

        Args:
            x: Input tensor in Lorentz space of shape (..., in_features)

        Returns:
            Output tensor in Lorentz space of shape (..., out_features)
        """
        c = torch.sigmoid(self.curvature) * (-1.9) + (-0.1)
        # Euclidean linear transform in tangent space
        x_tangent = F.linear(x, self.weight, self.bias)
        # Map back to Lorentz space via expmap
        return self.lorentz.expmap(x_tangent, c)


def demo() -> None:
    """Smoke test: HyperbolicLinear with assert tests."""
    layer = HyperbolicLinear(in_features=16, out_features=8)
    x = torch.randn(4, 16) * 0.1
    out = layer(x)
    assert out.shape == (4, 8), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output must not contain NaN"
    assert not torch.isinf(out).any(), "Output must not contain Inf"
    # Check curvature constraint
    c = torch.sigmoid(layer.curvature) * (-1.9) + (-0.1)
    assert c.item() >= -2.0 and c.item() < -0.1, f"Curvature out of bounds: {c.item()}"
    print("HyperbolicLinear demo passed")


if __name__ == "__main__":
    demo()
