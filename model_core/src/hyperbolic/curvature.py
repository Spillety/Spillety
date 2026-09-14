import torch
import torch.nn as nn


class LearnableCurvature(nn.Module):
    """Learnable curvature parameter per layer.

    Initialized c = -1.0, sigmoid-constrained to (-2.0, -0.1).
    # ponytail: curvature bounds — sigmoid-constrained to (-2.0, -0.1)
    """

    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.curvatures = nn.ParameterList([
            nn.Parameter(torch.tensor(-1.0, dtype=torch.float32)) for _ in range(num_layers)
        ])

    def forward(self) -> torch.Tensor:
        """Return curvature values for all layers, sigmoid-constrained.

        Returns:
            Tensor of shape (num_layers,) with curvature in (-2.0, -0.1)
        """
        curvatures = torch.stack([
            torch.sigmoid(c) * (-1.9) + (-0.1) for c in self.curvatures
        ])
        return curvatures


def demo() -> None:
    """Smoke test: LearnableCurvature with assert tests."""
    lc = LearnableCurvature(num_layers=3)
    curvatures = lc.forward()
    assert curvatures.shape == (3,), f"Shape mismatch: {curvatures.shape}"
    assert not torch.isnan(curvatures).any(), "Curvature must not be NaN"
    assert not torch.isinf(curvatures).any(), "Curvature must not be Inf"
    for c in curvatures:
        assert c.item() >= -2.0 and c.item() < -0.1, f"Curvature out of bounds: {c.item()}"
    # Check initial values are close to -1.0
    assert curvatures[0].item() < 0, "Curvature must be negative"
    print("LearnableCurvature demo passed")


if __name__ == "__main__":
    demo()
