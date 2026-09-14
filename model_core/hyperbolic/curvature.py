import torch
import torch.nn as nn


class LearnableCurvature(nn.Module):
    """Learnable curvature parameter per layer.

    Initialized c = -1.0, sigmoid-constrained to (-2.0, -0.1).
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
