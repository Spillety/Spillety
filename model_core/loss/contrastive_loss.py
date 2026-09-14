import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        sim = torch.matmul(a, p.T) / self.tau
        return F.cross_entropy(sim, torch.arange(anchor.size(0), device=anchor.device))
