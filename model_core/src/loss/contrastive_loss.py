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


def demo() -> None:
    loss_low = ContrastiveLoss(tau=0.05)
    loss_high = ContrastiveLoss(tau=0.5)
    a = torch.randn(32, 128)
    p = a + 0.01 * torch.randn(32, 128)
    assert loss_low(a, p).item() > loss_high(a, p).item()
    print("contrastive_loss demo passed")


if __name__ == "__main__":
    demo()
