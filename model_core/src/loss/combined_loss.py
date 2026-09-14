import math
import torch
import torch.nn as nn

from .focal_loss import FocalLoss
from .contrastive_loss import ContrastiveLoss


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta_init: float = 0.1, beta_final: float = 0.5, annealing_epochs: int = 50):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=2.0)
        self.contrastive = ContrastiveLoss(tau=0.1)
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.annealing_epochs = annealing_epochs
        self.current_beta = beta_init

    def update_beta(self, epoch: int) -> None:
        progress = min(epoch / self.annealing_epochs, 1.0)
        self.current_beta = self.beta_final + (self.beta_init - self.beta_final) * 0.5 * (1 + math.cos(math.pi * progress))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        fl = self.focal(logits, targets)
        cl = self.contrastive(anchor, positive)
        return self.alpha * fl + self.current_beta * cl


def demo() -> None:
    loss_fn = CombinedLoss(alpha=1.0, beta_init=0.1, beta_final=0.5, annealing_epochs=50)
    assert loss_fn.current_beta == 0.1
    loss_fn.update_beta(25)
    assert 0.1 <= loss_fn.current_beta <= 0.5
    logits = torch.randn(32, 6)
    targets = torch.randint(0, 6, (32,))
    anchor = torch.randn(32, 128)
    positive = anchor + 0.01 * torch.randn(32, 128)
    loss = loss_fn(logits, targets, anchor, positive)
    assert loss.item() > 0
    print("combined_loss demo passed")



