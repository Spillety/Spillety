import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        fl = self.alpha * (1 - pt) ** self.gamma * ce
        fl = fl.clamp(min=0.1)
        if self.reduction == "mean":
            return fl.mean()
        return fl.sum()


def demo() -> None:
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    loss = loss_fn(logits, torch.tensor([0, 1]))
    assert loss.item() < 0.01
    logits2 = torch.tensor([[0.1, -0.1]])
    loss2 = loss_fn(logits2, torch.tensor([0]))
    assert loss2.item() >= 0.1
    print("focal_loss demo passed")


if __name__ == "__main__":
    demo()
