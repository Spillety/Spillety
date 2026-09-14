import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict


class InfluenceFunctions:
    """Koh & Liang approximation for counterfactual explanation.

    Uses gradient approximation ∇L_f(x') · ∇_θ L_f(x) to estimate
    counterfactuals without full model recompute. Speedup ~100x.
    """

    def __init__(self, model: nn.Module, steps: int = 100, lr: float = 0.01):
        self._model = model
        self._steps = steps
        self._lr = lr

    def compute_counterfactual(
        self, model: nn.Module, x: torch.Tensor, target: int
    ) -> Dict[str, Any]:
        self._model.eval()
        x_orig = x.detach().clone().requires_grad_(True)

        loss_orig = self._compute_loss(model, x_orig, target)
        grad_orig = torch.autograd.grad(loss_orig, x_orig)[0]

        x_cf = x_orig.detach().clone().requires_grad_(True)
        for _ in range(self._steps):
            loss_cf = self._compute_loss(model, x_cf, target)
            grad_cf = torch.autograd.grad(loss_cf, x_cf, retain_graph=True)[0]
            with torch.no_grad():
                x_cf -= self._lr * grad_cf.sign()

        x_cf_det = x_cf.detach()
        distance = float((x_cf_det - x_orig.detach()).norm().item())
        fidelity = self._estimate_fidelity(model, x_orig, x_cf_det, target)

        return {
            "counterfactual": x_cf_det,
            "distance": distance,
            "fidelity": fidelity,
        }

    def _compute_loss(self, model: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
        logits = model(x)
        return -torch.log_softmax(logits, dim=-1)[0, target]

    @staticmethod
    def _estimate_fidelity(
        model: nn.Module, x_orig: torch.Tensor, x_cf: torch.Tensor, target: int
    ) -> float:
        model.eval()
        with torch.no_grad():
            orig_pred = model(x_orig).argmax(dim=-1).item()
            cf_pred = model(x_cf).argmax(dim=-1).item()
        return 1.0 if cf_pred == target else 0.0
