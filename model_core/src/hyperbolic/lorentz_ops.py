import torch
import torch.nn as nn
import numpy as np


class LorentzOps:
    """Hyperboloid model operations: expmap, logmap, mobius_add.

    Numerical stability: clamp tangent vectors norm <= 1e-5 before expmap.
    # ponytail: Lorentz vs. Poincare — Lorentz chosen for training stability; Poincare for visualization only
    """

    def __init__(self, dim: int = 128, eps: float = 1e-5) -> None:
        self.dim = dim
        self.eps = eps

    def expmap(self, x: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Exponential map from tangent space to hyperboloid.

        Args:
            x: Tangent vector of shape (..., dim)
            c: Curvature parameter (negative)

        Returns:
            Point on hyperboloid of shape (..., dim)
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Clamp tangent vector norm for numerical stability
        norm_clamped = torch.clamp(norm, max=1e5)
        # Prevent division by zero when norm is tiny
        safe_norm = torch.where(norm_clamped < self.eps, self.eps, norm_clamped)
        coeff = torch.sqrt(torch.tensor(abs(c), dtype=torch.float32)) * safe_norm
        # expmap formula: cosh(coeff) * x / norm + sinh(coeff) / coeff * x / norm
        cosh_coeff = torch.cosh(coeff)
        sinh_coeff = torch.sinh(coeff)
        scale = torch.where(
            norm_clamped < self.eps,
            torch.ones_like(norm),
            cosh_coeff + sinh_coeff / coeff,
        )
        return scale * x / norm_clamped.clamp(min=self.eps)

    def logmap(self, x: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Logarithmic map from hyperboloid to tangent space.

        Args:
            x: Point on hyperboloid of shape (..., dim)
            c: Curvature parameter (negative)

        Returns:
            Tangent vector of shape (..., dim)
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Clamp for inverse stability
        norm_clamped = torch.clamp(norm, min=self.eps, max=1e5)
        coeff = torch.sqrt(torch.tensor(abs(c), dtype=torch.float32)) * norm_clamped
        # logmap formula: acosh(1 + (1-c)*norm^2) / (sqrt(|c|)*norm) * x
        inner = torch.clamp(1.0 + (1.0 - c) * norm_clamped**2, min=self.eps, max=1e5)
        scale = torch.acosh(inner) / coeff
        return scale * x / norm_clamped

    def mobius_add(self, a: torch.Tensor, b: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Möbius addition of two points in Lorentz space.

        Args:
            a: First point of shape (..., dim)
            b: Second point of shape (..., dim)
            c: Curvature parameter

        Returns:
            Möbius sum of shape (..., dim)
        """
        norm_a_sq = torch.sum(a**2, dim=-1, keepdim=True)
        norm_b_sq = torch.sum(b**2, dim=-1, keepdim=True)
        dot_ab = torch.sum(a * b, dim=-1, keepdim=True)
        denom = 1.0 - c * dot_ab
        # Clamp denominator to prevent division by zero
        denom = torch.clamp(denom, min=self.eps)
        numer = (1 - c * norm_b_sq) * a + (1 + c * dot_ab) * b
        return numer / denom


def demo() -> None:
    """Smoke test: LorentzOps with assert tests."""
    ops = LorentzOps(dim=4)
    x = torch.randn(2, 4) * 0.1
    c = -1.0
    exp_result = ops.expmap(x, c)
    assert exp_result.shape == (2, 4), f"expmap shape mismatch: {exp_result.shape}"
    assert not torch.isnan(exp_result).any(), "expmap must not produce NaN"
    log_result = ops.logmap(exp_result, c)
    assert log_result.shape == (2, 4), f"logmap shape mismatch: {log_result.shape}"
    # Check roundtrip consistency
    roundtrip = ops.expmap(log_result, c)
    assert torch.allclose(roundtrip, exp_result, atol=1e-3), "Roundtrip failed"
    a = torch.randn(2, 4) * 0.1
    b = torch.randn(2, 4) * 0.1
    mobius_sum = ops.mobius_add(a, b, c)
    assert mobius_sum.shape == (2, 4), f"mobius_add shape mismatch: {mobius_sum.shape}"
    print("LorentzOps demo passed")


if __name__ == "__main__":
    demo()
