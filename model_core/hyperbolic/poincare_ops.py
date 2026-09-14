import torch


class PoincareBallOps:
    """Poincare ball model ops: expmap, logmap, mobius_add, distance.

    Canonical geometry for message passing (temp.md); Lorentz kept for ablation.
    Ball radius R = 1 / sqrt(|c|); outputs projected inside the ball.
    # ponytail: formulas assume constant negative curvature per call (c < 0)
    """

    def __init__(self, dim: int = 128, eps: float = 1e-5, max_norm: float | None = None) -> None:
        self.dim = dim
        self.eps = eps
        self.max_norm = max_norm

    def _radius(self, c: float | torch.Tensor) -> torch.Tensor:
        k = torch.sqrt(torch.as_tensor(abs(float(c)) if not isinstance(c, torch.Tensor) else torch.abs(c)))
        return 1.0 / k.clamp(min=self.eps)

    def project(self, x: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Project points inside the ball with safety margin eps."""
        r = self._radius(c)
        limit = r - self.eps
        if self.max_norm is not None:
            limit = min(limit, self.max_norm)
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.where(norm >= limit, x * (limit / norm), x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Mimer-Kelvin Mobius addition; result projected into the ball."""
        c_t = torch.as_tensor(float(c), dtype=x.dtype, device=x.device)
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c_t * xy + c_t * y2) * x + (1 - c_t * x2) * y
        denom = 1 + 2 * c_t * xy + c_t * c_t * x2 * y2
        out = num / denom.clamp(min=self.eps)
        return self.project(out, c)

    def expmap(self, x: torch.Tensor, v: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Exponential map at base point x for tangent vector v."""
        k = torch.sqrt(torch.as_tensor(abs(float(c)), dtype=x.dtype, device=x.device)).clamp(min=self.eps)
        lam = 2.0 / (1 + float(c) * torch.sum(x * x, dim=-1, keepdim=True)).clamp(min=self.eps)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        # Guard tiny tangent vectors: limit direction term to zero
        direction = v / v_norm.clamp(min=self.eps)
        factor = torch.tanh(k * lam * v_norm / 2) / k.clamp(min=self.eps)
        inner = torch.where(v_norm < self.eps, torch.zeros_like(factor), factor * direction)
        return self.mobius_add(x, inner, c)

    def expmap0(self, v: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Exponential map at origin; primary encoder lift Euclidean -> ball."""
        k = torch.sqrt(torch.as_tensor(abs(float(c)), dtype=v.dtype, device=v.device)).clamp(min=self.eps)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        direction = v / v_norm.clamp(min=self.eps)
        factor = torch.tanh(k * v_norm / 2) / k
        out = torch.where(v_norm < self.eps, torch.zeros_like(v), factor * direction)
        return self.project(out, c)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Logarithmic map at x: tangent vector pointing from x to y."""
        k = torch.sqrt(torch.as_tensor(abs(float(c)), dtype=x.dtype, device=x.device)).clamp(min=self.eps)
        lam = 2.0 / (1 + float(c) * torch.sum(x * x, dim=-1, keepdim=True)).clamp(min=self.eps)
        w = self.mobius_add(-x, y, c)
        w_norm = torch.norm(w, dim=-1, keepdim=True)
        arg = (k * w_norm).clamp(max=1 - self.eps)
        scale = 2.0 * torch.atanh(arg) / (k * lam * w_norm.clamp(min=self.eps))
        return torch.where(w_norm < self.eps, torch.zeros_like(x), scale * w)

    def logmap0(self, y: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Log map at origin: ball -> tangent space."""
        k = torch.sqrt(torch.as_tensor(abs(float(c)), dtype=y.dtype, device=y.device)).clamp(min=self.eps)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        arg = (k * y_norm).clamp(max=1 - self.eps)
        scale = 2.0 * torch.atanh(arg) / (k * y_norm.clamp(min=self.eps))
        return torch.where(y_norm < self.eps, torch.zeros_like(y), scale * y)

    def distance(self, x: torch.Tensor, y: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        """Geodesic distance d(x, y) = (2/k) * artanh(k * ||-x (+) y||)."""
        k = torch.sqrt(torch.as_tensor(abs(float(c)), dtype=x.dtype, device=x.device)).clamp(min=self.eps)
        w_norm = torch.norm(self.mobius_add(-x, y, c), dim=-1, keepdim=True)
        return (2.0 / k) * torch.atanh((k * w_norm).clamp(max=1 - self.eps))
