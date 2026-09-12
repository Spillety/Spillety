# 3B Hyperbolic Message Passing — Design

## Context
Lorentz model for hyperbolic message passing (numerically stable). Learnable curvature per layer, initialized c = -1.0. Custom PyTorch implementation (geomstats too slow). Numerical stability via tangent vector clamping and gradient clipping.

### [human] Design Decisions
- **Lorentz model**: Numerically more stable than Poincaré ball; Poincaré only for visualization
- **Curvature**: Learnable per-layer, initialized c = -1.0. Adaptive per-node not needed — hierarchy is global
- **Numerical stability**: Clamp tangent vectors to norm ≤ 1e-5 before exp map; gradient clipping max_norm=1.0
- **Library**: Custom PyTorch (geomstats too slow for training)

### [agent] resolved
- Lorentz model implemented via Möbius addition in hyperboloid model
- Per-layer curvature as `nn.Parameter`, initialized to -1.0, clipped to [-2.0, 0.0)
- Exponential map: clamp tangent norm before `expmap`; log map: clamp for inverse
- Gradient clipping: `torch.nn.utils.clip_grad_norm_` with max_norm=1.0

## Architecture
```
Input (Lorentz space) → HyperbolicLinear → HyperbolicMessagePassing → Output
  ├── Per-layer curvature: nn.Parameter (learnable, init c=-1.0)
  ├── Tangent clamp: norm ≤ 1e-5 before expmap
  └── Gradient clip: max_norm=1.0
```

**Note**: `# ponytail:` Lorentz vs. Poincaré — Lorentz chosen for training stability; Poincaré for visualization only

## Pushback Hypotheses
1. **H1**: Custom PyTorch hyperbolic ops may have numerical instability at extreme curvature. *Mitigation*: Unit tests for edge cases (c → 0, norm → 1); fallback to geomstats for validation.
2. **H2**: Learnable curvature may diverge during training (curvature → 0 = Euclidean). *Mitigation*: Constrain curvature via sigmoid transform to c ∈ (-2, -0.1); warm-start with c=-1.0. `# ponytail:` curvature bounds

## Open Questions
- Does learnable curvature converge to consistent values across layers?
- How does Lorentz model compare to Poincaré for downstream task accuracy?

## [human] Acceptance
- Lorentz model training converges < 24h
- Numerical stability tests pass (no NaN/Inf)
- Curvature stays within [-2.0, -0.1) during training
