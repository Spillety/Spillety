# 3B Hyperbolic Message Passing — Structure

```
hyperbolic-message-passing/
├── models/
│   ├── lorentz_ops.py            # Möbius addition, expmap, logmap in Lorentz
│   ├── hyperbolic_linear.py      # Hyperbolic linear layer (custom PyTorch)
│   ├── message_passing.py        # Hyperbolic message passing layer
│   └── curvature.py              # Learnable curvature parameter module
├── utilities/
│   ├── numerical_stability.py    # Tangent clamping, safe exp/log maps
│   └── validation.py             # Unit tests for edge cases (c→0, norm→1)
├── training/
│   ├── train_hyperbolic.py       # Training loop with gradient clipping
│   └── losses.py                 # Hyperbolic contrastive + task loss
└── tests/
    ├── __init__.py
    ├── test_lorentz_ops.py
    ├── test_numerical_stability.py
    └── test_curvature_bounds.py
```

## Key Components
- `lorentz_ops.py`: Hyperboloid model ops — `expmap`, `logmap`, `mobius_add`. English comments only.
- `curvature.py`: `nn.Parameter` per layer; sigmoid-constrained to (-2.0, -0.1)
- `numerical_stability.py`: Clamps tangent vectors pre-expmap; gradient clipping wrapper
- `message_passing.py`: Combines Lorentz ops + curvature for GNN message passing

## Pushback Hypotheses
1. **H1**: Custom Lorentz ops in pure PyTorch may be slower than geomstats for validation. *Mitigation*: Profile both; use geomstats for offline validation, custom for training.
2. **H2**: Sigmoid-constrained curvature may plateau at boundary values, preventing exploration. *Mitigation*: Use tanh transform with wider range; add entropy regularization on curvature. `# ponytail:` constraint strictness

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
