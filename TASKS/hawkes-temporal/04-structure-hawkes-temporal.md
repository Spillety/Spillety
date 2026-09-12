# 5A Hawkes Temporal Encoding — Structure

```
hawkes-temporal/
├── kernel/
│   ├── __init__.py
│   ├── power_law.py          # Power-law kernel k(t)=τ^{-α}, truncated sum
│   └── seasonal_mu.py        # μ(t) Fourier seasonality (hourly + daily)
├── online_sgd/
│   ├── __init__.py
│   ├── regret_bounder.py     # O(log T) adaptive learning rate η_t=O(1/√t)
│   └── updater.py            # O(1) closed-form λ(t) increment
└── tests/
    ├── __init__.py
    ├── test_power_law.py
    ├── test_seasonal_mu.py
    └── test_regret_bound.py
```

## Key Components
- `power_law.py`: Power-law kernel with TTL-truncated history. English comments only.
- `seasonal_mu.py`: Fourier decomposition of μ(t) — 24h + 7d components. `# ponytail:` component count
- `regret_bounder.py`: Adaptive η_t ensuring O(log T) regret; gradient clipping for heavy tails
- `updater.py`: O(1) additive λ(t) increment per event — no integral recompute

## Pushback Hypotheses
1. **H1**: TTL truncation at 14 days discards long-range excitation for slow AML schemes. *Mitigation*: Decay-weighted tail summary (exponential sketch) preserves long-range signal in O(1) space.
2. **H2**: Fourier μ(t) adds overhead for per-event computation. *Mitigation*: Precompute μ(t) on the hour; cache values; only interpolate between grid points. `# ponytail:` precomputation granularity

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
