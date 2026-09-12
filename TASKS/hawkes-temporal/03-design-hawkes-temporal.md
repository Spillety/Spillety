# 5A Hawkes Temporal Encoding — Design

## Context
Power-law kernel (not exponential) for self-exciting processes. Online SGD with regret bound O(log T). Time-varying μ(t) with hourly/daily seasonality. O(1) closed-form update where possible. AML event intensity λ(t) modeling.

### [human] Design Decisions
- **Power-law kernel**: k(t) = (t + c)^{-α} vs exponential decay — power-law captures long-range excitation essential for AML pattern chains
- **Online SGD**: Regret bound O(log T), not EM (EM is batch). Per-event update, no full recompute
- **μ(t) seasonality**: Hourly + daily components. Constant baseline ignores attack patterns in off-hours
- **O(1) update**: Where closed-form exists (power-law integral has analytic form), skip iterative optimization

### [agent] resolved
- Power-law kernel: λ(t) = μ(t) + Σ k(t - t_i) with k(τ) = τ^{-α}; α learned via online SGD
- μ(t) = μ_0 + Σ_s μ_s · I(season_s(t)); 24-hour + 7-day Fourier components
- Regret bound O(log T) achieved via adaptive learning rate η_t = O(1/√t)
- O(1) closed-form: for power-law kernel, λ(t) update is additive — no integral recompute needed

## Architecture
```
Event(t_i) → λ(t_i) = μ(t_i) + Σ_{t_j < t_i} (t_i - t_j)^{-α}
  ├── μ(t): Fourier seasonality (hourly + daily components)
  ├── α: Online SGD update (regret O(log T))
  └── λ(t_i) update: O(1) closed-form additive increment
```

**Note**: `# ponytail:` Power-law vs exponential — power-law chosen for long-range AML chains; exponential gives O(1) but misses multi-hop patterns

## Pushback Hypotheses
1. **H1**: Power-law kernel lacks closed-form integral, making O(1) update impossible for arbitrary history. *Mitigation*: Truncate history with TTL=14d (from Flink state); only sum over active window → O(window_size) bounded, effectively constant. `# ponytail:` truncation vs accuracy
2. **H2**: Online SGD may diverge for heavy-tailed power-law gradients. *Mitigation*: Adaptive clipping + learning rate warmup; regret bound guarantee holds only with bounded gradients. `# ponytail:` gradient stability

## Open Questions
- What truncation radius preserves fidelity for AML patterns spanning >14 days?
- How many Fourier components for μ(t) are needed before diminishing returns?

## [human] Acceptance
- Regret bound O(log T) verified empirically
- λ(t) p99 latency < 1ms per event
- Power-law α converges within 10k events
