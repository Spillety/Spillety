# 3A Causal Attention Implementation — Design

## Context
NOTEARS for initial DAG discovery (differentiable, O(n³) but n ≈ local subgraph ~100). PyTorch Geometric Temporal for temporal layers, custom causal attention on PyTorch. Domain knowledge prior for edge types.

### [human] Design Decisions
- **NOTEARS**: Initial DAG discovery — differentiable, O(n³) manageable at local subgraph scale
- **Hybrid DAG**: Domain knowledge (expert graph) for edge types (TRANSFERS → risk, MENTIONED_IN → risk); data-driven for weights
- **Backdoor covariates**: Address features (age, tx_count, unique_counterparties), amount, time-of-day, chain. Confounders: `is_exchange_internal` flag
- **Library**: PyG-Temporal for temporal layers; custom PyTorch causal attention

### [agent] resolved
- NOTEARS applied per local subgraph (BFS depth 2) — n ≤ 100, O(n³) ≈ 1M ops, acceptable
- Domain prior: expert graph defines allowed edge types; NOTEARS learns weights
- Confounder handling: `is_exchange_internal` as adjustment set member
- PyG-Temporal temporal convolution layers for message passing over time

## Architecture
```
Local Subgraph (BFS depth 2) → NOTEARS DAG → Causal Attention
  ├── Domain Prior: expert edge types → mask on adjacency
  ├── Covariates: address features, amount, time, chain
  └── PyG-Temporal layers → causal attention weights
```

**Note**: `# ponytail:` NOTEARS global vs. local — local subgraph chosen for O(n³) tractability

## Pushback Hypotheses
1. **H1**: NOTEARS on local subgraphs may miss global causal structure. *Mitigation*: GES with domain prior for production global DAG; NOTEARS only for initial seed.
2. **H2**: Domain knowledge prior may bias NOTEARS toward known patterns, reducing discovery of novel scam types. *Mitigation*: Soft constraint (weighted prior) instead of hard mask; allow high-confidence novel edges. `# ponytail:` prior strength

## Open Questions
- What is the optimal BFS depth for NOTEARS subgraph extraction?
- How to validate DAG quality when ground truth is unknown?

## [human] Acceptance
- NOTEARS DAG reconstruction accuracy ≥ 0.85
- Training converges < 24h on local subgraph batches
- Numerical stability tests pass (gradient norms bounded)
