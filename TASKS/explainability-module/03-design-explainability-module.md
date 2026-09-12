# 5C Structured Explainability Module — Design

## Context
Influence functions (Koh & Liang) for counterfactual explanation — 100x speedup over full recompute. Top-3 causal paths by backdoor-adjusted effect. HNSW search in Lorentz space. JSON Schema draft 2020-12. LLaMA 3.1 8B local (Temperature=0). Hardcoded jurisdiction→regulation mapping.

### [human] Design Decisions
- **Influence functions**: Koh & Liang approximation replaces full model recompute for counterfactuals. Speedup ~100x
- **Top-3 causal paths**: Backdoor-adjusted causal effect ranking. More paths = noise
- **HNSW in Lorentz**: Nearest-neighbor search in hyperbolic space. Brute force O(n) unacceptable
- **JSON Schema draft 2020-12**: AML-specific fields with custom validator. Modern schema format
- **LLaMA 3.1 8B**: Local inference, Temperature=0 (deterministic). No API dependency
- **Jurisdiction mapping**: Hardcoded dict. Dynamic loading is overkill for MVP

### [agent] resolved
- Influence functions: ∇L_f(x') · ∇_θ L_f(x) approximation; no backward pass over full dataset
- Top-3 paths: Dijkstra on backdoor-adjusted causal graph; top 3 by |effect|
- HNSW in Lorentz: Embeddings stored in Lorentz space; HNSW index for O(log n) counterfactual search
- JSON Schema: draft-2020-12 with `$schema` URI; custom AML keyword validation
- LLaMA 3.1 8B: GGUF quantized (Q4_K_M); vLLM backend; Temperature=0 for deterministic output
- Jurisdiction map: Python dict; YAML file loaded at init; fallback to EN rule

## Architecture
```
Alert → CounterfactualGenerator (Influence Functions) → CausalPathFinder (Top-3)
  ├── HNSW Search: Lorentz space nearest neighbor for counterfactual retrieval
  ├── JSON Schema: draft 2020-12 validation of explanation output
  ├── LLM Explainer: LLaMA 3.1 8B (Temperature=0) for narrative
  └── Jurisdiction: hardcoded mapping → regulation reference
```

**Note**: `# ponytail:` influence functions vs full recompute — influence functions chosen for 100x speedup; bounded fidelity loss

## Pushback Hypotheses
1. **H1**: Influence functions approximate counterfactuals poorly for non-differentiable GNN components. *Mitigation*: Gradient checkpointing at differentiable layers; hybrid approach for discrete outputs. `# ponytail:` differentiability gap
2. **H2**: Hardcoded jurisdiction mapping becomes stale as regulations evolve. *Mitigation*: Monthly review cycle; mapping is MVP approximation — Phase 2 adds dynamic source. `# ponytail:` mapping maintenance

## Open Questions
- What fidelity loss is acceptable for influence function counterfactuals vs ground truth?
- Does LLaMA 3.1 8B fit in A10G 24GB with vLLM overhead?

## [human] Acceptance
- Counterfactual explanation fidelity ≥ 0.9
- HNSW recall@k ≥ 0.95 in Lorentz space
- p99 explanation latency < 200ms
