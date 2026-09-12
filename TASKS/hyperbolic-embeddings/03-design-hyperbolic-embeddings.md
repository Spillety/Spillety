# 2.2 Hyperbolic Embeddings Storage — Design

## Context
128D float32 vector node property for Memgraph. Incremental update on each event, full recompute nightly. HNSW index for nearest scam cluster query.

### [human] Design Decisions
- **128D**: Sweet spot — 64D loses hierarchy, 256D no gain for this graph size
- **Incremental**: Lazy inference on event; full recompute nightly batch
- **HNSW**: Mandatory for O(log n) nearest-neighbor; brute force O(n) unacceptable for counterfactual generation

### [agent] resolved
- Embedding dimension validated against graph diameter and node count (~100M edges)
- Update strategy: delta update via event streaming, full recalc via nightly Spark job
- HNSW efConstruction=200, M=16 for recall/k≥0.95 benchmark

## Architecture
```
Event → EmbeddingUpdateService → Memgraph HNSW
  ├── lazy_inference(node) → float32[128] → upsert
  └── nightly_recompute() → full batch → rebuild HNSW
```

**Vector property**: `hyperbolic_emb: list<float>` (128 elements, float32)
**HNSW index**: Memgraph native vector index on `hyperbolic_emb`

## Pushback Hypotheses
1. **H1**: Incremental updates may drift embeddings vs. full recompute. *Mitigation*: nightly reconciliation with cosine similarity threshold (drift > 0.05 triggers full recalc). `# ponytail:` lazy vs. batch trade-off
2. **H2**: HNSW index rebuild on every full recompute causes write amplification. *Mitigation*: dual-buffer index swap — build new index offline, atomically swap.

## Open Questions
- What is the maximum acceptable embedding drift before nightly recompute?
- Does Memgraph HNSW support GPU-accelerated search for sub-10ms queries?

## [human] Acceptance
- recall@k ≥ 0.95 on HNSW benchmark
- Update latency < 50ms per event (lazy)
- Nightly recompute < 2h for full graph
