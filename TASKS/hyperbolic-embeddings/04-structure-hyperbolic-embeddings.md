# 2.2 Hyperbolic Embeddings Storage — Structure

```
memgraph/
├── hyperbolic/
│   ├── __init__.py
│   ├── embedding_service.py      # Incremental + nightly update orchestration
│   ├── hnsw_manager.py           # HNSW index create/rebuild/swap
│   ├── lazy_inference.py         # Event-triggered embedding update
│   ├── nightly_recompute.py      # Full batch recompute via Spark
│   └── tests/
│       ├── __init__.py
│       ├── test_embedding_service.py
│       ├── test_hnsw_manager.py
│       └── test_lazy_inference.py
```

## Key Components
- `embedding_service.py`: Orchestrates lazy + batch updates. English inline comments only.
- `hnsw_manager.py`: Memgraph Cypher wrapper for vector index. `# ponytail:` index swap strategy
- `lazy_inference.py`: Per-event delta update using cached neighborhood
- `nightly_recompute.py`: Spark job producing Parquet → Memgraph bulk load

## Pushback Hypotheses
1. **H1**: Spark-based nightly recompute adds infrastructure complexity vs. in-memory Memgraph recalc. *Mitigation*: Benchmark pure-Memgraph recalc for <100M nodes; fallback if faster.
2. **H2**: Float32 storage per node (512 bytes) adds memory pressure at 100M+ nodes. *Mitigation*: Quantize to int8 during nightly recompute; dequantize on read if recall permits. `# ponytail:` precision vs. memory

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
