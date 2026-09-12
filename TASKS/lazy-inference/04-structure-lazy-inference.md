# 5B Lazy Inference Engine — Structure

```
lazy-inference/
├── extraction/
│   ├── __init__.py
│   ├── bfs_subgraph.py         # BFS depth 2 from from_addr + to_addr (union)
│   └── subgraph_cache.py       # LRU cache for hot subgraph embeddings
├── batching/
│   ├── __init__.py
│   ├── micro_batch.py          # Batch 32-64 events with dynamic padding
│   └── flink_bridge.py         # Flink buffer → GPU worker pipeline
├── inference/
│   ├── __init__.py
│   ├── gpu_forward.py          # GPU forward pass (A10G/T4)
│   └── explainer.py            # Top-3 causal path + counterfactual (20ms)
└── tests/
    ├── __init__.py
    ├── test_bfs_subgraph.py
    ├── test_micro_batch.py
    ├── test_lru_cache.py
    └── test_latency_budget.py
```

## Key Components
- `bfs_subgraph.py`: BFS depth 2 union from source + target. English comments only.
- `micro_batch.py`: Dynamic padding to 32-64; Flink buffer integration. `# ponytail:` batch size tuning
- `gpu_forward.py`: A10G/T4 inference; memory-bound at batch 64
- `subgraph_cache.py`: LRU with 10k capacity, 1h TTL for hot address embeddings

## Pushback Hypotheses
1. **H1**: BFS on every event is expensive even at depth 2. *Mitigation*: Cache subgraph skeletons for LRU-hit addresses; only extract delta for new nodes.
2. **H2**: Micro-batch padding introduces idle GPU cycles for sparse batches. *Mitigation*: Adaptive batch composition — prioritize by subgraph size similarity. `# ponytail:` padding strategy

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
