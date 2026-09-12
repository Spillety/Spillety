# 5B Lazy Inference Engine — Design

## Context
Subgraph extraction via BFS depth 2 from `from_address` AND `to_address`. Micro-batch (32-64 events) for GPU utilization. LRU cache for hot addresses. Latency budget: 20ms extraction, 60ms forward pass, 20ms explanation. GPU: A10G/T4.

### [human] Design Decisions
- **BFS depth 2**: Union of neighborhoods from both `from_address` and `to_address`. Depth 1 loses context; depth 3 explodes complexity
- **Micro-batch 32-64**: Balances GPU utilization against latency budget. Flink buffer → GPU worker pipeline
- **LRU cache**: Hot addresses (exchanges, mixers) embeddings rarely change. Cache hit avoids subgraph extraction + forward pass
- **Latency budget**: 20ms extraction + 60ms forward + 20ms explanation = 100ms p99 under 200ms DoD

### [agent] resolved
- BFS depth 2: bidirectional union from source and target; max ~50-200 nodes per subgraph
- Micro-batch: 32-64 events packed into single GPU forward; dynamic padding to batch size
- LRU cache: capacity based on hot address count (~10k entries); TTL 1h for embeddings
- GPU: A10G 24GB / T4 16GB; batch size 64 fits in memory for GNN forward

## Architecture
```
Event Stream → SubgraphExtractor (BFS depth 2) → MicroBatchBuilder → GPU Forward → Explanation
  ├── LRU Cache: hot address embeddings (10k entries, 1h TTL)
  ├── Extraction: 20ms budget (BFS from from_addr + to_addr)
  ├── Forward: 60ms budget (micro-batch 32-64 on A10G/T4)
  └── Explanation: 20ms budget (Top-3 causal path + counterfactual)
```

**Note**: `# ponytail:` BFS depth 2 vs 3 — depth 2 chosen for latency; depth 3 adds 10x nodes but marginal accuracy

## Pushback Hypotheses
1. **H1**: BFS depth 2 from both addresses creates overlapping subgraphs, wasting compute. *Mitigation*: Deduplicate union before batching; cache shared intermediate nodes. `# ponytail:` subgraph dedup cost
2. **H2**: LRU cache invalidation on hot address embedding update causes cache thrashing. *Mitigation*: Write-through cache with versioning; invalidate only on confirmed cluster merge event.

## Open Questions
- What is the optimal LRU capacity given memory constraints on GPU?
- How does micro-batch size affect forward pass latency non-linearly?

## [human] Acceptance
- p99 inference latency < 200ms (20+60+20)
- LRU cache hit rate > 60% for hot addresses
- GPU memory utilization > 80% with batch size 32-64
