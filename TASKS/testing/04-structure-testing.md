# 04-structure-testing

> Phase: S (Structure) | Slug: testing | Status: In Progress

## 1. Project Structure

```
testing/
├── config/{pytest.yaml,chaos_scenarios.yaml}
├── tests/{unit/{test_kafka,test_flink,test_memgraph},integration/{test_pipeline,test_feature_store},e2e/{test_full_pipeline},chaos/{test_kafka_down,test_memgraph_crash,test_gpu_oom}}
├── fixtures/{synthetic_data.py,graph_generators.py,anonymized_production.py}
└── scripts/run_chaos_experiments.py
```

## 2. Synthetic Data Generators

```python
# fixtures/synthetic_data.py
def generate_transactions(n: int) -> list[dict]:
    """Faker-based transaction generation for unit tests."""
    from faker import Faker
    fake = Faker()
    return [{"tx_hash": fake.sha256(), "from": fake.hexadecimal(40), "amount": fake.random_number(digits=8)} for _ in range(n)]
```

```python
# fixtures/graph_generators.py
def generate_transaction_graph(n_nodes: int, n_edges: int) -> nx.DiGraph:
    """Graph generator for AML topology tests. Barabási-Albert for power-law."""
    G = nx.barabasi_albert_graph(n_nodes, n_edges // n_nodes)
    # Convert to directed AML-style graph
    return G.to_directed()
```

### [human] Graph generator fidelity
> Насколько графы от Barabási-Albert соответствуют реальным AML графам?

**Decision:** Barabási-Albert для topology. Реальные веса и timestamps из anonymized production. Power-law degree distribution сохранена. # ponytail: realism gap, add when topology mismatch > 30%.

### [agent] resolved
> BA для topology, реальные веса из anonymized production.

## 3. Chaos Engineering Tests

```python
# tests/chaos/test_kafka_down.py
def test_kafka_down_recovery():
    """Verify no message loss when Kafka is down for 5min."""
    # 1. Stop Kafka broker
    # 2. Produce messages to DLQ
    # 3. Restart Kafka
    # 4. Verify all messages processed (idempotent)
    assert message_count == expected_count
```

```python
# tests/chaos/test_gpu_oom.py
def test_gpu_oom_graceful_degradation():
    """Verify inference degrades gracefully on GPU OOM."""
    # Trigger OOM, verify fallback to CPU, latency < 500ms
    assert latency_p99 < 500 and no_data_loss
```

### [human] GPU OOM handling
> Что происходит при GPU OOM в production?

**Decision:** Автоматический fallback на CPU inference. Latency p99 < 500ms. Никакой потери данных. Alert в Prometheus. # ponytail: fallback strategy, add when CPU inference latency > 2s.

### [agent] resolved
> CPU fallback. Latency < 500ms. Alert + no data loss.

## 4. Key Components

- `synthetic_data.py`: Faker-based transaction generation. English comments.
- `graph_generators.py`: Barabási-Albert for AML topology. Power-law preserved. English comments.
- `anonymized_production.py`: 30% production data (anonymized). English comments.
- `test_kafka_down.py`, `test_memgraph_crash.py`, `test_gpu_oom.py`: Chaos scenarios. English comments.
- Coverage: 95% ML, 80% standard.

## 5. Pushback Hypotheses

1. **H1**: 95% ML coverage unattainable for stochastic models. *Mitigation*: Seed-based tests + 2σ tolerance bands. CI flaky < 5%. `# ponytail:` statistical testing
2. **H2**: Chaos causes data loss. *Mitigation*: Isolated environments + snapshots. Restore < 30min.

## 6. Self-Review

- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- Synthetic + production test data documented
- Pushback hypotheses: 2 ✅
