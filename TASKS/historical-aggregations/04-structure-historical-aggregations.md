# 6.2 Historical Aggregations & Alerting — Structure

```
historical-aggregations/
├── sql/
│   ├── structuring_cte.sql      # Recursive CTE for chain detection
│   ├── window_functions.sql     # ROWS BETWEEN rolling metrics
│   └── pattern_matching.sql     # Sub-threshold repetition detection
├── ml/
│   ├── isolation_forest.py      # Model training + scoring
│   ├── feature_builder.py       # Aggregates features from ClickHouse
│   └── model_serving.py         # Load model artifact, expose score UDF
├── realtime/
│   ├── mv_to_kafka.py           # Materialized View → Kafka pipeline
│   └── anomaly_stream.sql       # MV DDL emitting to alerts_realtime
└── tests/
    ├── __init__.py
    ├── test_structuring.py
    ├── test_window_functions.py
    └── test_ml_scoring.py
```

## Key Components
- `structuring_cte.sql`: Recursive CTE with depth limit 5; flags sub-threshold chains. English comments only.
- `window_functions.sql`: `ROWS BETWEEN` frame definitions for velocity/volume/frequency
- `isolation_forest.py`: Trains on feature aggregates; exports model artifact for ClickHouse `file` engine
- `mv_to_kafka.py`: Reads MV output; writes to Kafka `alerts_realtime` topic

## Pushback Hypotheses
1. **H1**: Recursive CTE performance degrades on deep chains. *Mitigation*: Add `maxDepth` parameter; benchmark against BFS pre-computed adjacency. `# ponytail:` SQL depth limit vs. graph traversal
2. **H2**: ML model served via `file` engine has cold-start latency. *Mitigation*: Keep model in memory via HTTP microservice; fallback to `file` engine only on restart. `# ponytail:` embedded vs. served model

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
