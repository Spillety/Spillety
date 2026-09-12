# 6.2 Historical Aggregations & Alerting — Design

## Context
SQL pattern matching for structuring transaction chains, ML anomaly detection on historical data, ClickHouse window functions `ROWS BETWEEN` for rolling metrics. Materialized View → Kafka for real-time alert propagation.

### [human] Design Decisions
- **SQL Pattern Matching**: Recursive CTEs to structure transaction chains (sender → intermediary → receiver); detect structuring via `amount < threshold` repeated ≥ 3 within 24h
- **ML Anomaly Detection**: Isolation Forest on features (amount, frequency, counterparty diversity); model served via ClickHouse `anomaly_score` function
- **Window Functions `ROWS BETWEEN`**: Rolling velocity `SUM(amount) OVER (PARTITION BY entity_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)`
- **MV → Kafka**: Materialized View on aggregated anomalies emits to Kafka topic `alerts_realtime` for downstream consumers

### [agent] resolved
- Structuring detection: Recursive CTE with depth limit 5; flag entities with ≥ 3 sub-threshold tx in 24h window
- Isolation Forest: Trained nightly on ClickHouse `s3`-exported features; model artifact loaded via `file` engine function
- `ROWS BETWEEN` window: Configurable frame for velocity (10 rows), volume (100 rows), and frequency (1 row)
- MV → Kafka: `Kafka` engine table writing anomaly scores; trigger on `anomaly_score > threshold`

## Architecture
```
Raw Transactions → Structuring CTE → Window Functions → ML Scoring → MV → Kafka
  ├── Recursive CTE: chain detection (sender→intermediary→receiver)
  ├── ROWS BETWEEN: rolling velocity/volume/frequency
  ├── Isolation Forest: anomaly_score per entity per day
  └── MV: anomaly_score > threshold → Kafka(alerts_realtime)
```

**Key Views**: `structuring_patterns`, `rolling_metrics`, `anomaly_scores`

## Pushback Hypotheses
1. **H1**: Recursive CTEs on large transaction graphs cause query timeouts. *Mitigation*: Limit depth to 5; use pre-computed adjacency cache; fall back to BFS in Python if CTE > 30s. `# ponytail:` SQL recursion vs. graph DB
2. **H2**: Isolation Forest model staleness — retrained nightly may miss intraday drift. *Mitigation*: Online anomaly threshold adjustment via `anomaly_score` UDF with exponential moving average. `# ponytail:` batch vs. online ML

## Open Questions
- What is the optimal `ROWS BETWEEN` frame size for velocity detection?
- Can Isolation Forest run inside ClickHouse or should it be external?

## [human] Acceptance
- Structuring detection recall ≥ 0.90 on labeled test set
- Window function queries complete < 5s on 100M rows
- MV → Kafka latency < 10s from anomaly score computation
- ML model AUC ≥ 0.85 on holdout set
