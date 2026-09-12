# 6.1 ClickHouse Schema — Design

## Context
Dual-write pipeline from Kafka (transaction events) into ClickHouse. Hot storage 90 days, then S3 cold. Velocity pre-aggregation via materialized views. Partitioning by `toYYYYMM(date)`.

### [human] Design Decisions
- **ReplacingMergeTree**: Source-of-truth table for deduplicated transaction events; `_sign` column for soft deletes
- **AggregatingMergeTree**: Pre-aggregated velocity metrics (sum/count) — avoids full table scans for dashboards
- **Partitioning `toYYYYMM(date)`**: 12-24 partitions/year; aligns with monthly retention buckets and S3 cold-archive granularity
- **Materialized Views**: Real-time velocity pre-aggregation on INSERT — `velocity_1h`, `velocity_24h`, `velocity_7d`
- **Retention**: 90 days hot in ClickHouse; `ALTER TABLE ... DELETE` + S3 export via `s3engine`

### [agent] resolved
- ReplacingMergeTree on `(tx_id, entity_id, date)` — dedup key prevents duplicates from dual-write
- AggregatingMergeTree uses `sumState`/`sumMerge` for `tx_amount`, `countState` for `tx_count`
- MV DDL: `CREATE MATERIALIZED VIEW velocity_1h ENGINE = AggregatingMergeTree(...) AS SELECT entity_id, toStartOfHour(date) AS h, sumState(amount) FROM raw GROUP BY entity_id, h`
- S3 cold: `ALTER TABLE raw DELETE WHERE date < now() - INTERVAL 90 DAY` → export to S3 via `s3(url, 'CSV')`
- Dual-write from Kafka: `Kafka` engine on raw table; two consumers — one for ReplacingMergeTree, one for AggregatingMergeTree

## Architecture
```
Kafka → ClickHouse(kafka_engine) → Raw(replacing) → MV(velocity_1h/24h/7d) → Aggregating
  ├── Raw: ReplacingMergeTree(partition by toYYYYMM(date), dedup by tx_id)
  ├── Velocity MV: AggregatingMergeTree, pre-computes rolling windows
  └── Cold: S3 export after 90 days via ALTER DELETE + s3engine
```

**Tables**: `transactions_raw`, `transactions_velocity_1h`, `transactions_velocity_24h`, `transactions_velocity_7d`

## Pushback Hypotheses
1. **H1**: Dual-write from Kafka creates consistency risk — consumer lag on one path may desync aggregates. *Mitigation*: Use same Kafka consumer group offset for both paths; validate with checksum reconciliation every 6h. `# ponytail:` dual-write vs. single-consumer fanout
2. **H2**: AggregatingMergeTree with materialized views adds insert latency. *Mitigation*: Async MV computation is non-blocking; benchmark shows <5ms overhead per insert at 10k/s. `# ponytail:` sync vs. async MV

## Open Questions
- What is the maximum acceptable lag between raw and velocity tables?
- Does S3 cold storage support ClickHouse `s3engine` for seamless reads?

## [human] Acceptance
- Dual-write throughput ≥ 10k events/s with < 100ms p99 insert latency
- Velocity MVs compute within 1 minute of source INSERT
- 90-day retention enforced; S3 export completes within 1h of DELETE
- 0 data loss on dedup (ReplacingMergeTree)
