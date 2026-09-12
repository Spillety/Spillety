# 2.3 Iceberg Audit Trail — Design

## Context
Iceberg on S3 + Trino for compliance audit trail. Schema evolution (backward-compatible only), daily partitioning, 7-year retention. Hot data (90 days) in ClickHouse, cold in Iceberg.

### [human] Design Decisions
- **Iceberg over Delta**: Open format, regulator-friendly, no Spark lock-in
- **Partitioning**: `date` (daily). Alert_id hash not needed — queries are temporal
- **Schema evolution**: Backward-compatible only; new fields optional; breaking changes via new table version
- **Retention**: 7 years (compliance). Hot 90d → ClickHouse, cold → Iceberg on S3

### [agent] resolved
- Table format: Iceberg 1.0 with Hive-compatible table properties
- Trino connector for ad-hoc SQL queries
- Partition evolution via `rewritePartitions` for schema changes
- S3 lifecycle policy: transition to Glacier after 1 year

## Architecture
```
Kafka → ClickHouse (90d hot) → Iceberg S3 (cold, 7y)
  ├── Trino: ad-hoc SQL queries
  ├── Schema evolution: ADD COLUMNS only (backward compat)
  └── Partition: date (YYYY-MM-DD)
```

**Table properties**:
- `format-version`: 2
- `partition-policy`: daily by `event_date`
- `write.format.default`: parquet
- `retention`: 7 years (S3 lifecycle + Iceberg snapshot expiration)

## Pushback Hypotheses
1. **H1**: Daily partitions create too many small files for 7-year retention (~2555 partitions). *Mitigation*: Auto-compact via Iceberg `RewriteDataFiles` nightly; bucket smaller dates into monthly partitions after 1 year. `# ponytail:` partition granularity
2. **H2**: Trino query latency on S3 cold data may exceed 1s SLA. *Mitigation*: Use Trino caching + Parquet columnar indexing; hot-path queries route to ClickHouse instead.

## Open Questions
- How many records per day to size S3 storage and query performance?
- Is Trino the right query engine, or would Athena/Presto be simpler?

## [human] Acceptance
- Schema evolution backward-compatible test passes
- Query latency < 1s for 90-day range
- 7-year retention policy enforced and tested
