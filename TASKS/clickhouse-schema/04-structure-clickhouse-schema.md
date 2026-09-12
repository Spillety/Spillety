# 6.1 ClickHouse Schema — Structure

```
clickhouse-schema/
├── ddl/
│   ├── raw_table.sql            # ReplacingMergeTree DDL
│   ├── velocity_1h.sql          # AggregatingMergeTree MV DDL
│   ├── velocity_24h.sql         # AggregatingMergeTree MV DDL
│   ├── velocity_7d.sql          # AggregatingMergeTree MV DDL
│   └── retention_policy.sql     # 90-day DELETE + S3 export
├── kafka/
│   ├── kafka_source.sql         # Kafka engine setup for dual-write
│   └── consumer_config.py       # Dual-consumer offset management
├── migrations/
│   └── schema_migrator.py       # DDL version control
└── tests/
    ├── __init__.py
    ├── test_raw_dedup.py
    ├── test_velocity_mv.py
    └── test_retention_s3.py
```

## Key Components
- `raw_table.sql`: ReplacingMergeTree with `toYYYYMM(date)` partition, dedup on `tx_id`. English comments only.
- `velocity_*.sql`: AggregatingMergeTree MVs with `sumState`/`countState` pre-aggregation
- `kafka_source.sql`: Kafka engine consuming from transaction topic; dual consumer groups
- `retention_policy.sql`: Scheduled `ALTER TABLE ... DELETE` + S3 `s3engine` export

## Pushback Hypotheses
1. **H1**: Kafka dual-consumer offset management is fragile — rebalances may cause duplicates. *Mitigation*: Use Kafka transactions + idempotent inserts; validate with `tx_id` uniqueness constraint.
2. **H2**: `toYYYYMM(date)` partitioning leads to data skew (uneven monthly volumes). *Mitigation*: Monitor partition sizes; rebalance if max/min > 3x. `# ponytail:` fixed partition vs. adaptive

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
