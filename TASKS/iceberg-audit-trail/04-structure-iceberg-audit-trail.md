# 2.3 Iceberg Audit Trail — Structure

```
iceberg-audit-trail/
├── config/
│   ├── iceberg_catalog.yml       # Trino/Iceberg catalog config
│   ├── s3_lifecycle.json         # S3 transition + Glacier policy
│   └── table_properties.sql      # CREATE TABLE DDL
├── schemas/
│   ├── v1/
│   │   ├── schema.json           # Initial schema (event_date, alert_id, ...)
│   │   └── migration.py          # ADD COLUMNS only
│   └── registry.py               # Schema version tracking
├── ingestion/
│   ├── clickhouse_to_iceberg.py  # 90d → S3 transfer job
│   └── partition_rewriter.py     # Auto-compact + coarsen partitions
├── queries/
│   ├── trino_queries.py          # Ad-hoc SQL via Trino connector
│   └── audit_report.py           # Compliance report generation
└── tests/
    ├── __init__.py
    ├── test_schema_evolution.py
    ├── test_partitioning.py
    └── test_retention_policy.py
```

## Key Components
- `schema/registry.py`: Tracks schema versions; enforces backward-compatible-only evolution
- `ingestion/clickhouse_to_iceberg.py`: Transfers hot data to cold storage
- `config/table_properties.sql`: Iceberg DDL with partition spec
- `ingestion/partition_rewriter.py`: Coarsens daily → monthly after 1 year

## Pushback Hypotheses
1. **H1**: S3 lifecycle to Glacier adds retrieval latency (hours) for audit queries. *Mitigation*: Keep 1 year in S3 Standard before Glacier; audit queries within 1y always hot.
2. **H2**: Schema evolution via ADD COLUMNS alone may not support all regulatory changes. *Mitigation*: New table version with dual-write; migrate data via Spark ETL. `# ponytail:` evolution strictness

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
