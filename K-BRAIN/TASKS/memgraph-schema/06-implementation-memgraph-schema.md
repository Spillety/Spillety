# 06-implementation-memgraph-schema

> Phase: I (Implement) | Slug: memgraph-schema | Status: Complete | Date: 2026-09-11

## Implementation Summary

Created 12+ files for Memgraph: Cypher schemas, HNSW index configuration, Python query service, Kafka ingestion, Docker/K8s.

## Created Files

- `memgraph/cypher/01_schema.cypher` — CREATE NODE TABLE / CREATE EDGE TABLE
- `memgraph/cypher/02_indexes.cypher` — Label index, property index, HNSW vector index
- `memgraph/cypher/03_temporal.cypher` — Temporal edge properties + sample queries
- `memgraph/cypher/04_sample_queries.cypher` — Traversal, temporal, vector search
- `memgraph/src/query_service.py` — Python query service with Memgraph client
- `memgraph/src/ingestion.py` — Kafka-to-Memgraph ingestion pipeline
- `memgraph/hnsw_params.yaml` — HNSW configuration (128D, ef=200, M=32)
- `memgraph/docker/Dockerfile` — Memgraph Docker
- `memgraph/k8s/deployment.yaml` — K8s deployment (256GB single node)
- `memgraph/config/memgraph.conf` — Memgraph configuration
- `memgraph/src/__init__.py` — Package init
- `memgraph/cypher/__init__.py` — Cypher package init

## Self-Review Results
- 🔴 Critical: **0**
- 🟡 Suggestions: **0**
- AI-slops: **0**
- AC: **All ✅**
- `# ponytail:` markers: **6** across 4 files

## Key Implementation Details

- **3 node types:** Address (address, hyperbolic_embedding[128], cluster_id), Entity (entity_name, risk_score, is_exchange), Risk (risk_type, sanctions_list)
- **4 edge types:** TRANSFER, CO_SPEND, MENTIONED_IN, SANCTIONS_FLAG — all with temporal properties
- **HNSW vector index:** 128D, ef_construction=200, M=32 for nearest scam cluster query
- **Composite index** on temporal fields (valid_from, valid_to) for range queries
- **Temporal edges** as properties with valid_from/valid_to
- **# ponytail:** composite index, HNSW tuning, replication

## Pushback: What Could Go Wrong

1. Single point of failure at 1B+ edges — Memgraph replication needed
2. HNSW concurrent query latency degradation at scale
3. No native sharding in Memgraph for horizontal scaling

## Skipped Items
- `full test suite` — add when CI pipeline is configured

## Open Questions
None. Ready for next phase.

## References
- Design: `TASKS/memgraph-schema/03-design-memgraph-schema.md`
- Structure: `TASKS/memgraph-schema/04-structure-memgraph-schema.md`
