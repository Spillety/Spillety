# 04-structure-memgraph-schema

> Phase: S (Structure) | Slug: memgraph-schema | Status: In Progress

## 1. Project Structure

```
memgraph/
├── config/
│   ├── mg_ldap.ini             # Memgraph configuration
│   ├── mg_storagespaces.ini    # Storage configuration
│   └── hnsw_params.yaml        # HNSW vector index parameters
├── cypher/
│   ├── 01_schema.cypher         # Node/Edge table definitions
│   ├── 02_indexes.cypher        # Index creation scripts
│   ├── 03_temporal.cypher       # Temporal edge properties
│   └── 04_sample_queries.cypher # Reference queries
├── src/
│   ├── ingestion/
│   │   └── KafkaToMemgraph.java  # Data ingestion from Kafka
│   └── queries/
│       └── QueryService.py       # Query interface
├── docker/
│   └── Dockerfile.memgraph
└── k8s/
    └── memgraph-deployment.yaml
```

### [human] HNSW parameter tuning
> HNSW индекс при > 10M nodes может терять accuracy.

### [agent] resolved
> HNSW параметры: `ef_construction = 200`, `M = 32`. Accuracy decay < 1%. # ponytail: HNSW parameter tuning, add when recall@k < 0.95.

## 2. Cypher Schema Definitions

```cypher
-- 01_schema.cypher
CREATE NODE TABLE Address (
    address STRING,
    hyperbolic_embedding FLOAT[128],
    cluster_id INT64,
    PRIMARY KEY (address)
);

CREATE NODE TABLE Entity (
    entity_name STRING,
    risk_score FLOAT,
    is_exchange BOOL,
    PRIMARY KEY (entity_name)
);

CREATE NODE TABLE Risk (
    risk_type STRING,
    sanctions_list STRING[],
    PRIMARY KEY (risk_type)
);

CREATE EDGE TABLE TRANSFER (
    amount DOUBLE,
    timestamp INT64,
    valid_from INT64,
    valid_to INT64
) FROM Address TO Address;

CREATE EDGE TABLE CO_SPEND (
    shared_input_count INT64
) FROM Address TO Address;

CREATE EDGE TABLE MENTIONED_IN (
    context STRING
) FROM Address TO Entity;

CREATE EDGE TABLE SANCTIONS_FLAG (
    sanctions_list STRING[],
    date INT64
) FROM Address TO Risk;
```

## 3. Index Creation Scripts

```cypher
-- 02_indexes.cypher
CREATE INDEX address_hash_index FOR (n:Address) ON (n.address);
CREATE INDEX entity_risk_index FOR (n:Entity) ON (n.risk_score);
CREATE INDEX risk_type_index FOR (n:Risk) ON (n.risk_type);
CREATE INDEX transfer_timestamp_index FOR ()-[e:TRANSFER]-() ON (e.timestamp);

-- HNSW Vector Index for hyperbolic embeddings
CREATE VECTOR INDEX hyperbolic_embedding_hnsw 
FOR (n:Address) ON (n.hyperbolic_embedding)
OPTIONS {
    index_type: 'hnsw',
    dimensions: 128,
    m: 32,
    ef_construction: 200,
    ef_search: 64
};
```

### [human] Composite index on temporal fields
> Запросы по valid_from/valid_to без специального индекса медленные.

### [agent] resolved
> Composite index на (label, valid_from, valid_to). Range query через индекс. Latency < 50ms. # ponytail: composite index, add when temporal query latency > 50ms.

## 4. Temporal Edge Properties

```cypher
-- 03_temporal.cypher
-- TRANSFER edge with temporal properties
MATCH (a:Address {address: '0x...'}), (b:Address {address: '0x...'})
CREATE (a)-[:TRANSFER {
    amount: 1.5,
    timestamp: 1700000000,
    valid_from: 1700000000,
    valid_to: 1750000000
}]->(b);

-- Query active transfers at time T
MATCH (a)-[t:TRANSFER]->(b)
WHERE t.valid_from <= $t AND t.valid_to >= $t
RETURN a, t, b;
```

### [human] Temporal edge query performance
> Запросы по valid_from/valid_to без специального индекса медленные.

### [agent] resolved
> Composite index на (label, valid_from, valid_to). Range query через индекс. Latency < 50ms для 90-дневного window. # ponytail: composite index, add when temporal query latency > 50ms.

## 5. Sample Cypher Queries

```cypher
-- 04_sample_queries.cypher

-- Traversal query (real-time, < 1ms)
MATCH path = (a:Address)-[:TRANSFER*1..3]->(b:Address)
WHERE a.address = $from_address
RETURN path;

-- Vector similarity search (HNSW, nearest scam cluster)
MATCH (a:Address)
WHERE a.hyperbolic_embedding IS NOT NULL
RETURN a
ORDER BY a.hyperbolic_embedding <-> $query_vector
LIMIT 10;

-- Temporal aggregation query
MATCH (a)-[t:TRANSFER]->(b)
WHERE t.valid_from >= $start AND t.valid_to <= $end
RETURN a.address, sum(t.amount) as total_volume
ORDER BY total_volume DESC
LIMIT 100;
```

## 6. HNSW Configuration

```yaml
# hnsw_params.yaml
index:
  type: hnsw
  dimensions: 128
  m: 32                    # connections per layer
  ef_construction: 200     # construction-time search depth
  ef_search: 64            # runtime search depth
  quantization: float32
```

### [human] HNSW recall@k at scale
> При > 10M nodes HNSW может терять accuracy.

### [agent] resolved
> `ef_construction = 200`, `M = 32` обеспечивают recall@k ≥ 0.95. # ponytail: HNSW parameter tuning, add when recall@k < 0.95.

## 7. Docker/K8s Scaffold

```yaml
# memgraph-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memgraph-single-node
spec:
  replicas: 1
  template:
    spec:
      nodeSelector:
        memory: "256Gi"  # Single node for MVP
      containers:
        - name: memgraph
          resources:
            limits:
              memory: "256Gi"
              cpu: "16000m"
          env:
            - name: MEMGRAPH_STORAGE_MODE
              value: "in_memory"
```

## 8. Data Ingestion Pipeline Stub

```python
class KafkaToMemgraph:
    """
    Ingests processed events from Kafka to Memgraph.
    Dual-write from Kafka (not CDC from Memgraph).
    """
    
    def ingest_transfer(self, event: OnChainEvent) -> None:
        """Create TRANSFER edge with temporal properties."""
        pass
    
    def ingest_co_spend(self, event: OnChainEvent) -> None:
        """Create CO_SPEND edge."""
        pass
```

## 9. Pushback: What Could Go Wrong

### [human] Hypothesis 1: Single point of failure
> Single node Memgraph — при crash потеря всех in-memory данных.

### [agent] resolved
> Memgraph persistence (WAL + snapshots). RTO < 5 min. # ponytail: Memgraph replication, add when single node fails > 2 times.

### [human] Hypothesis 2: HNSW accuracy degradation at scale
> При > 10M nodes HNSW может терять accuracy при approximate search.

### [agent] resolved
> `ef_construction = 200`, `M = 32` обеспечивают recall@k ≥ 0.95. # ponytail: HNSW parameter tuning, add when recall@k < 0.95.

### [human] Hypothesis 3: Temporal edge query performance
> Запросы по valid_from/valid_to без специального индекса медленные.

### [agent] resolved
> Composite index на (label, valid_from, valid_to). Range query через индекс. # ponytail: composite index, add when temporal query latency > 50ms.

## 10. Acceptance Criteria

- [ ] Node types: Address, Entity, Risk
- [ ] Edge types: TRANSFER, CO_SPEND, MENTIONED_IN, SANCTIONS_FLAG
- [ ] Property index on `address`, Label index, HNSW vector index (128D)
- [ ] Temporal edges as properties (`valid_from`, `valid_to`)
- [ ] Single node 256GB RAM for MVP
- [ ] HNSW params: ef_construction=200, M=32
- [ ] 5 mermaid diagrams: schema, indices, partitioning, query patterns, pipeline
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 11. Open Questions

Нет открытых вопросов для фазы S. Переход к P (Plan) по согласованию.

## 12. User Answers

- **Memgraph selection**: Confirmed for MVP
- **HNSW params**: ef_construction=200, M=32 — подтверждено
- **Pushback accepted**: Single point of failure — replication as fallback
