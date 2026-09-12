// 02_indexes.cypher — Property indexes, composite index, and HNSW vector index

// Property index (hash) on address for O(1) lookup via hash index
CREATE INDEX address_hash_index FOR (a:Address) ON (a.address);

// # ponytail: composite index, add when temporal query latency > 50ms
// Composite index on temporal fields (valid_from, valid_to) for TRANSFER edge
// Defer until query profiling shows temporal filter is a bottleneck.
// Memgraph supports composite indexes on edge properties starting from v2.14.
CREATE INDEX IF NOT EXISTS transfer_temporal_idx FOR ()-[e:TRANSFER]-() ON (e.valid_from, e.valid_to);

// HNSW vector index on hyperbolic_embedding for nearest scam cluster query
// 128D float32, ef_construction=200, M=32, cosine distance
// # ponytail: HNSW parameter tuning, add when recall@k < 0.95
CREATE VECTOR INDEX hyperbolic_embedding_hnsw FOR (a:Address) ON (a.hyperbolic_embedding)
OPTIONS {
    type = 'hnsw',
    dimensions = 128,
    metric = 'cosine',
    ef_construction = 200,
    M = 32
};
