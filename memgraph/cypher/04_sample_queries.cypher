// 04_sample_queries.cypher — Traversal, temporal, and vector search queries

// --- Traversal Query ---
// BFS traversal up to depth 3 from an address via TRANSFER edges
// Uses hash index on address for O(1) lookup, then traverses edges
MATCH (a:Address {address: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb'})-[:TRANSFER*1..3]->(b:Address)
RETURN b.address, b.cluster_id
LIMIT 100;

// --- Temporal Query ---
// Find all transfers within a specific time window using valid_from/valid_to
MATCH (a:Address)-[t:TRANSFER]->(b:Address)
WHERE t.valid_from <= datetime('2026-09-01')
  AND (t.valid_to IS NULL OR t.valid_to >= datetime('2026-09-01'))
RETURN a.address, b.address, t.amount, t.timestamp
ORDER BY t.timestamp DESC
LIMIT 1000;

// --- Vector Search (HNSW) ---
// Nearest scam cluster query via HNSW vector search on hyperbolic_embedding
// O(log n) nearest neighbor search in hyperbolic space
MATCH (a:Address)
WHERE a.hyperbolic_embedding IS NOT NULL
RETURN a.address, a.cluster_id
ORDER BY a.hyperbolic_embedding <-> $query_vector
LIMIT 10;

// --- Co-spend Clustering ---
// Traverse :CO_SPEND edges to find co-spending relationships
MATCH (a:Address)-[:CO_SPEND]->(b:Address)
WHERE a.cluster_id IS NOT NULL AND b.cluster_id IS NOT NULL
RETURN a.address, b.address, shared_input_count
ORDER BY shared_input_count DESC
LIMIT 50;

// --- Risk/Sanctions Query ---
// Find addresses flagged by sanctions
MATCH (a:Address)-[s:SANCTIONS_FLAG]->(r:Risk)
WHERE s.list = $sanctions_list
RETURN a.address, r.risk_type, s.date
ORDER BY s.date DESC
LIMIT 100;

// --- Entity Risk Query ---
// Get entities with risk scores above threshold
MATCH (e:Entity)
WHERE e.risk_score > $threshold
RETURN e.entity_name, e.risk_score, e.is_exchange
ORDER BY e.risk_score DESC
LIMIT 50;
