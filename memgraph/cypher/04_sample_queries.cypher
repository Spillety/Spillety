// 04_sample_queries.cypher — traversal, temporal, vector, and ontology queries.

// --- Traversal Query ---
// Local subgraph depth 2 from a wallet via TRANSACTS (lazy inference input).
MATCH (w:Wallet {address: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb'})-[:TRANSACTS*1..2]-(n)
RETURN labels(n) AS labels, n.address AS address, n.cluster_id AS cluster_id
LIMIT 100;

// --- Temporal Query ---
// Active TRANSACTS at a point in time using valid_from/valid_to.
MATCH (a)-[t:TRANSACTS]->(b)
WHERE t.valid_from <= datetime('2026-09-01')
  AND (t.valid_to IS NULL OR t.valid_to >= datetime('2026-09-01'))
RETURN a.address, b.address, t.tx_hash, t.amount
ORDER BY t.timestamp DESC
LIMIT 1000;

// --- Vector Search (HNSW) ---
// Nearest scam cluster across embedded labels for hyperbolic_distance.
MATCH (w:Wallet)
WHERE w.hyperbolic_embedding IS NOT NULL
RETURN w.address AS node_id, 'Wallet' AS kind, w.hyperbolic_embedding <-> $query_vector AS distance
UNION ALL
MATCH (m:Mixer)
WHERE m.hyperbolic_embedding IS NOT NULL
RETURN m.mixer_id AS node_id, 'Mixer' AS kind, m.hyperbolic_embedding <-> $query_vector AS distance
UNION ALL
MATCH (e:Exchange)
WHERE e.hyperbolic_embedding IS NOT NULL
RETURN e.exchange_id AS node_id, 'Exchange' AS kind, e.hyperbolic_embedding <-> $query_vector AS distance
ORDER BY distance
LIMIT 10;

// --- SAME_AS Lookup ---
// Wallets linked to a person via KYC evidence.
MATCH (w:Wallet)-[s:SAME_AS]->(p:Person {person_id: $person_id})
WHERE s.valid_from <= datetime($at) AND (s.valid_to IS NULL OR s.valid_to >= datetime($at))
RETURN w.address, s.evidence, s.valid_from
LIMIT 50;

// --- MENTIONED_IN Query ---
// News articles mentioning a wallet.
MATCH (w:Wallet {address: $address})-[m:MENTIONED_IN]->(n:NewsArticle)
RETURN n.url, n.title, m.snippet, m.valid_from
ORDER BY m.valid_from DESC
LIMIT 100;

// --- MATCHES_PATTERN Query ---
// Pattern hits above score threshold.
MATCH (w:Wallet)-[m:MATCHES_PATTERN]->(p:Pattern)
WHERE m.score > $threshold
RETURN w.address, p.pattern_id, p.name, m.score
ORDER BY m.score DESC
LIMIT 50;
