// 02_indexes.cypher — point-lookup, composite temporal, and HNSW vector indexes.
// HNSW params mirror hnsw_params.yaml (128D, cosine, ef_construction=200, M=32).

// Point lookups on MERGE keys (back the uniqueness constraints in 01_schema).
CREATE INDEX wallet_address_idx FOR (w:Wallet) ON (w.address);
CREATE INDEX person_id_idx FOR (p:Person) ON (p.person_id);
CREATE INDEX mixer_id_idx FOR (m:Mixer) ON (m.mixer_id);
CREATE INDEX exchange_id_idx FOR (e:Exchange) ON (e.exchange_id);
CREATE INDEX news_url_idx FOR (n:NewsArticle) ON (n.url);
CREATE INDEX pattern_id_idx FOR (p:Pattern) ON (p.pattern_id);

// Composite temporal indexes on every edge type for point-in-time queries.
CREATE INDEX transacts_temporal_idx FOR ()-[e:TRANSACTS]-() ON (e.valid_from, e.valid_to);
CREATE INDEX same_as_temporal_idx FOR ()-[e:SAME_AS]-() ON (e.valid_from, e.valid_to);
CREATE INDEX mentioned_in_temporal_idx FOR ()-[e:MENTIONED_IN]-() ON (e.valid_from, e.valid_to);
CREATE INDEX matches_pattern_temporal_idx FOR ()-[e:MATCHES_PATTERN]-() ON (e.valid_from, e.valid_to);

// HNSW vector indexes for nearest scam cluster lookup (hyperbolic_distance).
// Per-query recall tuning uses ef_runtime=50, see hnsw_params.yaml.
CREATE VECTOR INDEX wallet_hyperbolic_hnsw FOR (w:Wallet) ON (w.hyperbolic_embedding)
OPTIONS {type = 'hnsw', dimensions = 128, metric = 'cosine', ef_construction = 200, M = 32};
CREATE VECTOR INDEX mixer_hyperbolic_hnsw FOR (m:Mixer) ON (m.hyperbolic_embedding)
OPTIONS {type = 'hnsw', dimensions = 128, metric = 'cosine', ef_construction = 200, M = 32};
CREATE VECTOR INDEX exchange_hyperbolic_hnsw FOR (e:Exchange) ON (e.hyperbolic_embedding)
OPTIONS {type = 'hnsw', dimensions = 128, metric = 'cosine', ef_construction = 200, M = 32};
// # ponytail: HNSW parameter tuning, add when recall@k < 0.95
