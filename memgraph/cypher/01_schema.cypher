// 01_schema.cypher — temp.md graph ontology (wave C in-place rewrite).
// Nodes: Wallet, Person, Mixer, Exchange, NewsArticle, Pattern.
// Edges: TRANSACTS, SAME_AS, MENTIONED_IN, MATCHES_PATTERN.
// Every edge carries valid_from / valid_to (NULL valid_to = still active).
// Hyperbolic Poincare-ball 128D embeddings live on Wallet, Mixer, Exchange.

// --- Uniqueness constraints (MERGE keys used by writers in src/ingestion.py) ---
CREATE CONSTRAINT ON (w:Wallet) ASSERT w.address IS UNIQUE;
CREATE CONSTRAINT ON (p:Person) ASSERT p.person_id IS UNIQUE;
CREATE CONSTRAINT ON (m:Mixer) ASSERT m.mixer_id IS UNIQUE;
CREATE CONSTRAINT ON (e:Exchange) ASSERT e.exchange_id IS UNIQUE;
CREATE CONSTRAINT ON (n:NewsArticle) ASSERT n.url IS UNIQUE;
CREATE CONSTRAINT ON (p:Pattern) ASSERT p.pattern_id IS UNIQUE;

// --- Node shapes (reference; Memgraph itself is schemaless) ---
// (:Wallet {address, hyperbolic_embedding, cluster_id, first_seen, hawkes_lambda})
// (:Person {person_id, kyc_status})
// (:Mixer {mixer_id, hyperbolic_embedding, cluster_id})
// (:Exchange {exchange_id, hyperbolic_embedding, cluster_id, jurisdiction})
// (:NewsArticle {url, title, lang, published_at})
// (:Pattern {pattern_id, name, description})

// --- Edge shapes (all temporal; actor is Wallet, Mixer or Exchange) ---
// (actor)-[:TRANSACTS {tx_hash, amount, asset, timestamp, valid_from, valid_to}]->(actor)
// (:Wallet)-[:SAME_AS {evidence, valid_from, valid_to}]->(:Person)
// (:Wallet)-[:MENTIONED_IN {snippet, valid_from, valid_to}]->(:NewsArticle)
// (:Wallet)-[:MATCHES_PATTERN {score, valid_from, valid_to}]->(:Pattern)
