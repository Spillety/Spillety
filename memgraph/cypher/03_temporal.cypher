// 03_temporal.cypher — temporal edge writes and point-in-time queries.
// Every edge type carries valid_from / valid_to; NULL valid_to = still active.

// Create a TRANSACTS edge with validity window
// MATCH (a:Wallet {address: $from}), (b:Wallet {address: $to})
// CREATE (a)-[:TRANSACTS {tx_hash: $tx_hash, amount: $amount, asset: $asset, timestamp: datetime($ts), valid_from: datetime($valid_from), valid_to: NULL}]->(b);

// Create a SAME_AS edge (KYC wallet linkage) with validity window
// MATCH (w:Wallet {address: $addr}), (p:Person {person_id: $person_id})
// CREATE (w)-[:SAME_AS {evidence: $evidence, valid_from: datetime($valid_from), valid_to: NULL}]->(p);

// Create a MENTIONED_IN edge (news mention) with validity window
// MATCH (w:Wallet {address: $addr}), (n:NewsArticle {url: $url})
// CREATE (w)-[:MENTIONED_IN {snippet: $snippet, valid_from: datetime($valid_from), valid_to: NULL}]->(n);

// Create a MATCHES_PATTERN edge (screening hit) with validity window
// MATCH (w:Wallet {address: $addr}), (p:Pattern {pattern_id: $pattern_id})
// CREATE (w)-[:MATCHES_PATTERN {score: $score, valid_from: datetime($valid_from), valid_to: NULL}]->(p);

// Query: all active TRANSACTS at a point in time
MATCH (a)-[t:TRANSACTS]->(b)
WHERE t.valid_from <= datetime('2026-09-12T00:00:00')
  AND (t.valid_to IS NULL OR t.valid_to >= datetime('2026-09-12T00:00:00'))
RETURN a.address, b.address, t.tx_hash, t.amount, t.valid_from, t.valid_to
ORDER BY t.timestamp DESC
LIMIT 1000;

// Query: TRANSACTS history for a wallet within a date range
MATCH (a:Wallet)-[t:TRANSACTS]->(b)
WHERE a.address = $address
  AND t.valid_from >= datetime($start_date)
  AND (t.valid_to IS NULL OR t.valid_to <= datetime($end_date))
RETURN b.address, t.tx_hash, t.amount, t.timestamp
ORDER BY t.timestamp DESC
LIMIT 500;

// Query: expire stale edges by setting valid_to
// MATCH ()-[e:TRANSACTS]->()
// WHERE e.valid_to IS NULL AND e.timestamp < datetime($cutoff)
// SET e.valid_to = datetime($cutoff)
// RETURN count(*);
