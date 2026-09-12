// 03_temporal.cypher — Temporal edge property setup and sample temporal queries
// Temporal edges are implemented as properties (valid_from, valid_to) on the edge,
// not as separate edge types — avoids schema explosion per design decision.

// Create a temporal TRANSFER edge with validity window
// MATCH (a:Address {address: $from}), (b:Address {address: $to})
// CREATE (a)-[:TRANSFER {amount: $amount, timestamp: datetime(), valid_from: datetime(), valid_to: NULL}]->(b);

// Create a CO_SPEND edge with temporal context
// MATCH (a:Address {address: $a}), (b:Address {address: $b})
// CREATE (a)-[:CO_SPEND {shared_input_count: $count, valid_from: datetime(), valid_to: NULL}]->(b);

// Create a SANCTIONS_FLAG edge with temporal validity
// MATCH (a:Address {address: $addr}), (r:Risk {risk_type: $type})
// CREATE (a)-[:SANCTIONS_FLAG {list: $list, date: datetime(), valid_from: datetime(), valid_to: NULL}]->(r);

// Query: All active transfers at a point in time
// Returns transfers where valid_from <= now AND (valid_to IS NULL OR valid_to >= now)
MATCH (a:Address)-[t:TRANSFER]->(b:Address)
WHERE t.valid_from <= datetime('2026-09-12T00:00:00')
  AND (t.valid_to IS NULL OR t.valid_to >= datetime('2026-09-12T00:00:00'))
RETURN a.address, b.address, t.amount, t.timestamp, t.valid_from, t.valid_to
ORDER BY t.timestamp DESC
LIMIT 1000;

// Query: Transfer history for an address within a date range
MATCH (a:Address)-[t:TRANSFER]->(b:Address)
WHERE a.address = $address
  AND t.valid_from >= datetime($start_date)
  AND t.valid_to <= datetime($end_date)
RETURN b.address, t.amount, t.timestamp
ORDER BY t.timestamp DESC
LIMIT 500;

// Query: Expire transfers by setting valid_to
// MATCH (a:Address)-[t:TRANSFER]->(b:Address)
// WHERE t.valid_to IS NULL AND t.timestamp < datetime($cutoff)
// SET t.valid_to = datetime($cutoff)
// RETURN count(*);
