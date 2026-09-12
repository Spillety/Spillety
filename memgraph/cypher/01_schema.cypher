// 01_schema.cypher — Node & Edge table definitions for Memgraph
// Node type: Address (blockchain address with hyperbolic embedding for scam clustering)
CREATE NODE TABLE IF NOT EXISTS Address (
    address STRING,
    hyperbolic_embedding FLOAT[128],
    cluster_id STRING,
    PRIMARY KEY(address)
);

// Node type: Entity (identified entity — exchange, wallet cluster, etc.)
CREATE NODE TABLE IF NOT EXISTS Entity (
    entity_name STRING,
    risk_score FLOAT,
    is_exchange BOOLEAN,
    PRIMARY KEY(entity_name)
);

// Node type: Risk (sanctions and risk classification data)
CREATE NODE TABLE IF NOT EXISTS Risk (
    risk_type STRING,
    sanctions_list STRING,
    PRIMARY KEY(risk_type)
);

// Edge type: TRANSFER (temporal transfer between addresses)
CREATE EDGE TABLE IF NOT EXISTS TRANSFER (
    amount FLOAT,
    timestamp DATETIME,
    valid_from DATETIME,
    valid_to DATETIME
);

// Edge type: CO_SPEND (co-spend clustering — shared inputs heuristic)
CREATE EDGE TABLE IF NOT EXISTS CO_SPEND (
    shared_input_count INTEGER
);

// Edge type: MENTIONED_IN (contextual mention edge)
CREATE EDGE TABLE IF NOT EXISTS MENTIONED_IN (
    context STRING
);

// Edge type: SANCTIONS_FLAG (sanctions flagging edge)
CREATE EDGE TABLE IF NOT EXISTS SANCTIONS_FLAG (
    list STRING,
    date DATETIME
);
