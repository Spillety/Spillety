-- 01: raw on-chain transactions, OLAP mirror of Kafka `raw-events`.
-- Apply order: 01 -> 02 -> 03 (see README.md).
-- NOTE: no live ClickHouse in this env, syntax verified by review only.

CREATE DATABASE IF NOT EXISTS aml;

CREATE TABLE IF NOT EXISTS aml.transactions_raw
(
    tx_hash String,
    event_time DateTime64(3, 'UTC'),
    from_address String,
    to_address String,
    amount Float64,
    asset LowCardinality(String),
    block_number UInt64,
    updated_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(event_time)
ORDER BY (from_address, to_address, event_time, tx_hash)
TTL event_time + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;
