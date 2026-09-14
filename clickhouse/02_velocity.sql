-- 02: velocity aggregates over 01.transactions_raw (temp.md: velocity checks).
-- One AggregatingMergeTree table + materialized view per window (1h/24h/7d).
-- NOTE: no live ClickHouse in this env, syntax verified by review only.

CREATE TABLE IF NOT EXISTS aml.velocity_1h
(
    window_start DateTime('UTC'),
    address String,
    tx_count AggregateFunction(count, UInt8),
    total_amount AggregateFunction(sum, Float64),
    counterparties AggregateFunction(uniq, String)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (address, window_start)
TTL window_start + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS aml.velocity_24h
(
    window_start DateTime('UTC'),
    address String,
    tx_count AggregateFunction(count, UInt8),
    total_amount AggregateFunction(sum, Float64),
    counterparties AggregateFunction(uniq, String)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (address, window_start)
TTL window_start + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS aml.velocity_7d
(
    window_start DateTime('UTC'),
    address String,
    tx_count AggregateFunction(count, UInt8),
    total_amount AggregateFunction(sum, Float64),
    counterparties AggregateFunction(uniq, String)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (address, window_start)
TTL window_start + INTERVAL 90 DAY;

CREATE MATERIALIZED VIEW IF NOT EXISTS aml.mv_velocity_1h
TO aml.velocity_1h AS
SELECT
    toStartOfHour(event_time) AS window_start,
    from_address AS address,
    countState() AS tx_count,
    sumState(amount) AS total_amount,
    uniqState(to_address) AS counterparties
FROM aml.transactions_raw
GROUP BY window_start, address;

CREATE MATERIALIZED VIEW IF NOT EXISTS aml.mv_velocity_24h
TO aml.velocity_24h AS
SELECT
    toStartOfDay(event_time) AS window_start,
    from_address AS address,
    countState() AS tx_count,
    sumState(amount) AS total_amount,
    uniqState(to_address) AS counterparties
FROM aml.transactions_raw
GROUP BY window_start, address;

CREATE MATERIALIZED VIEW IF NOT EXISTS aml.mv_velocity_7d
TO aml.velocity_7d AS
SELECT
    toStartOfWeek(event_time) AS window_start,
    from_address AS address,
    countState() AS tx_count,
    sumState(amount) AS total_amount,
    uniqState(to_address) AS counterparties
FROM aml.transactions_raw
GROUP BY window_start, address;
