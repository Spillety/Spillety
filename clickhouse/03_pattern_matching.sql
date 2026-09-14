-- 03: structuring pattern (temp.md: pattern matching statistics).
-- Flags senders with >= min_tx_count transfers below amount_threshold
-- inside window_hours. Tune via clickhouse/config.yaml.
-- NOTE: no live ClickHouse in this env, syntax verified by review only.
-- Params: amount_threshold = 10000.0, min_tx_count = 10, window_hours = 24.

SELECT
    from_address,
    window_end,
    txs,
    max_amount
FROM
(
    SELECT
        from_address,
        max(event_time) OVER w AS window_end,
        count() OVER w AS txs,
        max(amount) OVER w AS max_amount,
        dateDiff('hour', min(event_time) OVER w, max(event_time) OVER w) AS span_h
    FROM aml.transactions_raw
    WHERE amount < 10000.0
    WINDOW w AS (
        PARTITION BY from_address
        ORDER BY event_time, tx_hash
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    )
)
WHERE txs >= 10 AND span_h <= 24
GROUP BY from_address, window_end, txs, max_amount
ORDER BY window_end DESC;
