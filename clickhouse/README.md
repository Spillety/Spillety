# ClickHouse OLAP (temp.md: velocity checks, historical aggregations, pattern matching statistics)

Greenfield, живого ClickHouse в окружении нет — SQL выверен ревью, исполнение невозможно (см. итог ревью в отчёте).

## Порядок применения

1. `01_transactions_raw.sql` — `aml.transactions_raw` (`ReplacingMergeTree`), зеркало Kafka `raw-events`, TTL 90d.
2. `02_velocity.sql` — `aml.velocity_1h/24h/7d` (`AggregatingMergeTree`) + materialized views.
3. `03_pattern_matching.sql` — structuring: N переводов ниже порога в окне (`ROWS BETWEEN`).

```bash
clickhouse-client --multiquery < clickhouse/01_transactions_raw.sql
clickhouse-client --multiquery < clickhouse/02_velocity.sql
clickhouse-client --multiquery < clickhouse/03_pattern_matching.sql
```

## Пороги и окна

Все магические числа — в `config.yaml` рядом: окна `1h/24h/7d`, structuring
(`amount_threshold`, `min_tx_count`, `window_hours`), `retention_days`.
Значения structuring — `# ponytail` (первичные догадки до калибровки на истории алертов).

## Чтение агрегатов

```sql
SELECT address, sumMerge(total_amount) AS volume_24h
FROM aml.velocity_24h
WHERE window_start >= now() - INTERVAL 1 DAY
GROUP BY address;
```
