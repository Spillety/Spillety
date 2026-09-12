# 9.1 Feature Store — Design

> Phase: D (Design) | Slug: feature-store | Status: In Progress

## 1. Feast Selection & On-Chain Features

Feast open-source для AML-требований. On-chain features: tx_count, balance_history, cluster_risk_score. Derived: velocity, Hawkes λ(t). Dual-write Kafka → Feature Store + Flink.

### [human] Feast vs Tecton
> Tecton managed, Feast self-hosted. Для регулятора нужен полный контроль.

**Decision:** Feast. Open-source, no vendor lock-in, полный аудит. # ponytail: Tecton fallback, add when Feast operational overhead > 40h/week.

### [agent] resolved
> Feast выбран. Online — Redis. Offline — PostgreSQL → ClickHouse при масштабировании.

### [human] Feature schema design
> Какие on-chain и derived features включить в baseline?

**Decision:** On-chain: tx_count (1h/24h), balance_history (sliding window), cluster_risk_score (from entity resolution). Derived: velocity (tx_count/time_window), Hawkes λ(t) (from Flink). # ponytail: feature count, add when >30 features.

### [agent] resolved
> 3 on-chain + 2 derived = 5 baseline features. Extensible via Feast feature service.

## 2. Dual-Write Architecture: Kafka → Feature Store + Flink

Kafka как источник истины. Два потока: (1) Feast ingestion для batch features, (2) Flink для real-time λ(t).

```
Kafka → Feast Ingestion → Offline Store (PostgreSQL) → Training
Kafka → Feast Ingestion → Online Store (Redis) → Serving
Kafka → Flink Stateful → Hawkes λ(t) → Feature Service
```

### [human] Dual-write consistency
> Как гарантировать что Feast и Flink видят одни и те же данные?

**Decision:** Kafka offsets как source of truth. Feast ingestion и Flink потребляют из одного topic partition. Dedup по tx_hash. # ponytail: ordering guarantee, add when partition lag > 10s.

### [agent] resolved
> Один Kafka topic. Offset-based dedup. Feast и Flink читают одинаковую позицию.

## 3. Train-Serve Skew < 1%

Feast гарантирует один code path. Monitoring KS test daily.

### [human] Skew detection
> Как именно измерять skew между training и serving?

**Decision:** KS test для каждого feature. Alert при p<0.01 или drift > 1%. # ponytail: drift threshold, add when feature count > 50.

### [agent] resolved
> KS test daily. Skew = max KS statistic across features. Alert при > 1%.

## 4. Pushback: What Could Go Wrong

### [human] Hypothesis 1: Feast offline store becomes bottleneck
> PostgreSQL offline store тормозит при большом объёме features.

**Decision:** PgBouncer + индексы. Миграция на ClickHouse при >100M vectors. # ponytail: ClickHouse offline, add when offline query latency > 5s.

### [agent] resolved
> PgBouncer + индексы. Мониторинг query latency. При >5s — ClickHouse.

### [human] Hypothesis 2: Dual-write introduces data inconsistency
> Feast и Flink могут видеть разные данные из-за race condition.

**Decision:** Kafka transactional idempotent producer. Feast и Flink используют same transaction ID. At-least-once с dedup. # ponytail: exactly-once, add when inconsistency detected > 2 times.

### [agent] resolved
> Transactional producer. Dedup по tx_hash в обоих потоках.

## 5. Acceptance Criteria

- [ ] Feast Feature Store развёрнут (Redis online, PostgreSQL offline)
- [ ] On-chain features: tx_count, balance_history, cluster_risk_score
- [ ] Derived features: velocity, Hawkes λ(t)
- [ ] Dual-write Kafka → Feast + Flink
- [ ] Train-serve skew < 1% (KS test daily)
- [ ] Docker image pinning для train/serve
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 6. Open Questions

- Сколько Redis shard-ов нужно для online store при 1M+ адресов?
- Нужен ли Feast online store для λ(t) или только batch?

## 7. Pushback Summary

| # | Гипотеза | Решение |
|---|----------|---------|
| 1 | Offline store bottleneck | PgBouncer → ClickHouse at >100M |
| 2 | Dual-write inconsistency | Kafka transactional idempotent |
