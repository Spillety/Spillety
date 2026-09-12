# 10.3 Load & Performance Testing — Design

> Phase: D (Design) | Slug: load-testing | Status: In Progress

## 1. Tool Selection: k6 & Locust

k6 для API/load testing, Locust для distributed load simulation. Покрытие p50/p95/p99 latency, throughput, error rate.

### [human] k6 vs Locust
> k6 или Locust для load testing?

**Decision:** k6 для HTTP/gRPC API тестов (scriptable, CLI-first). Locust для распределенных сценариев (WebSocket, gRPC streaming). # ponytail: unified tool, add when k6 + Locust overhead > 20% maintenance.

### [agent] resolved
> k6 для API, Locust для distributed. Оба в CI pipeline.

### [human] Metrics collection
> Какие метрики собирать?

**Decision:** p50/p95/p99 latency, throughput (tx/s), error rate (%). Для ML inference — GPU utilization, memory usage. # ponytail: GPU metrics, add when GPU OOM occurs during load.

### [agent] resolved
> p50/p95/p99, throughput, error rate. GPU metrics supplementary.

## 2. Load Profiles: Ramp-up, Spike 10x, Sustained 72h

Три профиля нагрузки для различных сценариев.

### [human] Profile definitions
> Какие профили нагрузки тестировать?

**Decision:** 1) Ramp-up: 0 → 100% target over 30min (gradual load test). 2) Spike: instant 10x baseline (flash loan attack simulation). 3) Sustained: 72h at 80% target (steady-state endurance). # ponytail: spike magnitude, add when attack patterns exceed 10x.

### [agent] resolved
> Ramp-up 30min, Spike 10x instant, Sustained 72h at 80%.

## 3. Performance Budgets & SLOs

### [human] Latency budgets per component
> Какие latency budgets задать?

**Decision:** Kafka ingestion p99 < 100ms, Flink processing p99 < 200ms, Memgraph query p99 < 50ms, Inference p99 < 200ms. End-to-end p99 < 500ms. # ponytail: end-to-end budget, add when p99 > 500ms in testing.

### [agent] resolved
> Component budgets: Kafka 100ms, Flink 200ms, Memgraph 50ms, Inference 200ms. E2E < 500ms.

### [human] Error rate thresholds
> При каком error rate считать систему неработоспособной?

**Decision:** Error rate > 1% — SLO violation, alert. > 5% — rollback. 0.1% acceptable during spike. # ponytail: spike tolerance, add when spike error > 5%.

### [agent] resolved
> 0.1% during spike, > 1% alert, > 5% rollback.

## 4. Pushback: What Could Go Wrong

### [human] Hypothesis 1: Sustained 72h testing causes resource exhaustion
> 72h нагрузка может выявить утечки памяти/ресурсов, но дорогая.

**Decision:** Автоматизированный запуск ночью. Мониторинг memory leak detection (RSS growth > 1%/hour). # ponytail: memory leak detection, add when RSS growth > 1%/hour.

### [agent] resolved
> Nightly automated run. RSS growth monitoring. Alert при > 1%/hour.

### [human] Hypothesis 2: k6 + Locust double-infrastructure cost
> Два инструмента удваивают инфраструктурную сложность.

**Decision**: k6 для 80% тестов (API). Locust только для distributed scenarios. Shared CI runner. Общая стоимость < 20% от infra. # ponytail: tool consolidation, add when maintenance > 20% of sprint.

### [agent] resolved
> k6 primary, Locust secondary. Shared CI runner. Cost < 20%.

## 5. Acceptance Criteria

- [ ] k6 + Locust configured for API and distributed testing
- [ ] Three profiles: ramp-up, spike 10x, sustained 72h
- [ ] p50/p95/p99 latency: Kafka 100ms, Flink 200ms, Inference 200ms, E2E 500ms
- [ ] Throughput and error rate tracked
- [ ] Error rate: > 1% alert, > 5% rollback
- [ ] 72h sustained test runs nightly with memory leak detection
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 6. Pushback Summary

| # | Гипотеза | Решение |
|---|----------|---------|
| 1 | 72h testing causes resource exhaustion | Nightly run + RSS growth monitoring |
| 2 | Double tool infrastructure cost | k6 primary (80%), Locust secondary |
