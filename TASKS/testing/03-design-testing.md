# 10.1 Testing — Design

> Phase: D (Design) | Slug: testing | Status: In Progress

## 1. Coverage Requirements & Test Data Strategy

80% coverage standard, 95% ML components. Synthetic data (Faker + graph generators) + anonymized production.

### [human] Coverage split
> 80% vs 95% — как именно разделить?

**Decision:** 80% для standard services (Kafka, Flink, Memgraph). 95% для ML pipeline (training, inference, feature extraction). ML components требуют больше тестов из-за stochasticity. # ponytail: coverage threshold, add when ML logic complexity grows.

### [agent] resolved
> 80% standard, 95% ML. Stochastic tests: deterministic seeds + statistical bounds.

### [human] Synthetic vs production data
> Сколько synthetic данных нужно для покрытия?

**Decision:** Faker + graph generators для unit/integration тестов. Анонимизированный production для e2e и нагрузки. Соотношение: 70% synthetic, 30% anonymized production. # ponytail: production ratio, add when synthetic coverage gaps > 20%.

### [agent] resolved
> 70% synthetic + 30% anonymized production. Faker для данных, graph generators для topology.

## 2. Chaos Engineering: Failure Scenarios

Kafka down, Memgraph crash, GPU OOM — интегрировано в блоки инфраструктуры и ML.

### [human] Chaos scenarios priority
> Какие сценарии тестировать первыми?

**Decision:** 1) Kafka down (message loss/duplication), 2) Memgraph crash (graph consistency), 3) GPU OOM (inference degradation). Каждый тест с recovery verification. # ponytail: additional scenarios, add when new critical service deployed.

### [agent] resolved
> 3 priority scenarios. Recovery verification обязательна для каждого.

## 3. Test Pyramid for AML Pipeline

```
        ┌─────────────┐
        │  E2E Tests   │  10% ← production data
        ├─────────────┤
        │ Integration  │  30% ← synthetic
        ├─────────────┤
        │ Unit Tests   │  60% ← synthetic
        └─────────────┘
```

### [human] E2E test scope
> Что включать в E2E тесты?

**Decision:** Полный pipeline: Kafka → Flink → Memgraph → Feature Store → Model → Alert. Проверка end-to-end latency и correctness. # ponytail: E2E frequency, add when pipeline changes > 2/week.

### [agent] resolved
> Полный pipeline E2E. Запуск при каждом pipeline change. Latency budget < 200ms.

## 4. Pushback: What Could Go Wrong

### [human] Hypothesis 1: 95% ML coverage is unattainable for stochastic models
> Стохастические ML модели сложно покрыть 95% unit тестами.

**Decision:** Достаточные seed-based детерминированные тесты + statistical tolerance bands (assert < 2σ from expected). # ponytail: statistical testing approach, add when CI fails > 3 times/week.

### [agent] resolved
> Seed-based tests + tolerance bands. CI flaky rate < 5%.

### [human] Hypothesis 2: Chaos engineering causes data loss in production-like environments
> Chaos tests могут разрушить dev/staging данные.

**Decision**: Изолированные environments для chaos. Production-like data generation только в staging. Snapshots перед каждым chaos test. # ponytail: snapshot strategy, add when data restore time > 30min.

### [agent] resolved
> Изолированные environments. Snapshots перед chaos. Восстановление < 30min.

## 5. Acceptance Criteria

- [ ] Coverage: 80% standard, 95% ML components
- [ ] Synthetic data: Faker + graph generators (70%) + anonymized production (30%)
- [ ] Chaos engineering: Kafka down, Memgraph crash, GPU OOM
- [ ] E2E tests: full pipeline < 200ms latency
- [ ] CI flaky rate < 5%
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 6. Pushback Summary

| # | Гипотеза | Решение |
|---|----------|---------|
| 1 | 95% ML coverage unattainable | Seed-based + 2σ tolerance bands |
| 2 | Chaos causes data loss | Isolated env + snapshots |
