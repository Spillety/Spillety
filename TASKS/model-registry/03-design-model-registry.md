# 9.2 Model Registry — Design

> Phase: D (Design) | Slug: model-registry | Status: In Progress

## 1. MLflow Selection & SemVer Versioning

MLflow для model registry. SemVer + experiment tracking. Каждая итерация модели — отдельная версия.

### [human] MLflow vs alternatives
> MLflow, DVC, или custom registry?

**Decision:** MLflow. Полный lifecycle: experiment → model → stage. Отдельный от Feature Store, хотя MLflow может объединять оба. # ponytail: DVC fallback, add when Git-based model versioning > 100 experiments.

### [agent] resolved
> MLflow выбран. SemVer для моделей, experiment tracking встроен.

### [human] SemVer scheme
> Какой формат версионирования для ML моделей?

**Decision:** SemVer: MAJOR.BUILD.PATCH. MAJOR — архитектурные изменения. BUILD — dataset/data changes. PATCH — hyperparameter tuning. # ponytail: MAJOR threshold, add when architecture changes > 3/month.

### [agent] resolved
> SemVer MAJOR.BUILD.PATCH. PATCH = hyperparams, BUILD = data, MAJOR = architecture.

## 2. Stage Transitions: Staging → Production → Archived

Manual approval для Production. Аудит всех переходов.

```
Staging → [Manual Approval] → Production → [Performance Degradation] → Archived
                ↓
         Automated Rollback Trigger
```

### [human] Manual approval gate
> Кто утверждает переход в Production?

**Decision:** Модератор AML-команды + ML engineer. Обязательный чек-лист: accuracy, latency, fairness. # ponytail: approval automation, add when team > 10 analysts.

### [agent] resolved
> Двойное одобрение: модератор + ML engineer. Чек-лист в MLflow.

## 3. Shadow Deployment & Automated Rollback

Новые модели параллельно с production. Сравнение на реальном traffic. Rollback при degradation.

### [human] Shadow deployment mechanism
> Как сравнивать новую модель с production без влияния на users?

**Decision:** Shadow mode: новая модель получает те же входы, но её выводы не используются. Сравнение AUC-PR с production. # ponytail: shadow duration, add when shadow period > 2 weeks.

### [agent] resolved
> Shadow mode с shared feature pipeline. Сравнение метрик в реальном времени.

### [human] Automated rollback criteria
> При каких условиях автоматический rollback?

**Decision:** AUC-PR drop > 5% или latency spike > 50%. Rollback к предыдущей версии в < 2 minutes. # ponytail: rollback threshold, add when false rollback rate > 5%.

### [agent] resolved
> AUC-PR decay > 5% → rollback. Latency p99 spike > 50% → rollback. Rollback в <2min.

## 4. Pushback: What Could Go Wrong

### [human] Hypothesis 1: MLflow registry becomes SPOF
> MLflow server может стать единой точкой отказа.

**Decision:** MLflow server в K8s с HPA. Backend — MySQL/PostgreSQL с replica. S3/GCS для model artifacts. # ponytail: HA setup, add when registry downtime > 5min/month.

### [agent] resolved
> K8s с HPA + реплики БД. Артефакты в S3. Мониторинг uptime.

### [human] Hypothesis 2: Shadow deployment adds latency to serving path
> Двойной forward pass (production + shadow) увеличивает latency.

**Decision:** Shadow inference на отдельном GPU worker pool. Не влияет на production path. # ponytail: GPU isolation, add when shadow pool utilization > 80%.

### [agent] resolved
> Отдельный GPU pool для shadow. Производительность не затронута.

## 5. Acceptance Criteria

- [ ] MLflow registry развёрнут с SemVer версионированием
- [ ] Stage transitions: Staging → Production → Archived (audit trail)
- [ ] Shadow deployment: параллельное сравнение с production
- [ ] Automated rollback: AUC-PR decay > 5% или latency spike > 50%
- [ ] Manual approval gate для Production
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 6. Open Questions

- Нужен ли model card generation автоматически?
- Как интегрировать MLflow с Feature Store для feature versioning?

## 7. Pushback Summary

| # | Гипотеза | Решение |
|---|----------|---------|
| 1 | MLflow SPOF | K8s HPA + реплики БД + S3 artifacts |
| 2 | Shadow adds latency | Отдельный GPU pool для shadow |
