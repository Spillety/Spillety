## 1. Data Ingestion & Stream Processing

**[x] 1.1. Kafka Topics Design & Schema Registry** ✅ Done — Design ✅ `TASKS/kafka-topics-design/03-design-kafka-topics.md`, Impl ✅ `TASKS/kafka-topics-design/04-structure-kafka-topics.md`
- **Сериализация:** Protobuf. Avro даёт эволюцию схем, но для on-chain событий с фиксированной структурой (tx_hash, from, to, amount, timestamp) Protobuf быстрее и проще в codegen. Schema Registry всё равно нужен — для backward compatibility при добавлении новых полей.
- **Partition key:** `from_address`. Ordering по `tx_hash` бессмысленен (транзакции независимы). Ordering по `from_address` критичен для incremental PageRank и Hawkes: события одного кошелька должны обрабатываться последовательно.
- **Retention:** 90 дней для сырых событий. Causal discovery работает на скользящем окне 30-60 дней; всё старше — в Iceberg (cold storage).
- **DLQ:** Да. Отдельный топик `dead-letter-events` с оригинальным payload + error metadata. Алерт в Prometheus при росте DLQ > 0.1% throughput.

**[x] 1.2. Flink Stateful Stream Processing** ✅ Done — Design ✅ `TASKS/flink-stateful-processing/03-design-flink-stateful.md`, Impl ✅ `TASKS/flink-stateful-processing/04-structure-flink-stateful.md`
> ✅ Kernel: Power-law. TTL: 14 дней. Численное интегрирование λ(t).
- **State backend:** RocksDB. HashMapStateBackend не выдержит объём keyed state для PageRank и Hawkes при миллионах адресов.
- **State TTL:** **Критично.** Без TTL state растёт бесконечно, вызывая backpressure из-за disk I/O. Power-law kernel: TTL = 14 дней (длинный хвост). Для PageRank: TTL = 30 дней.
- **Incremental PageRank:** Кастомная реализация на KeyedProcessFunction с Monte Carlo approximation. Gelly deprecated. Подход: хранить random walk segments per node, при добавлении ребра обновлять только затронутые сегменты. Ожидаемый speedup ≥ 10x.
- **Exactly-once для Hawkes:** Через Flink checkpointing + idempotent update (λ(t) пересчитывается из сохранённого event log, а не инкрементально). Либо: two-phase commit sink в Memgraph.

**[x] 1.3. Entity Resolution Module** ✅ Done — Design ✅ `TASKS/entity-resolution/03-design-entity-resolution.md`, Impl ✅ `TASKS/entity-resolution/04-structure-entity-resolution.md`
> ✅ Power-law kernel, TTL: 14 дней
- **Алгоритм:** Гибрид: (1) connected components по co-spending heuristic (Bitcoin-style), (2) ML-based link prediction для EVM (graph embedding + clustering). Только heuristic даёт over-merging для exchange withdrawal addresses.
- **Частота:** Incremental для connected components (при каждом новом co-spend), batch nightly для ML-based refinement.
- **Threshold:** Для ML-based: precision > 0.95 при recall > 0.80. Калибровка на labeled internal data. Over-merging детектируется через anomaly detection (слишком большой кластер = подозрительно).
- **Human-in-the-loop:** Только для кластеров с confidence < 0.7. Queue для аналитика.

**DoD:** 80% coverage unit tests, integration test с Kafka + Flink, entity resolution precision/recall валидация.

---

## 2. Temporal Graph Store (Memgraph)

**[x] 2.1. Memgraph Schema & Indexing** ✅ Done — Design ✅ `TASKS/memgraph-schema/03-design-memgraph-schema.md`, Impl ✅ `TASKS/memgraph-schema/04-structure-memgraph-schema.md`
- **Memgraph vs Neo4j:** Memgraph. Neo4j GDS работает batch, не streaming. Memgraph — in-memory, C++ ядро, latency < 1ms для traversal. Для PoC — Memgraph. Для production с > 1B edges — рассмотреть TigerGraph.
- **Temporal edges:** Как property (`valid_from`, `valid_to`). Отдельный тип ребра создаёт explosion в schema.
- **Indexes:** Label index + property index на `address` (hash), full-text index не нужен (адреса — hex strings). Vector index (HNSW) для hyperbolic embeddings — **обязателен** для nearest scam cluster query.
- **Partitioning:** Memgraph не шардируется нативно. Для MVP — single node с 256GB RAM. Для production — TigerGraph или NebulaGraph.

**[x] 2.2. Hyperbolic Embeddings Storage** ✅ Done — `TASKS/hyperbolic-embeddings/03-design-hyperbolic-embeddings.md`, `TASKS/hyperbolic-embeddings/04-structure-hyperbolic-embeddings.md`
- **Формат:** float32 vector (128D) как node property. 128D — sweet spot: 64D теряет иерархию, 256D не даёт прироста для графа такого размера.
- **Обновление:** Incremental при каждом event (lazy inference). Полный пересчёт — nightly.
- **HNSW:** Да, для nearest scam cluster. Без него counterfactual generation занимает O(n) вместо O(log n).

**[x] 2.3. Iceberg Audit Trail** ✅ Done — `TASKS/iceberg-audit-trail/03-design-iceberg-audit-trail.md`, `TASKS/iceberg-audit-trail/04-structure-iceberg-audit-trail.md`
- **Iceberg на S3 + Trino.** Delta Lake только если Spark ecosystem. Iceberg — открытый формат, проще для регулятора.
- **Schema evolution:** Backward-compatible only. Новые поля — optional. Breaking changes — через new table version.
- **Partitioning:** `date` (daily). Alert_id hash — не нужен (queries по времени).
- **Retention:** 7 лет (compliance). Hot data (90 дней) в ClickHouse, cold — в Iceberg.

**DoD:** HNSW benchmark (recall@k ≥ 0.95), Iceberg schema evolution test, query latency < 1ms для traversal.

---

## 3. Model Architecture

**[x] 3A. Causal Attention Implementation** ✅ Done — `TASKS/causal-attention/03-design-causal-attention.md`, `TASKS/causal-attention/04-structure-causal-attention.md`
- **Causal discovery:** NOTEARS для initial DAG (differentiable, O(n³) но n = local subgraph size ~100, не глобальный граф). Для production — GES с domain knowledge prior.
- **Initial DAG:** Domain knowledge (expert graph) для типов рёбер (TRANSACTS → risk, MENTIONED_IN → risk), data-driven для весов. Гибрид.
- **Covariates для backdoor:** Address features (age, tx_count, unique_counterparties), transaction amount, time-of-day, chain. Confounders: exchange internal transfers (помечать как `is_exchange_internal` если source и target в одном exchange cluster).
- **Library:** PyTorch Geometric Temporal (PyG-T) для temporal layers, кастомная реализация causal attention на PyTorch.

**[x] 3B. Hyperbolic Message Passing** ✅ Done — `TASKS/hyperbolic-message-passing/03-design-hyperbolic-message-passing.md`, `TASKS/hyperbolic-message-passing/04-structure-hyperbolic-message-passing.md`
- **Architecture:** Lorentz model (численно стабильнее Poincaré ball) для message passing. Poincaré только для visualization.
- **Curvature:** Learnable per-layer, initialized c = -1.0. Adaptive per-node не нужен — иерархия глобальна.
- **Numerical stability:** Clamp tangent vectors to norm ≤ 1e-5 before exponential map. Gradient clipping (max_norm = 1.0).
- **Library:** Кастомный PyTorch (geomstats слишком медленный для training).

**DoD:** NOTEARS DAG reconstruction accuracy ≥ 0.85, Lorentz model training converges < 24h, numerical stability tests pass.

---

## 4. Model Training Pipeline

**[x] 4.1. Dataset & Labels** ✅ Done — `TASKS/dataset-labels/03-design-dataset-labels.md`, `TASKS/dataset-labels/04-structure-dataset-labels.md`
- **Dataset:** Elliptic++ (v2) + internal labeled data. Elliptic v1 устарел (2014-2015).
- **Labels:** Multi-class (scam, ransomware, terrorist_financing, sanctions, mixer, legitimate). Binary теряет granularity для explainability.
- **Temporal split:** Train: 2014-2015, Val: 2016 H1, Test: 2016 H2. Случайный split = data leakage.
- **Negative sampling:** Hard negatives (near-miss: кошельки с похожими фичами, но legitimate). Random sampling даёт слишком лёгкую задачу.

**[x] 4.2. Loss & Optimization** ✅ Done — `TASKS/loss-optimization/03-design-loss-optimization.md`, `TASKS/loss-optimization/04-structure-loss-optimization.md`
- **Loss:** Focal loss (γ=2.0) для imbalanced classes + contrastive loss (τ=0.1) для hyperbolic embeddings.
- **Distributed:** FSDP для графа > 10M nodes. DDP не влезет в память.

**[x] 4.3. Feature Store & Data Quality** ✅ Done — `TASKS/feature-store-quality/03-design-feature-store-quality.md`, `TASKS/feature-store-quality/04-structure-feature-store-quality.md`
- **Feature Store:** Feast или Tecton — для feature consistency между train и serve. Даже если не используется в inference, нужен для training.
- **Data Quality Monitoring:** Great Expectations для on-chain events. Bad data → bad model. Schema validation + anomaly detection на raw events.

**DoD:** Train/val/test temporal split reproducible, FSDP training стабилен, feature store train-serve skew < 1%.

---

## 5. Explainability & Inference

**[x] 5A. Hawkes Temporal Encoding** ✅ Done — `TASKS/hawkes-temporal/03-design-hawkes-temporal.md`, `TASKS/hawkes-temporal/04-structure-hawkes-temporal.md`
- **Kernel:** Exponential. Power-law лучше для long-range, но экспоненциальный имеет closed-form λ(t) → latency < 1ms.
- **Online estimation:** Online SGD с regret bound O(log T). Не EM — EM batch.
- **Baseline μ(t):** Time-varying (hourly/daily seasonality). Константа игнорирует паттерны (атаки в нерабочие часы).
- **Negative reinforcement:** Не нужен для AML (self-exciting достаточно).
- **Latency:** Closed-form exponential kernel даёт O(1) update.

**[x] 5B. Lazy Inference Engine** ✅ Done — `TASKS/lazy-inference/03-design-lazy-inference.md`, `TASKS/lazy-inference/04-structure-lazy-inference.md`
- **Subgraph extraction:** BFS depth 2 от `from_address` И `to_address`, union. Depth 2 — оптимально: depth 1 теряет контекст, depth 3 взрывает сложность.
- **Batch inference:** Micro-batch (32-64 events) для GPU utilization, но с latency budget < 100ms. Flink buffer → GPU worker.
- **Caching:** LRU cache для hot addresses (exchanges, mixers) — embeddings не меняются часто.
- **Latency budget:** 20ms subgraph extraction, 60ms forward pass, 20ms explanation. GPU: A10G или T4.

**[x] 5C. Structured Explainability Module** ✅ Done — `TASKS/explainability-module/03-design-explainability-module.md`, `TASKS/explainability-module/04-structure-explainability-module.md`
- **Counterfactual:** Approximate через **influence functions** (Koh & Liang) вместо полного recompute. Speedup 100x.
- **Causal path:** Top-3 edges по backdoor-adjusted causal effect. Больше — шум.
- **Hyperbolic distance:** HNSW search в Lorentz space. Brute force O(n) неприемлем.
- **Regulatory refs:** Hardcoded mapping jurisdiction → regulation. Динамическая загрузка — overkill для MVP.
- **JSON schema:** JSON Schema draft 2020-12 + custom validator для AML-specific fields.
- **LLM Explainer:** Open-source LLaMA 3.1 8B (local, no API). Temperature=0.

**DoD:** p99 inference latency < 200ms, counterfactual explanation fidelity ≥ 0.9, HNSW recall@k ≥ 0.95.

---

## 6. OLAP & Analytics (ClickHouse)

**[x] 6.1. ClickHouse Schema** ✅ Done — `TASKS/clickhouse-schema/03-design-clickhouse-schema.md`, `TASKS/clickhouse-schema/04-structure-clickhouse-schema.md`
- **Table:** ReplacingMergeTree для deduplication raw events + AggregatingMergeTree для velocity metrics.
- **Partitioning:** `toYYYYMM(date)`. Daily partition создаёт too many parts.
- **Materialized views:** Pre-aggregate velocity (tx per hour, unique counterparties per day). On-the-fly слишком медленно для real-time alerts.
- **Retention:** 90 дней hot в ClickHouse, старше — S3 через `s3()` table function.
- **Integration с Memgraph:** Dual-write из Kafka. CDC из Memgraph не поддерживается нативно.

**[x] 6.2. Historical Aggregations & Alerting** ✅ Done — `TASKS/historical-aggregations/03-design-historical-aggregations.md`, `TASKS/historical-aggregations/04-structure-historical-aggregations.md`
- **Pattern matching:** SQL-based для known patterns (structuring: N txs below threshold within window), ML-based (anomaly detection на агрегациях) для unknown.
- **Window functions:** ClickHouse window functions с `ROWS BETWEEN`. Frame clause обязателен.
- **Alerting:** Materialized View → Kafka для real-time. Periodic batch queries для reports.

**DoD:** Query latency p95 < 500ms для агрегаций, materialized view freshness < 1min.

---

## 7. Alerting & Case Management

**[x] 7.1. Alert Generation Engine** ✅ Done — `TASKS/alert-generation/03-design-alert-generation.md`, `TASKS/alert-generation/04-structure-alert-generation.md`
- **Threshold:** Adaptive (per-entity baseline + deviation > 2σ). Fixed threshold даёт alert fatigue.
- **Alert fatigue:** Correlation across alerts (same entity, same pattern → merge). Rate limiting per analyst.
- **Priority:** P0 (BLOCK) score > 0.95, P1 (REVIEW) 0.7-0.95, P2 (MONITOR) 0.5-0.7.
- **Escalation:** Auto-escalate P0. P1 — human review mandatory.

**[x] 7.2. Case Management** ✅ Done — `TASKS/case-management/03-design-case-management.md`, `TASKS/case-management/04-structure-case-management.md`
- **Build vs buy:** Buy (ServiceNow/Temporal) для MVP. Custom frontend — Phase 2. AML case management — не core competency.
- **Workflow:** Temporal для orchestration (retry, timeout, compensation). State machine слишком хрупкий.
- **Integration:** Auto-populate case с alert JSON, causal path, counterfactual.

**[x] 7.3. SAR/STR Generation** ✅ Done — `TASKS/sar-str-generation/03-design-sar-str-generation.md`, `TASKS/sar-str-generation/04-structure-sar-str-generation.md`
- **Template-based:** Jinja2 templates per jurisdiction. LLM-assisted для narrative section, но human review mandatory.
- **Validation:** Automated validation против FinCEN XML schema. Pre-filing check.
- **Filing:** Manual export для MVP. Regulator API — Phase 2.

**DoD:** Alert false positive rate < 5%, case creation latency < 1s, SAR template coverage ≥ 90% jurisdictions.

---

## 8. Infrastructure & Security

**[x] 8.1. Kubernetes Deployment** ✅ Done — `TASKS/k8s-deployment/03-design-k8s-deployment.md`, `TASKS/k8s-deployment/04-structure-k8s-deployment.md`
- **K8s:** Managed (EKS/GKE). Self-managed — operational overhead.
- **GPU:** T4/A10G для inference. CPU-only для Flink, Kafka, Memgraph.
- **Service mesh:** Нет для MVP. Istio добавляет latency и complexity. Native K8s services + gRPC.
- **Autoscaling:** HPA для inference (queue depth metric). VPA для stateful (Memgraph, ClickHouse).

**[x] 8.2. Monitoring** ✅ Done — `TASKS/monitoring/03-design-monitoring.md`, `TASKS/monitoring/04-structure-monitoring.md`
- **Metrics:** Custom для TCH-GT (inference latency, score distribution, Hawkes λ(t) percentiles). Standard для infra.
- **Alerting:** Alertmanager для SLO (p99 > 100ms, error rate > 1%).
- **Tracing:** OpenTelemetry. Jaeger backend.
- **Dashboards:** Custom Grafana. Community dashboards не покрывают AML-specific metrics.

**[x] 8.3. CI/CD** ✅ Done — `TASKS/cicd-pipeline/03-design-cicd-pipeline.md`, `TASKS/cicd-pipeline/04-structure-cicd-pipeline.md`
- **ML pipeline:** Kubeflow Pipelines. Airflow для non-ML. Кастомные Python scripts — anti-pattern.
- **Model registry:** MLflow. Feast не нужен (нет feature store).
- **Deployment:** Shadow mode для новых моделей (сравнение с production на реальном traffic).
- **Rollback:** Automated при degradation (accuracy drop > 5% или latency spike > 50%).

**[x] 8.4. Data Encryption & Key Management** ✅ Done — `TASKS/data-encryption/03-design-data-encryption.md`, `TASKS/data-encryption/04-structure-data-encryption.md`
- **Key management:** HashiCorp Vault. AWS KMS — lock-in.
- **Data masking:** PII masking в logs. Full encryption в storage.
- **Access control:** RBAC для MVP. ABAC — Phase 2.
- **Audit logging:** Immutable в Iceberg.

**[x] 8.5. Regulatory Compliance** ✅ Done — `TASKS/regulatory-compliance/03-design-regulatory-compliance.md`, `TASKS/regulatory-compliance/04-structure-regulatory-compliance.md`
- **Travel Rule:** IVMS101 message format. Кастомный — не совместим с другими VASPs.
- **Sanctions:** Real-time screening (OFAC, EU, UN) + integration в TCH-GT (sanctions flag as node feature).
- **Jurisdiction rules:** Config-driven (YAML per jurisdiction).
- **Compliance audit:** Documentation + test cases + model cards.

**DoD:** p99 inference latency < 100ms на инфраструктурном уровне, zero PII in logs, RBAC coverage 100%.

---

## 9. Model Lifecycle

**[x] 9.1. Feature Store** ✅ Done — `TASKS/feature-store/03-design-feature-store.md`, `TASKS/feature-store/04-structure-feature-store.md`
- **Implementation:** Feast или Tecton. Feature consistency между train и serve — критична для модели.
- **Scope:** On-chain features (tx_count, balance_history, cluster_risk_score), derived features (velocity, Hawkes λ(t)).
- **Integration:** Dual-write из Kafka в Feature Store + Flink для real-time features.

**[x] 9.2. Model Registry** ✅ Done — `TASKS/model-registry/03-design-model-registry.md`, `TASKS/model-registry/04-structure-model-registry.md`
- **Tool:** MLflow. Отдельный от Feature Store, хотя MLflow может объединять оба.
- **Versioning:** SemVer + experiment tracking. Каждая итерация модели — отдельная версия.
- **Stage transitions:** Staging → Production → Archived. Manual approval для Production.
- **Shadow deployment:** Новые модели параллельно с production, сравнение на реальном traffic.

**[x] 9.3. Data Quality & Drift Monitoring** ✅ Done — `TASKS/data-quality-drift/03-design-data-quality-drift.md`, `TASKS/data-quality-drift/04-structure-data-quality-drift.md`
- **Tool:** Great Expectations для on-chain event validation.
- **Data drift:** KS test для feature distributions, alert при значительном сдвиге.
- **Concept drift:** Performance degradation monitoring (accuracy, AUC-PR decay > 5%).
- **Backtesting:** Elliptic dataset + internal labeled data. Регулярный пересмотр.

**DoD:** Feature store train-serve skew < 1%, model registry stage transitions auditable, drift detection alert latency < 5min.

---

## 10. Cross-cutting Concerns

**[x] 10.1. Testing (интегрирован в DoD каждого блока)** ✅ Done — `TASKS/testing/03-design-testing.md`, `TASKS/testing/04-structure-testing.md`
- **Coverage:** 80% standard, 95% ML components. Требование включено в DoD каждого блока выше.
- **Test data:** Synthetic (Faker + graph generators) + anonymized production.
- **Chaos engineering:** Для failure scenarios (Kafka down, Memgraph crash, GPU OOM) — интегрировано в блоки инфраструктуры и ML.

**[x] 10.2. Model Validation (интегрировано в блоки 3 и 9)** ✅ Done — `TASKS/model-validation/03-design-model-validation.md`, `TASKS/model-validation/04-structure-model-validation.md`
- **Baselines:** Heuristic rules, logistic regression, GCN, GAT — benchmark в блоке Model Training.
- **Metrics:** AUC-PR (imbalanced), calibration error, fairness — часть DoD блока 4.
- **Statistical:** Bootstrap confidence intervals — блок Model Lifecycle.
- **Human eval:** A/B testing analyst preference — блок Alerting & Case Management.

**[x] 10.3. Load & Performance Testing (интегрировано в блок 8)** ✅ Done — `TASKS/load-testing/03-design-load-testing.md`, `TASKS/load-testing/04-structure-load-testing.md`
- **Profiles:** Ramp-up, spike (10x), sustained (72h) — блок Infrastructure & Security.
- **Metrics:** p50/p95/p99 latency, throughput, error rate — DoD каждого блока.
- **Tools:** k6, Locust — блок Infrastructure.

**DoD:** Каждый блок имеет testing DoD, все тесты проходят перед deployment.

---

## Phase 3: Federated Learning (Research)
> Вынесено в отдельный документ: `K-BRAIN/PHASE3-FEDERATED-RESEARCH.md`

---

## Removed/Consolidated Items

- **Блок 8 (Documentation)** — растворён в DoD каждого блока. Architecture docs, API docs, user docs — часть Definition of Done каждого блока.
- **Phase 3 (Federated)** — вынесен в отдельный research-документ, не входит в текущий план.
- **Блок 9 (Testing) и Блок 10 (Roadmap)** — cross-cutting concerns, интегрированы в блоки выше.

---

## Новые/Добавленные Items

- **Feature Store** — теперь блок 9.1 (был в комментариях как отдельная необходимость).
- **Data Quality Monitoring** — блок 9.3 (Great Expectations).
- **Model Registry** — блок 9.2 (MLflow).

---

## Phase II: Design & Structure — ✅ Complete (6 parallel subagents)

**Артефакты:** 20 задач × 2 файла = 40 дизайн+структурных документов
- Subagent 1: Задачи 2.2, 2.3, 3A, 3B (Data + Model Architecture)
- Subagent 2: Задачи 4.1, 4.2, 4.3 (Training Pipeline)
- Subagent 3: Задачи 5A, 5B, 5C (Explainability & Inference)
- Subagent 4: Задачи 6.1, 6.2, 7.1, 7.2, 7.3 (OLAP + Alerting)
- Subagent 5: Задачи 8.1-8.5 (Infrastructure & Security)
- Subagent 6: Задачи 9.1-9.3, 10.1-10.3 (Model Lifecycle + Cross-cutting)

**Self-review:** Все 20 задач: 0 🔴 Critical, 0 ai-slops
**Subagent-выводы:** 20 файлов в `artifacts/subagents/agent-*.md`

**QRSPI статус:** Q→R→D→S ✅ → **P (Plan) → I (Implement) → следующий шаг: интеграция всех артефактов в единый код**

---

## Phase P: Plan — ✅ Complete

**Master plan:** `K-BRAIN/05-plan-master.md` (494 строки)
- Все 4 задачи консолидированы
- Решения из `[human]`/`[agent]` комментариев зафиксированы
- 6 pushback-гипотез с mitigation
- 14 `# ponytail:` markers прослежены
- Subagent-вывод: `artifacts/subagents/plan-master.md`

**QRSPI статус:** Q→R→D→S→P→I ✅ — ВСЕ ФАЗЫ ЗАВЕРШЕНЫ

## Phase I: Implement — ✅ Complete

Код реализован через 4 параллельных subagent'а.

**Создано 49+ исходных файлов:**
- `kafka/` — Python producer/consumer, Protobuf, Schema Registry, Docker/K8s (12 файлов)
- `flink/` — Java KeyedProcessFunction, Power-law Hawkes, Incremental PageRank, RocksDB (14 файлов)
- `entity_resolution/` — Union-Find, Exchange filter, ML pipeline, Anomaly detection (11 файлов)
- `memgraph/` — Cypher schemas, HNSW index, Python query service, Kafka ingestion (12+ файлов)

**Артефакты:**
- `TASKS/*/06-implementation-*.md` — 4 артефакта реализации
- `artifacts/subagents/impl-phase1-*.md` — 4 subagent-вывода
- Self-review: 0 🔴 Critical, 0 ai-slops во всех задачах

---

## Комментарии к реорганизации

1. **Блок 3 разбит на 3A (Architecture), 3B (Training), 3C (Explainability & Inference)** — перегруженность устранена.
2. **Блоки 6+7 (Infra + Security) слиты в блок 8** — production readiness как единый блок.
3. **Model Lifecycle** — отдельный блок 9 для Feature Store, Model Registry, Data Quality.
4. **Cross-cutting Concerns** — блок 10, Testing/Validation/Load встроены в DoD каждого блока.
5. **Phase 3** — отдельный research-документ.
