# Реализация Spillety

Документ описывает инженерные детали текущей реализации: алгоритмы, инфраструктуру, точки расширения.

## Алгоритмы

### Layer 0 — Загрузка данных

`spillety.data.loader.load_elliptic` читает три CSV Elliptic++: `features` (203 769 × 167), `classes`, `edgelist` (234 355 edges). `temporal_split` делит по `time_step` без шаффла: train ≤ 30, valid 31–40, test 41–49. Это предотвращает утечку будущего.

### Layer 1 — Entity Resolution

`spillety.entity.resolution` строит сигналы для пар адресов:
- **CIOH** (co-spending): общий вход в транзакцию — O(1) через `edge_set`
- **Temporal co-occurrence**: `|t_a − t_b| ≤ 1`
- **Degree similarity**: `1 − |deg_a − deg_b| / max_deg`

Fusion — `LogisticRegression` на трёх сигналах. Приоры калибруются на Elliptic++ ground truth. Кластеризация — connected components через `networkx` (O(N+M)).

### Layer 2 — Embeddings

`spillety.embeddings.contrastive` реализует PCA-proxy для GraphSAGE:
- `StandardScaler + PCA(165 → 32)` + L2-нормализация
- NT-Xent loss с temperature `τ = 0.1`
- Anchor loss с weighted margin: `w = 0.3 + 0.7·Jaccard`
- Оценка: silhouette score, recall@K

Production-upgrade: заменить `encode_pca` на `torch_geometric.nn.GraphSAGE` без смены интерфейса.

### Layer 3 — Causal Filter

`spillety.causal.filter` реализует DAG-proxy:
- `is_spurious(q, a)`: confounded ∧ distance_explained
- Confounded = оба узла имеют hub-соседа (top-5% degree)
- Distance explained = `|d(q,a) − d(c,a)| < ε`, `ε = 5.0`
- E-value: `RR + √(RR·(RR−1))`, Tier 1 threshold `E > 2.0`
- Falsification test: χ² per stratum + partial correlation

### Layer 4 — Retrieval

`spillety.retrieval.hnsw` использует `sklearn.neighbors.NearestNeighbors` (kd_tree/ball_tree/brute) как proxy для HNSW:
- Построение индекса: `build_index(embeddings)`
- Запрос: `query(index, Z, k=10)`
- Recall benchmark: `evaluate_recall(approx, exact)`

Production-upgrade: `hnswlib` с `M=24, ef_construction=128`.

### Layer 5 — GBDT + Calibration

`spillety.models.baseline`:
- `GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)`
- Балансировка через `sample_weight`

`spillety.models.calibration`:
- Isotonic regression vs Platt scaling (sigmoid)
- Выбор по минимальному ECE на hold-out
- `expected_calibration_error(n_bins=10)`

Production-upgrade: `lightgbm.LGBMClassifier`.

### Layer 6 — Evidence

`spillety.evidence.worm`:
- Canonical JSON + SHA256 hash
- HMAC-SHA256 proxy для Ed25519 (64B)
- Merkle tree: `build_merkle(leaves)`, `merkle_proof(levels, idx)`
- Верификация: `verify_evidence`, `verify_proof`

`spillety.evidence.audit`:
- WORM log: цепочка хешей с `prev_hash`
- OpenTimestamps proxy (daily anchoring)

### Layer 7 — Output & Tiers

`spillety.cost.operating`:
- Cost-optimal threshold: `Cost(τ) = C_FP·FP + C_FN·FN`
- Budget constraint: `alerts ≤ B`
- Tier metrics: precision, recall, FTE hours

`spillety.metrics.dashboard`:
- PR-AUC, ROC-AUC, Brier, ECE
- Precision@K, Recall@K
- Latency p50/p99
- Drift KS-test

### Temporal Validation

`spillety.temporal.validation`:
- Walk-forward splits: expanding train < test
- Drift detection: KS-test + Cohen d per feature
- Silhouette score на старых vs новых кластерах
- Классификация: норма / дрейф / новый паттерн / шум

## Инфраструктура

### Docker

`infra/Dockerfile`: Python 3.11 slim, editable install `spillety`, запуск через `uvicorn spillety.serve:app`.

`infra/docker-compose.yml`: spillety (8000) + redis (6379) + postgres (5432).

### Kubernetes

`infra/k8s/`:
- `deployment.yaml`: 2 реплики spillety + redis + postgres
- `service.yaml`: ClusterIP для всех сервисов
- `configmap.yaml`: переменные окружения
- `hpa.yaml`: autoscaling 2–10 реплик по CPU/memory
- `pvc.yaml`: 10 Gi для postgres

### Ansible

`infra/ansible/setup.yml`:
- Установка системных зависимостей
- Клонирование репозитория
- Создание venv, установка пакета
- Systemd unit для FastAPI

### CI/CD

`.github/workflows/ci.yml`:
- Lint: ruff, black, mypy
- Test: pytest
- Smoke: pipeline на 1000 txs
- Build Docker + health check

### FastAPI Serve

`spillety/serve.py`:
- `/api/health` — healthcheck
- `/api/metrics` — запуск pipeline, возврат метрик
- `/api/plot/{tab}` — seaborn графики (PR, reliability, embeddings)
- `/api/evidence` — evidence JSON
- `/api/worm` — WORM audit chain

## Демостенд

`demo/` — React + Vite приложение:
- Панель параметров: C_FP, C_FN, n_components, k, tau
- Кнопка «Запустить pipeline»
- Вкладки:
  - **Graph** — react-flow визуализация транзакций
  - **Embeddings** — scatter plot PCA (licit/illicit)
  - **PR-curve** — precision/recall
  - **Reliability** — calibration diagram
  - **Evidence** — JSON-карточки
  - **WORM** — Merkle root и proofs
- `demo/scripts/bootstrap.sh` — автоматический деплой

## CLI

`python -m spillety.cli smoke` — smoke test на 1000 txs.
`python -m spillety.cli fit --data data/elliptic_raw` — обучение на полных данных.

## Тесты

`tests/test_features.py` — graph и temporal features.
`tests/test_models.py` — baseline и calibration.
`tests/test_pipeline.py` — end-to-end pipeline + evidence verify.

## Зависимости

Базовые: numpy, pandas, scikit-learn, scipy, networkx.
Опциональные: lightgbm, hnswlib, torch, torch-geometric.

## Точки расширения

| Компонент | Текущее | Upgrade |
|-----------|---------|---------|
| Encoder | PCA 32 | GraphSAGE 128 |
| ANN | sklearn NN | hnswlib |
| GBDT | GradientBoosting | LightGBM |
| Calibration | Isotonic/Platt | Beta calibration |
| Causal | Hub proxy | PC-algorithm/NOTEARS |
| Signature | HMAC-SHA256 | Ed25519 |
