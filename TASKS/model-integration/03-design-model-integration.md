# Model Integration — Design

> Phase: D (Design) | Slug: model-integration | Status: Draft
> Предшественники: 3A, 3B, 4.1-4.3, 5A-5C дизайны завершены

## 1. Scope

Интеграция 8 артефактов (3A–5C) в единый `model_core/` Python package. Порядок реализации: A → B → C → D.

### [human] Integration Strategy
> Как объединить 8 независимых дизайнов в один рабочий код?

**Decision:** Единый `model_core/` package с подмодулями. Shared `pyproject.toml`. Каждый подмодуль — отдельный namespace. Чистый PyTorch (без PyG-Temporal). Feast заменён на Redis напрямую. # ponytail: Feast, add when train-serve skew > 1%.

### [agent] resolved
> Один `model_core/` с подмодулями: `causal_attention/`, `hyperbolic/`, `dataset/`, `loss/`, `feature_store/`, `hawkes/`, `inference/`, `explainability/`. Shared `pyproject.toml`. Pure PyTorch temporal. Redis для feature store MVP.

## 2. Architecture

```
model_core/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── dataset/          # 4.1 Elliptic++ v2, temporal split, hard negatives
│   ├── loss/             # 4.2 Focal, Contrastive, Combined, FSDP
│   ├── feature_store/    # 4.3 Redis-based feature store + GE-lite
│   ├── causal_attention/ # 3A NOTEARS DAG + causal attention
│   ├── hyperbolic/       # 3B Lorentz message passing
│   ├── hawkes/           # 5A Power-law temporal encoding
│   ├── inference/        # 5B Lazy inference engine
│   └── explainability/   # 5C Influence functions, HNSW, JSON Schema
├── tests/
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_causal_attention.py
│   ├── test_hyperbolic.py
│   ├── test_hawkes.py
│   ├── test_inference.py
│   └── test_explainability.py
└── config/
    ├── model_config.yaml
    ├── split_config.json
    └── training_config.yaml
```

### [human] Dependency Chain
> Какие подмодули зависят друг от друга?

**Decision:**
- `dataset/` → base для всех (provides `EllipticDataset` class)
- `loss/` → зависит от `dataset/` (label definitions)
- `hawkes/` → независимый (только event stream)
- `causal_attention/` → зависит от `dataset/` (subgraph extraction)
- `hyperbolic/` → зависит от `dataset/` (embeddings)
- `feature_store/` → зависит от `dataset/` (feature definitions)
- `inference/` → зависит от `causal_attention/`, `hyperbolic/`, `hawkes/`
- `explainability/` → зависит от `inference/`, `causal_attention/`

### [agent] resolved
> `dataset/` — базовый модуль. `loss/` и `hawkes/` — независимы. `causal_attention/`, `hyperbolic/`, `feature_store/` — зависят от `dataset/`. `inference/` и `explainability/` — верхний уровень.

## 3. Key Design Decisions

### 3.1 Pure PyTorch Temporal
- PyG-Temporal заменён на чистый PyTorch `nn.Module` с temporal convolution
- `# ponytail:` PyG-Temporal, add when temporal ops exceed pure PyTorch performance by >5%

### 3.2 Redis Feature Store (MVP)
- Feast отложен до Phase 2
- Redis для online features, PostgreSQL для offline
- `# ponytail:` Feast, add when train-serve skew > 1%

### 3.3 Shared `pyproject.toml`
- Единый dependency management для всех подмодулей
- Каждый подмодуль импортируется через `model_core.dataset`, `model_core.loss` и т.д.

### 3.4 Model Training Pipeline
- Unified `Trainer` class с FSDP support
- Config-driven: `training_config.yaml` controls loss weights, FSDP strategy, optimizer
- Sequential: Focal-only 20 epochs → add Contrastive

## 4. Pushback Hypotheses

### [human] H1: Pure PyTorch temporal layers underperform vs PyG-Temporal
> Кастомный temporal convolution может быть медленнее или менее функционален.

**Decision:** Benchmark после реализации. Если gap > 5% — пересмотреть. # ponytail: PyG-Temporal, add when performance gap > 5%.

### [agent] resolved
> Чистый PyTorch baseline. Benchmark на Elliptic++ v2.

### [human] H2: Redis-only feature store не масштабируется для training
> Offline training features из PostgreSQL могут быть медленными.

**Decision:** PostgreSQL offline + Redis online. При >100M vectors — миграция на ClickHouse. # ponytail: ClickHouse offline, add when offline query latency > 5s.

### [agent] resolved
> PostgreSQL для offline, Redis для online. Мониторинг query latency.

### [human] H3: Shared `pyproject.toml` создаёт dependency conflicts
> Разные подмодули могут нуждаться в разных версиях библиотек.

**Decision:** Один `pyproject.toml` с pinned versions. Если конфликт — разбить на два `pyproject.toml` для изолированных подмодулей. # ponytail: split pyproject, add when dependency conflict detected.

### [agent] resolved
> Один pyproject.toml. Pinned versions. Monitor for conflicts.

## 5. Acceptance Criteria

- [ ] `model_core/` package с 8 подмодулями
- [ ] Pure PyTorch temporal (no PyG-Temporal)
- [ ] Redis-based feature store (no Feast)
- [ ] Shared `pyproject.toml`
- [ ] `dataset/` module: Elliptic++ v2 loader, temporal split, hard negatives
- [ ] `loss/` module: Focal (γ=2.0), Contrastive (τ=0.1), Combined, FSDP
- [ ] `hawkes/` module: Power-law kernel, online SGD, seasonal μ(t)
- [ ] `causal_attention/` module: NOTEARS DAG, causal attention
- [ ] `hyperbolic/` module: Lorentz ops, message passing
- [ ] `inference/` module: BFS depth 2, micro-batch 32-64, LRU cache
- [ ] `explainability/` module: influence functions, HNSW Lorentz, JSON Schema, LLaMA 3.1 8B
- [ ] 0 🔴 Critical, 0 ai-slops
- [ ] All `# ponytail:` markers present
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 6. Open Questions

Нет открытых вопросов. Переход к P (Plan) по согласованию.
