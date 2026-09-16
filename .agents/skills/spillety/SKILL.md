---
name: spillety
description: Навигация по проекту Spillety KYT для AI-агента — приоритетный порядок чтения репозитория, актуальные пути и автообновляемые ссылки
---

# Spillety — Agent Skill

Этот файл — краткий гид для агента (LLM), читающего проект Spillety с GitHub. Соблюдай порядок чтения и учитывай актуальность разделов.

## Порядок чтения 1→5 (Primary — читай строго по порядку)

1. `README.md` — обзор проекта, таблица подходов, навигация.
2. `temp.md` — эталонная архитектура v2.0 (финальная, заменяет все предыдущие итерации).
3. `docs/notebooks/_elliptic_loader.py` — единый загрузчик Elliptic-датасета (прокси к `spillety/data/loader.py`).
4. `spillety/pipeline/pipeline.py` — end-to-end пайплайн Layers 0→7 (fit/predict/evaluate, smoke на 1000 txs).
5. `docs/architecture.md` — детальная архитектура (если отсутствует — считай источником `temp.md` + `spillety/pipeline/pipeline.py`).

> Прочитав 1→5, агент понимает систему координат риска, data-flow и точки входа.

## Primary

Актуальные, обязательные к чтению источники (не устаревают):

- `README.md`
- `temp.md`
- `docs/notebooks/_elliptic_loader.py`
- `spillety/pipeline/pipeline.py`
- `docs/architecture.md`

## Secondary (может устаревать)

Теоретические материалы — полезны для контекста, но вторичны и могут отставать от кода:

- `docs/theory/*.md` — 12 глав (01_context_and_motivation.md … 12_metrics_and_quality.md)
- `docs/theory/files/*` — иллюстрации к теории

> Приоритет: код и Primary > Secondary. При противоречии верь коду.

## Infra

Инфраструктура деплоя/окружения:

- `infra/**/*` — Docker, CI, конфиги окружения (если директория отсутствует — infra ещё не выделена, см. `Makefile`, `scripts/`)

## Demo

Демонстрация и презентация:

- `demo/**/*` — демо-приложение (если отсутствует — см. `docs/presentation.md` / `docs/presentation.html`)
- `docs/presentation.md` и `docs/presentation.html` — fallback для Demo

## Notebooks (читай после Primary)

Нумерованные ноутбуки `docs/notebooks/01..13` — иди строго по порядку:

- `01_baseline_tabular.ipynb` … `13_end_to_end_pipeline.ipynb`
- `docs/notebooks/_elliptic_loader.py` — уже прочитан в Primary, используется всеми ноутбуками
- `docs/notebooks/requirements-notebooks.txt` и `docs/notebooks/_theme.py` — окружение

## Links

<!-- AUTO-GEN:START -->
- `README.md`
- `docs/architecture.md`
- `docs/theory/01_context_and_motivation.md`
- `docs/theory/02_blockchain_on-chain_basics.md`
- `docs/theory/03_graph_theory_and_embeddings.md`
- `docs/theory/04_contrastive_and_metric_learning.md`
- `docs/theory/05_entity_resolution_and_baseline_fusion.md`
- `docs/theory/06_casual_inferense_and_dag.md`
- `docs/theory/07_gbdt_calibration_decision_path.md`
- `docs/theory/08_const_function_and_operating_point.md`
- `docs/theory/09_temoral_validating_and_self-evolution.md`
- `docs/theory/10_evidence_generation_provenance_and_worm_audit.md`
- `docs/theory/11_regulatory_compliance.md`
- `docs/theory/12_metrics_and_quality.md`
- `pyproject.toml`
- `scripts/download_elliptic.py`
- `scripts/gen_skill.py`
- `spillety/__init__.py`
- `spillety/causal/__init__.py`
- `spillety/causal/filter.py`
- `spillety/cli.py`
- `spillety/cost/__init__.py`
- `spillety/cost/operating.py`
- `spillety/data/__init__.py`
- `spillety/data/loader.py`
- `spillety/embeddings/__init__.py`
- `spillety/embeddings/contrastive.py`
- `spillety/entity/__init__.py`
- `spillety/entity/resolution.py`
- `spillety/evidence/__init__.py`
- `spillety/evidence/audit.py`
- `spillety/evidence/worm.py`
- `spillety/features/__init__.py`
- `spillety/features/graph.py`
- `spillety/features/news.py`
- `spillety/features/temporal.py`
- `spillety/metrics/__init__.py`
- `spillety/metrics/dashboard.py`
- `spillety/models/__init__.py`
- `spillety/models/baseline.py`
- `spillety/models/calibration.py`
- `spillety/pipeline/__init__.py`
- `spillety/pipeline/pipeline.py`
- `spillety/retrieval/__init__.py`
- `spillety/retrieval/hnsw.py`
- `spillety/serve.py`
- `spillety/temporal/__init__.py`
- `spillety/temporal/validation.py`
- `temp.md`
- `tests/test_features.py`
- `tests/test_models.py`
- `tests/test_pipeline.py`
<!-- AUTO-GEN:END -->
