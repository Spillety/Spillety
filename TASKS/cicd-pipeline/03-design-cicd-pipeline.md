# 8.3 CI/CD Pipeline — Design

## Context
Kubeflow Pipelines для ML, Airflow для non-ML, MLflow model registry, shadow deployment, automated rollback при accuracy drop > 5% или latency spike > 50%.

### [human] Design Decisions
- **Kubeflow Pipelines**: ML training pipelines — не кастомные скрипты
- **Airflow**: Orchestration non-ML (data ingest, feature engineering, reporting)
- **MLflow**: Model registry — версионирование, stage transitions (Staging → Production)
- **Shadow deployment**: Новая модель параллельно с production, сравнение на реальном traffic
- **Automated rollback**: Accuracy drop > 5% или latency spike > 50% → automatic rollback

### [agent] resolved
- Kubeflow: Pipeline definitions as Python DSL; Argo Workflows backend
- Airflow: DAGs for data ingestion, entity resolution batch, reporting; CeleryExecutor
- MLflow: Tracking server + registry; model versions tagged with metrics; `mlflow.transition_model_version_stage`
- Shadow: Copy production traffic to shadow model; compare predictions; no side effects
- Rollback: Monitor `accuracy_score` and `p99_latency` via Prometheus; automated rollback script triggers on breach

## Architecture
```
Code Commit → CI → Kubeflow/Airflow → MLflow Registry
  ├── ML Pipeline: Kubeflow Pipelines → Train → Validate → Register
  ├── Non-ML Pipeline: Airflow → Data Ingest → Feature Engineering
  ├── Deployment: Shadow mode → compare → promote/rollback
  └── Rollback: accuracy drop > 5% OR latency spike > 50% → auto-rollback
```

**Stages**: Dev → Staging (shadow) → Production (with rollback guardrails)

## Pushback Hypotheses
1. **H1**: Kubeflow Pipelines имеет высокий порог входа и overhead. *Mitigation*: Start with simple pipelines; use KFP SDK v2 for simplified DSL. `# ponytail:` Kubeflow complexity
2. **H2**: Shadow deployment требует удвоенного inference capacity. *Mitigation*: Shadow model uses same GPU pool; traffic mirror at 10% initially. `# ponytail:` shadow capacity cost

## Open Questions
- Как часто запускать Kubeflow pipelines (nightly или event-driven)?
- Какой latency spike threshold срабатывает rollback (50% of baseline)?

## [human] Acceptance
- ML pipeline runs in < 4h from commit to model registry
- Shadow deployment compares predictions on 100% production traffic
- Automated rollback triggers within 5min of metric breach
- Zero manual intervention for rollback
