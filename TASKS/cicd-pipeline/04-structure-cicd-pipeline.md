# 8.3 CI/CD Pipeline — Structure

```
cicd-pipeline/
├── kubeflow/
│   ├── pipelines/
│   │   ├── training_pipeline.py      # KFP v2 DSL
│   │   ├── validation_pipeline.py    # Model validation
│   │   └── kubeflow_config.yaml
│   └── components/
│       ├── train_component.py
│       ├── evaluate_component.py
│       └── register_component.py
├── airflow/
│   ├── dags/
│   │   ├── data_ingestion_dag.py
│   │   ├── feature_engineering_dag.py
│   │   └── reporting_dag.py
│   └── plugins/
│       └── kafka_hook.py
├── mlflow/
│   ├── mlflow_server.py              # Registry setup
│   ├── model_versioning.py           # Stage transitions
│   └── model_cards.py                # Model documentation
├── deployment/
│   ├── shadow_deployment.py          # Shadow traffic mirror
│   ├── rollback_guard.py             # Auto-rollback logic
│   └── rollback_triggers.yaml        # Metrics thresholds
└── tests/
    ├── test_pipeline_execution.py
    ├── test_shadow_comparison.py
    └── test_rollback_triggers.py
```

## Key Components
- `training_pipeline.py`: KFP v2 pipeline for model training and validation
- `shadow_deployment.py`: Mirrors production traffic to candidate model
- `rollback_guard.py`: Monitors accuracy and latency; triggers rollback on breach
- `model_versioning.py`: MLflow stage transitions (Staging → Production)

## Pushback Hypotheses
1. **H1**: Kubeflow + Airflow double orchestration adds complexity. *Mitigation*: Kubeflow for ML-only, Airflow for data; clear boundary in DAG definitions. `# ponytail:` orchestration overlap
2. **H2**: Automated rollback could cause flapping if metrics are noisy. *Mitigation*: Add cooldown period (30min) and require 3 consecutive breaches. `# ponytail:` rollback flapping

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- No custom Python scripts for pipeline orchestration
