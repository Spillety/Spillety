# 04-structure-model-registry

> Phase: S (Structure) | Slug: model-registry | Status: In Progress

## 1. Project Structure

```
model_registry/
├── config/{mlflow_server.yaml,stages.yaml,rollback_policy.yaml}
├── src/{registry/{mlflow_client,stage_manager,shadow_deployer},validation/baseline_eval,rollback/autorollback}.py
├── tests/{test_stage_transitions,test_shadow_deploy,test_rollback}.py
└── scripts/promote_model.py,scripts/rollback_model.py
```

## 2. MLflow Client & Stage Manager

```python
# src/registry/mlflow_client.py
class MLflowClient:
    """MLflow wrapper with SemVer versioning and stage transitions."""
    def log_model(self, model, version: str, stage: str, metrics: dict) -> str:
        run = self.start_run()
        mlflow.log_param("semver", version)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")
        self.transition_stage(run_id, stage)
        return run_id
    def transition_stage(self, run_id: str, stage: str) -> None:
        """Staging → Production → Archived. Manual approval for Production."""
        if stage == "Production" and not self._has_approval(run_id):
            raise ApprovalRequiredError(f"Manual approval required for {run_id}")
        mlflow.set_registered_model_version_stage(run_id, stage)
```

```python
# src/registry/stage_manager.py
class StageManager:
    """Manages stage transitions with audit trail."""
    def __init__(self):
        self.approval_chain = ["aml_moderator", "ml_engineer"]
    def request_production(self, run_id: str) -> bool:
        approvals = [self._check_approval(a, run_id) for a in self.approval_chain]
        return all(approvals)
```

### [human] Approval chain
> Кто именно должен одобрять переход в Production?

**Decision:** AML модератор + ML engineer. Оба подтверждения обязательны. # ponytail: approval automation, add when team > 10.

### [agent] resolved
> Двойное одобрение. Check-list в MLflow.

## 3. Shadow Deployer & Auto-Rollback

```python
# src/registry/shadow_deployer.py
class ShadowDeployer:
    """Shadow deployment: new model receives same inputs, outputs compared."""
    def __init__(self, production_model_uri: str):
        self.production = mlflow.pyfunc.load_model(production_model_uri)
        self.shadow = None
    def deploy_shadow(self, candidate_uri: str) -> None:
        self.shadow = mlflow.pyfunc.load_model(candidate_uri)
    def compare(self, X: pd.DataFrame) -> dict:
        prod_pred = self.production.predict(X)
        shadow_pred = self.shadow.predict(X)
        return {"auc_pr_delta": compute_auc_pr_diff(prod_pred, shadow_pred)}
```

```python
# src/registry/autorollback.py
class AutoRollback:
    """Automatic rollback on performance degradation."""
    THRESHOLDS = {"auc_pr_decay": 0.05, "latency_spike_p99": 50}
    def check_and_rollback(self, current_metrics: dict) -> bool:
        if current_metrics["auc_pr_decay"] > self.THRESHOLDS["auc_pr_decay"]:
            return self._rollback("auc_pr_decay")
        if current_metrics["latency_spike_p99"] > self.THRESHOLDS["latency_spike_p99"]:
            return self._rollback("latency_spike")
        return False
```

### [human] Rollback timing
> Как быстро должен сработать rollback?

**Decision:** < 2 minutes from detection to rollback. Health check interval: 30 seconds. # ponytail: rollback speed, add when false rollback > 5%.

### [agent] resolved
> Health check каждые 30s. Rollback в <2min от детекции.

## 4. Key Components

- `mlflow_client.py`: SemVer versioning + stage transitions. English comments.
- `stage_manager.py`: Manual approval gate + audit trail. `# ponytail:` approval automation
- `shadow_deployer.py`: Parallel inference on shared feature pipeline. English comments.
- `autorollback.py`: AUC-PR decay > 5% or latency spike > 50%. English comments.
- Tests: 80% coverage for registry logic.

## 5. Pushback Hypotheses

1. **H1**: MLflow registry becomes SPOF. *Mitigation*: K8s HPA + реплики БД + S3 artifacts. `# ponytail:` HA setup
2. **H2**: Shadow deployment adds latency. *Mitigation*: Отдельный GPU pool для shadow inference.

## 6. Self-Review

- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- SemVer + stage transitions documented
- Pushback hypotheses: 2 ✅
