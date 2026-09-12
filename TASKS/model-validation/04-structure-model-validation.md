# 04-structure-model-validation

> Phase: S (Structure) | Slug: model-validation | Status: In Progress

## 1. Project Structure

```
model_validation/
├── config/{baselines.yaml,metrics.yaml,ab_test.yaml}
├── src/{validation/{baselines,auc_pr,calibration,fairness},statistics/bootstrap,human_eval/ab_test}.py
├── tests/{test_baselines,test_calibration,test_bootstrap,test_ab_test}.py
└── scripts/run_baseline_benchmark.py,scripts/run_ab_eval.py
```

## 2. Baseline Models

```python
# src/validation/baselines.py
class BaselineSuite:
    """Heuristic, LR, GCN, GAT benchmark suite."""
    def run_all(self, X_train, y_train, X_val, y_val) -> dict:
        results = {}
        results["heuristic"] = self._heuristic(X_val)  # Domain rules
        results["lr"] = LogisticRegression().fit(X_train, y_train).score(X_val, y_val)
        results["gcn"] = GCN(in_channels=n_features, hidden=64).fit(X_train, y_train)
        results["gat"] = GAT(in_channels=n_features, hidden=64).fit(X_train, y_train)
        return results  # {model_name: {"auc_pr": ..., "f1": ...}}
```

### [human] GAT vs GCN fallback
> При каких условиях выбирать GCN вместо GAT?

**Decision:** Если GAT не улучшает AUC-PR над LR на > 2% — fallback на GCN. Dropout p=0.5 + early stopping. # ponytail: GAT threshold, add when GAT consistently beats GCN.

### [agent] resolved
> GAT → GCN if AUC-PR improvement < 2%. Dropout p=0.5.

## 3. Calibration & Fairness

```python
# src/validation/calibration.py
class Calibrator:
    """Platt scaling for calibration error < 0.05."""
    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit Platt scaling on validation set."""
        calibrator = CalibratedClassifierCV(LogisticRegression(), cv='prefit')
        calibrator.fit(scores.reshape(-1, 1), labels)
        return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    def calibration_error(self, scores, labels) -> float:
        """Expected calibration error (ECE)."""
        bins = np.linspace(0, 1, 15)
        return expected_calibration_error(scores, labels, bins)
```

```python
# src/validation/fairness.py
class FairnessChecker:
    """Fairness metrics across demographics for regulatory compliance."""
    def check(self, scores, labels, demographics: dict) -> dict:
        """Demographic parity, equalized odds across groups."""
        return {group: {"demographic_parity": ..., "equalized_odds": ...} for group in demographics}
```

### [human] Fairness metrics selection
> Какие fairness metrics считать?

**Decision:** Demographic parity + equalized odds. Для каждого демографического сегмента (country, entity type). # ponytail: metric set, add when new demographics emerge.

### [agent] resolved
> Demographic parity + equalized odds per group.

## 4. Bootstrap & A/B Testing

```python
# src/statistics/bootstrap.py
class BootstrapCI:
    """1000 bootstrap iterations, 95% CI for AUC-PR."""
    def compute(self, scores, labels, n_iterations=1000) -> dict:
        """Bootstrap confidence intervals."""
        auc_prs = [compute_auc_pr(*resample(scores, labels)) for _ in range(n_iterations)]
        return {"mean": np.mean(auc_prs), "ci_lower": np.percentile(auc_prs, 2.5), "ci_upper": np.percentile(auc_prs, 97.5)}
```

```python
# src/human_eval/ab_test.py
class ABTester:
    """Double-blind human evaluation for model preference."""
    def __init__(self, n_analysts=20):
        self.n_analysts = n_analysts
    def run(self, alerts_a, alerts_b) -> dict:
        """Mann-Whitney U test for preference significance."""
        return {"u_statistic": ..., "p_value": ..., "preference": ...}
```

### [human] Bootstrap iterations
> 1000 итераций — достаточно?

**Decision:** 1000 для 95% CI. Если width > 0.05 — увеличить до 5000. # ponytail: iterations, add when CI width > 0.05.

### [agent] resolved
> 1000 base, 5000 if CI width > 0.05.

## 5. Key Components

- `baselines.py`: Heuristic, LR, GCN, GAT suite. English comments.
- `calibration.py`: Platt scaling + ECE. Calibration error < 0.05. English comments.
- `fairness.py`: Demographic parity + equalized odds. English comments.
- `bootstrap.py`: 1000 iterations, 95% CI. English comments.
- `ab_test.py`: Double-blind A/B, Mann-Whitney U. English comments.
- Tests: 95% coverage.

## 6. Pushback Hypotheses

1. **H1**: GAT overfits on small graph data. *Mitigation*: Dropout p=0.5 + early stopping. GCN fallback if < 2% improvement. `# ponytail:` GAT threshold
2. **H2**: Human A/B introduces bias. *Mitigation*: Double-blind + Mann-Whitney U. Preference gap < 5% = inconclusive.

## 7. Self-Review

- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- 4 baselines + metrics documented
- Pushback hypotheses: 2 ✅
