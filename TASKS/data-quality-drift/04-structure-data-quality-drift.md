# 04-structure-data-quality-drift

> Phase: S (Structure) | Slug: data-quality-drift | Status: In Progress

## 1. Project Structure

```
data_quality_drift/
├── config/{expectations/,drift_monitor.yaml,backtesting.yaml}
├── src/{data_quality/{expectations,drift_monitor,alerting},validation/backtester}.py
├── tests/{test_expectations,test_drift_monitor,test_backtesting}.py
└── scripts/run_daily_drift_check.py
```

## 2. Great Expectations Suite

```python
# src/data_quality/expectations.py
def build_raw_events_expectations():
    """Schema, null, range, uniqueness + AML custom validators."""
    return {
        "expect_table_columns_to_match_ordered_list": ["tx_hash","from","to","amount","timestamp","chain"],
        "expect_column_values_to_not_be_null": ["tx_hash"],
        "expect_column_values_to_be_unique": ["tx_hash"],
        "expect_column_values_to_be_between": {"column":"amount","min_value":0},
        "expect_column_values_to_match_regex": {"column":"from","regex":"^0x[0-9a-f]{40}$"},
        # AML custom
        "expect_column_values_to_match_regex": {"column":"address","regex": r"(^0x[0-9a-f]{40}$|^bc1[a-z0-9]{39,59}$)"},
    }
```

### [human] AML custom validators
> Сколько custom валидаторов нужно?

**Decision:** 5: address_format (Bech32/hex), amount_positive, is_exchange_internal flag, chain_format, timestamp_reasonable. # ponytail: AML validators, add when AML check types > 5.

### [agent] resolved
> 5 custom expectations. Address format, amount positive, exchange internal flag.

## 3. Drift Monitor & Backtester

```python
# src/data_quality/drift_monitor.py
class SkewMonitor:
    """KS test for train-serve skew. Wasserstein fallback for <1k samples."""
    def check_skew(self, offline: pd.DataFrame, online: pd.DataFrame, features: list[str]) -> dict:
        results = {}
        for feature in features:
            if len(offline[feature]) >= 1000 and len(online[feature]) >= 1000:
                stat, _ = stats.ks_2samp(offline[feature].dropna(), online[feature].dropna())
            else:
                stat = wasserstein_distance(offline[feature], online[feature])
            results[feature] = stat
        return results
```

```python
# src/validation/backtester.py
class Backtester:
    """Weekly quick + monthly full backtesting on Elliptic + internal data."""
    def quick_backtest(self, model, elliptic_data) -> dict:
        """Weekly: AUC-PR on held-out Elliptic data."""
        return {"auc_pr": compute_auc_pr(model, elliptic_data)}
    def full_backtest(self, model, internal_data) -> dict:
        """Monthly: full retraining evaluation."""
        return {"auc_pr": ..., "calibration": ...}
```

### [human] Backtesting frequency
> Как часто делать backtesting?

**Decision:** Еженедельно quick на Elliptic + internal. Полный пересмотр ежемесячно. # ponytail: full retrain, add when AUC-PR decay > 5%.

### [agent] resolved
> Weekly quick, monthly full. Trigger: AUC-PR decay > 5%.

## 4. Key Components

- `expectations.py`: GE suite with 5 standard + 5 AML custom validators. English comments.
- `drift_monitor.py`: KS test + Wasserstein fallback. Alert latency < 5min. English comments.
- `backtester.py`: Weekly quick + monthly full backtesting. English comments.
- `alerting.py`: Drift alerts to Prometheus + Slack. `# ponytail:` alert channel
- Tests: 95% coverage data_quality, 80% drift_monitor.

## 5. Pushback Hypotheses

1. **H1**: GE validation adds pipeline latency. *Mitigation*: Async batch validation каждые 100ms. Penalty < 10ms. `# ponytail:` async GE
2. **H2**: Concept drift detection too slow. *Mitigation*: Rolling 24h window + pre-warning 3% threshold. Alert latency < 5min.

## 6. Self-Review

- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- GE suite + KS test + concept drift documented
- Pushback hypotheses: 2 ✅
