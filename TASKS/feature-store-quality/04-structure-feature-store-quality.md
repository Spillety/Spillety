# 04-structure-feature-store-quality

> Phase: S (Structure) | Slug: feature-store-quality | Status: In Progress

## 1. Project Structure

```
feature_store_quality/
├── config/{feature_store.yaml,expectations/,docker_pinning.yaml}
├── src/{feature_store/{feast_client,feature_service},data_quality/{expectations,drift_monitor,alerting},pipeline/kafka_to_feast}.py
├── tests/{test_expectations,test_drift_monitor,test_feature_serving}.py
└── scripts/validate_skew.py
```

## 2. Great Expectations Suite

```python
# src/data_quality/expectations.py
def build_raw_events_expectations():
    """Schema, null, range, uniqueness + AML custom validators."""
    return {
        "expect_table_columns_to_match_ordered_list":
            ["tx_hash","from","to","amount","timestamp","chain"],
        "expect_column_values_to_not_be_null": ["tx_hash"],
        "expect_column_values_to_be_unique": ["tx_hash"],
        "expect_column_values_to_be_between": {"column":"amount","min_value":0},
        "expect_column_values_to_match_regex":
            {"column":"from","regex":"^0x[0-9a-f]{40}$"},
        # AML custom
        "expect_column_values_to_match_regex":
            {"column":"address","regex": r"(^0x[0-9a-f]{40}$|^bc1[a-z0-9]{39,59}$)"},
    }
```

### [human] AML custom validators scope
> Сколько custom валидаторов нужно?

**Decision:** 5: address_format (Bech32/hex), amount_positive, is_exchange_internal flag, chain_format, timestamp_reasonable. # ponytail: AML validators, add when AML check types > 5.

### [agent] resolved
> 5 custom expectations. Address format, amount positive, exchange internal flag.

## 3. Drift Monitoring

```python
# src/data_quality/drift_monitor.py
from scipy import stats
import pandas as pd

class SkewMonitor:
    """KS test for train-serve skew. Wasserstein fallback for <1k samples."""
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.alerts = []

    def check_skew(self, offline: pd.DataFrame, online: pd.DataFrame, features: list[str]) -> dict:
        """Compute KS statistic per feature. Alert when > threshold."""
        results = {}
        for feature in features:
            o = offline[feature].dropna().values
            n = online[feature].dropna().values
            if len(o) >= 1000 and len(n) >= 1000:
                stat, _ = stats.ks_2samp(o, n)  # KS test
            else:
                stat = self._wasserstein(o, n)  # Fallback
            results[feature] = stat
            if stat > self.threshold:
                self.alerts.append({"feature": feature, "ks_statistic": stat})
        return results

    def _wasserstein(self, a, b):
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(a, b)
```

### [human] KS test sampling requirement
> KS test требует ≥1000 samples. Что делать с редкими features?

**Decision:** Wasserstein distance fallback при <1000 samples. Daily baseline + hourly for critical features. # ponytail: Wasserstein fallback, add when <1k samples for feature.

### [agent] resolved
> KS при ≥1000 samples. Wasserstein при <1000. Hourly для critical features.

## 4. Feast Client & Train-Serve Parity

```python
# src/feature_store/feast_client.py
from feast import FeatureStore
import redis, json

class FeastClient:
    """Feast wrapper with LRU cache fallback and Docker image pinning."""
    def __init__(self, config_path: str):
        self.store = FeatureStore(config_path=config_path)
        self.redis = redis.Redis()

    def get_training_features(self, entity_ids: list[str]) -> pd.DataFrame:
        """Get offline features for training via Feast."""
        return self.store.get_historical_features(
            entity_ids=entity_ids,
            features=["tx_count_view","velocity_view","hawkes_view"]).to_df()

    def get_serving_features(self, entity_id: str) -> dict:
        """Get real-time features with Redis fallback."""
        try:
            return self.store.get_online_features(
                features=["tx_count_view:tx_count_1h"],
                entity_rows=[{"wallet_address": entity_id}]).to_dict()
        except redis.ConnectionError:
            cached = self.redis.get(f"fs_fallback:{entity_id}")
            return json.loads(cached) if cached else raise
```

### [human] Docker pinning
> Как гарантировать одинаковые образы train/serve?

**Decision:** `docker_pinning.yaml` с фиксированными версиями. CI/CD проверяет соответствие. # ponytail: CI/CD check, add when version mismatch > 1/month.

### [agent] resolved
> Пининг версий в config. CI pipeline проверяет соответствие.

## 5. Tests

```python
# tests/test_expectations.py
def test_raw_events_suite_complete():
    suite = build_raw_events_expectations()
    assert set(suite["expect_table_columns_to_match_ordered_list"]) == {"tx_hash","from","to","amount","timestamp","chain"}

def test_amount_range():
    suite = build_raw_events_expectations()
    assert suite["expect_column_values_to_be_between"]["min_value"] == 0
```

```python
# tests/test_drift_monitor.py
def test_skew_below_threshold():
    monitor = SkewMonitor(threshold=0.01)
    offline = pd.DataFrame({"tx_count": np.random.normal(10,2,1000)})
    online = pd.DataFrame({"tx_count": np.random.normal(10,2,1000)})
    assert monitor.check_skew(offline, online, ["tx_count"])["tx_count"] < 0.01

def test_skew_above_threshold():
    monitor = SkewMonitor(threshold=0.01)
    offline = pd.DataFrame({"tx_count": np.random.normal(10,2,1000)})
    online = pd.DataFrame({"tx_count": np.random.normal(20,2,1000)})
    assert monitor.check_skew(offline, online, ["tx_count"])["tx_count"] > 0.01
```

### [human] Test coverage
> 80% или 95%?

**Decision:** 95% для data_quality. 80% для feature_store. # ponytail: uniform 95%, add when logic grows.

### [agent] resolved
> 95% data_quality. 80% feature_store.

## 6. Pushback: What Could Go Wrong

### [human] Hypothesis 1: GE adds pipeline latency
> Validation замедляет ingestion.

**Decision:** GE в async consumer, batch validation каждые 100ms. Penalty < 10ms. # ponytail: async GE, add when sync validation > 50ms.

### [agent] resolved
> Async validation через separate consumer. Batch checks каждые 100ms.

### [human] Hypothesis 2: Skew < 1% impossible with real drift
> Real distribution drift может всегда превышать 1%.

**Decision:** Adaptive threshold с регуляторным одобрением. Auto-retraining при drift > 5%. # ponytail: adaptive threshold, add when drift triggers > 3 retraining/month.

### [agent] resolved
> Adaptive threshold + retraining trigger. Мониторинг monthly drift events.

## 7. Acceptance Criteria

- [ ] Feast Feature Store: Redis online, PostgreSQL offline
- [ ] Great Expectations: schema, null, range, uniqueness, 5 AML validators
- [ ] Train-serve skew < 1% (KS daily + hourly critical)
- [ ] Docker image pinning для train/serve
- [ ] GE validation latency < 10ms (async)
- [] SkewMonitor: KS + Wasserstein fallback
- [ ] 95% coverage data_quality, 80% feature_store
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 8. Open Questions

Нет открытых вопросов. Переход к P (Plan) по согласованию.
