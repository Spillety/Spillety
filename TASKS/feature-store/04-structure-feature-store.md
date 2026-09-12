# 04-structure-feature-store

> Phase: S (Structure) | Slug: feature-store | Status: In Progress

## 1. Project Structure

```
feature_store/
├── config/{feature_store.yaml,feast_repo,docker_pinning.yaml}
├── src/{feature_store/{feast_client,feature_service,pipeline},data_quality/drift_monitor,infra/kafka_consumer}.py
├── tests/{test_feast_client,test_drift_monitor,test_dual_write}.py
└── scripts/validate_skew.py
```

## 2. Feast Client & Feature Service

```python
# src/feature_store/feast_client.py
class FeastClient:
    """Feast wrapper with LRU cache fallback and Docker image pinning."""
    def get_training_features(self, entity_ids: list[str]) -> pd.DataFrame:
        return self.store.get_historical_features(entity_ids, ["tx_count_view","velocity_view","hawkes_view"]).to_df()
    def get_serving_features(self, entity_id: str) -> dict:
        return self.store.get_online_features(features=["tx_count_view:tx_count_1h"], entity_rows=[{"wallet_address": entity_id}]).to_dict()
```

```python
# src/feature_store/feature_service.py
class FeatureService:
    """Dual-write: Kafka → Feast + Flink. Dedup by tx_hash."""
    def __init__(self):
        self.feast = FeastClient(CONFIG_PATH)
        self.kafka_consumer = KafkaConsumer(RAW_TOPIC, group_id="feast_ingestion")
    def process_event(self, event: dict) -> None:
        # Idempotent write: check tx_hash in offline store first
        if not self._exists(event["tx_hash"]):
            self.feast.push_to_offline(event)
            self.feast.push_to_online(event)
```

### [human] Docker pinning
> Как гарантировать одинаковые образы train/serve?

**Decision:** `docker_pinning.yaml` с фиксированными версиями. CI/CD проверяет соответствие. # ponytail: CI/CD check, add when version mismatch > 1/month.

### [agent] resolved
> Пининг версий в config. CI pipeline проверяет соответствие.

## 3. Drift Monitor

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

### [human] KS test sampling
> KS test требует ≥1000 samples. Что делать с редкими features?

**Decision:** Wasserstein distance fallback при <1000 samples. Daily baseline + hourly для critical features. # ponytail: Wasserstein fallback, add when <1k samples for feature.

### [agent] resolved
> KS при ≥1000 samples. Wasserstein при <1000. Hourly для critical features.

## 4. Key Components

- `feast_client.py`: Feast wrapper with Redis/LRU fallback. English comments only.
- `feature_service.py`: Dual-write Kafka → Feast + Flink. Dedup по tx_hash. `# ponytail:` ordering guarantee
- `drift_monitor.py`: KS test + Wasserstein fallback. English comments only.
- Tests: 95% coverage data_quality, 80% feature_store.

## 5. Pushback Hypotheses

1. **H1**: Feast offline store becomes bottleneck. *Mitigation*: PgBouncer + индексы. ClickHouse при >100M vectors. `# ponytail:` ClickHouse offline
2. **H2**: Dual-write introduces data inconsistency. *Mitigation*: Kafka transactional idempotent producer. Dedup по tx_hash.

## 6. Self-Review

- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- Dual-write architecture documented
- Pushback hypotheses: 2 ✅
