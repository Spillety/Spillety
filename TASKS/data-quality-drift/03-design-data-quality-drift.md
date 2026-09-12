# 9.3 Data Quality & Drift Monitoring — Design

> Phase: D (Design) | Slug: data-quality-drift | Status: In Progress

## 1. Great Expectations для On-Chain Events

Great Expectations для schema validation + anomaly detection на raw on-chain events. Bad data → bad model.

### [human] GE scope
> Какие валидации включить в Great Expectations suite?

**Decision:** Schema validation, null checks, range checks для amount/timestamp, uniqueness для tx_hash. Custom для AML: address_format, amount_positive, chain_format. # ponytail: custom validators, add when AML checks > 20.

### [agent] resolved
> 5 standard + 5 custom expectations. Address format, amount positive, chain format, timestamp reasonable, is_exchange_internal flag.

## 2. Feature Distribution Drift: KS Test

KS test для feature distributions между training и serving. Alert при значительном сдвиге.

### [human] Drift detection method
> KS test или другой метод для drift detection?

**Decision:** KS test для каждого feature. Alert при p<0.01 или drift > 1%. Wasserstein fallback для <1000 samples. # ponytail: Wasserstein fallback, add when <1k samples.

### [agent] resolved
> KS test daily. Wasserstein при <1000 samples. Hourly для critical features.

## 3. Concept Drift: AUC-PR Decay > 5%

Performance degradation monitoring. AUC-PR decay > 5% — trigger для retraining.

### [human] Concept drift vs data drift
> Как отличить concept drift от data drift?

**Decision:** Data drift = feature distribution shift (KS test). Concept drift = label relationship change (AUC-PR decay). Отдленные метрики, разные alert'и. # ponytail: combined alert, add when both drift types active simultaneously.

### [agent] resolved
> Два типа drift: data (KS) и concept (AUC-PR). Раздельные alerts. Combined alert если оба > threshold.

## 4. Backtesting

Elliptic dataset + internal labeled data. Регулярный пересмотр модели на свежих данных.

### [human] Backtesting frequency
> Как часто делать backtesting?

**Decision:** Еженедельно на Elliptic + internal labeled data. Полный пересмотр ежемесячно. # ponytail: full retrain, add when AUC-PR decay > 5% on backtest.

### [agent] resolved
> Weekly quick backtest, monthly full retest. Trigger: AUC-PR decay > 5%.

## 5. Pushback: What Could Go Wrong

### [human] Hypothesis 1: GE validation adds pipeline latency
> GE validation замедляет ingestion pipeline.

**Decision:** GE в async consumer, batch validation каждые 100ms. Penalty < 10ms. # ponytail: async GE, add when sync validation > 50ms.

### [agent] resolved
> Async validation через separate consumer. Batch checks каждые 100ms. Penalty < 10ms.

### [human] Hypothesis 2: Concept drift detection too slow
> AUC-PR decay обнаруживается только после потери бизнес-ценности.

**Decision:** Rolling AUC-PR на sliding window 24h. Alert при 3% decay (pre-warning), rollback при 5%. # ponytail: pre-warning threshold, add when false positive rate > 10%.

### [agent] resolved
> Rolling window 24h. Pre-warning 3%, critical 5%. Alert latency < 5min.

## 6. Acceptance Criteria

- [ ] Great Expectations suite: schema, null, range, uniqueness + AML custom validators
- [ ] KS test для feature distributions, alert при > 1% drift
- [ ] Concept drift: AUC-PR decay > 5% trigger retraining
- [ ] Backtesting: weekly quick + monthly full
- [ ] Drift detection alert latency < 5min
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 7. Pushback Summary

| # | Гипотеза | Решение |
|---|----------|---------|
| 1 | GE adds pipeline latency | Async batch validation, penalty < 10ms |
| 2 | Concept drift detection too slow | Rolling 24h window, pre-warning 3% |
