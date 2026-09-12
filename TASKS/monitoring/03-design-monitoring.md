# 8.2 Monitoring — Design

## Context
Custom metrics для TCH-GT, Prometheus + Alertmanager для SLO (p99 > 100ms, error rate > 1%), OpenTelemetry + Jaeger для tracing, custom Grafana dashboards.

### [human] Design Decisions
- **Custom metrics**: TCH-GT inference latency, score distribution, Hawkes λ(t) percentiles; стандартные метрики для инфраструктуры
- **SLO via Alertmanager**: p99 latency > 100ms, error rate > 1% → pages; p99 > 200ms → critical
- **Tracing**: OpenTelemetry SDK в каждом сервисе, Jaeger как backend
- **Grafana**: Custom dashboards; community dashboards не покрывают AML-метрики

### [agent] resolved
- Custom metrics: `inference_latency_ms`, `tch_gt_score_distribution`, `hawkes_lambda_percentile` — экспортируются через Prometheus client
- Alertmanager rules: `avg_over_time(inference_latency_ms_p99[5m]) > 100` → warning; `> 200` → critical
- Error rate: `rate(inference_errors_total[5m]) / rate(inference_requests_total[5m]) > 0.01`
- OpenTelemetry: OTLP exporter → Jaeger collector; sampling rate 10% для MVP
- Grafana: datasource Prometheus; dashboards per team (ML, Infra, Compliance)

## Architecture
```
Services → OpenTelemetry SDK → OTLP → Jaeger
  ├── Prometheus: scrape metrics (custom + infra)
  │   ├── Alertmanager: SLO evaluation → pages
  │   └── Grafana: custom dashboards
  └── Custom Metrics: TCH-GT, Hawkes λ(t), score distribution
```

**Retention**: Prometheus 14d, Jaeger 7d, Grafana dashboards version-controlled

## Pushback Hypotheses
1. **H1**: Custom metrics для TCH-GT могут иметь high cardinality → Prometheus OOM. *Mitigation*: Label cardinality limits; `__name__` and label restrictions; shard by team. `# ponytail:` cardinality vs. granularity
2. **H2**: Jaeger tracing at 10% sampling misses rare failures. *Mitigation*: Adaptive sampling — 100% for error traces, 10% for success. `# ponytail:` sampling strategy

## Open Questions
- Какие именно AML-метрики нужны для Grafana dashboards?
- Нужен ли unified view combining TCH-GT + Hawkes metrics?

## [human] Acceptance
- p99 inference latency < 100ms, alert fires within 5min of breach
- Error rate > 1% triggers Alertmanager page
- All services emit OpenTelemetry traces
- Custom Grafana dashboards cover all critical paths
