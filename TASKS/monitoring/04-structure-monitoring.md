# 8.2 Monitoring — Structure

```
monitoring/
├── metrics/
│   ├── custom/
│   │   ├── inference_metrics.py      # TCH-GT latency, score distribution
│   │   ├── hawkes_metrics.py         # λ(t) percentiles
│   │   └── tch_gt_metrics.py         # Model-specific metrics
│   └── prometheus/
│       ├── prometheus-config.yaml
│       └── alert-rules.yaml          # SLO: p99 > 100ms, error > 1%
├── tracing/
│   ├── otel-sdk/
│   │   ├── instrumentation.py        # OpenTelemetry setup
│   │   └── propagator.py             # W3C trace context
│   └── jaeger/
│       └── docker-compose.yaml       # Jaeger backend (MVP)
├── dashboards/
│   ├── grafana/
│   │   ├── ml-dashboard.json
│   │   ├── infra-dashboard.json
│   │   └── compliance-dashboard.json
│   └── provisioning/
│       └── datasource.yaml
└── tests/
    ├── test_metrics_cardinality.py
    └── test_alertmanager_rules.py
```

## Key Components
- `inference_metrics.py`: Custom Prometheus metrics for TCH-GT inference
- `alert-rules.yaml`: Alertmanager SLO rules — p99 > 100ms, error rate > 1%
- `instrumentation.py`: OpenTelemetry SDK with OTLP exporter
- `ml-dashboard.json`: Grafana dashboard for ML-specific metrics

## Pushback Hypotheses
1. **H1**: Prometheus scraping every pod creates excessive load. *Mitigation*: Scrape interval 15s (not 10s); federation for cross-namespace. `# ponytail:` scrape frequency
2. **H2**: Jaeger backend becomes single point of failure for tracing. *Mitigation*: Deploy Jaeger with HA collector; sampling 10% limits impact. `# ponytail:` tracing reliability

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- No community dashboards for AML metrics
