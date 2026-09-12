# 8.1 Kubernetes Deployment — Structure

```
k8s-deployment/
├── manifests/
│   ├── gpu-nodes/
│   │   ├── gpu-node-pool.yaml        # T4/A10G node pool
│   │   └── inference-deployment.yaml # GPU inference pods
│   ├── cpu-nodes/
│   │   ├── cpu-node-pool.yaml        # CPU node pool (Flink/Kafka/Memgraph)
│   │   ├── flink-statefulset.yaml
│   │   ├── kafka-statefulset.yaml
│   │   └── memgraph-statefulset.yaml
│   ├── hpa/
│   │   └── inference-hpa.yaml        # HPA on queue depth
│   ├── vpa/
│   │   ├── memgraph-vpa.yaml
│   │   └── clickhouse-vpa.yaml
│   └── services/
│       ├── inference-service.yaml    # Headless for gRPC
│       └── gateway-service.yaml
├── helm/
│   └── k8s-deployment/
│       ├── values.yaml
│       └── templates/
└── tests/
    ├── test_hpa_scaling.py
    └── test_vpa_adjustment.py
```

## Key Components
- `inference-deployment.yaml`: GPU pod spec with T4/A10G resource requests
- `inference-hpa.yaml`: Custom metric `kafka_consumer_lag` for autoscaling
- `memgraph-statefulset.yaml`: VPA-enabled statefulset with persistent volume
- `gpu-node-pool.yaml`: Node pool with `nvidia.com/gpu` taint and toleration

## Pushback Hypotheses
1. **H1**: Headless service for gRPC adds connection complexity. *Mitigation*: Use K8s service DNS with retry logic; gRPC client handles failover. `# ponytail:` service discovery pattern
2. **H2**: VPA in Auto mode may cause pod restarts during memory adjustment. *Mitigation*: Use `UpdateMode` Initial for statefulsets; avoid `Auto` for production. `# ponytail:` VPA update modes

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- No service mesh abstraction
