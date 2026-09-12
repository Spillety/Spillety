# 8.1 Kubernetes Deployment — Design

## Context
Managed K8s (EKS/GKE) для AML-платформы. GPU T4/A10G для inference, CPU-only для Flink/Kafka/Memgraph. Нет service mesh для MVP. HPA по queue depth, VPA для stateful.

### [human] Design Decisions
- **Managed K8s**: EKS или GKE — self-managed слишком сложен для MVP
- **GPU узлы**: T4/A10G для inference (lazy inference engine); CPU-only для Flink, Kafka, Memgraph
- **Без service mesh**: Istio добавляет 2-5ms latency и operational overhead; используем native K8s Services + gRPC
- **HPA по queue depth**: Custom metric `kafka_consumer_lag` для autoscaling inference pods
- **VPA для stateful**: Memgraph и ClickHouse нуждаются в VPA, т.к. memory usage непредсказуем

### [agent] resolved
- GPU node pool: `g4dn.xlarge` (T4) или `g5.xlarge` (A10G); label `gpu=true`
- CPU node pool: `m5.2xlarge` для Flink/Kafka/Memgraph; label `cpu-only=true`
- HPA: `kubectl autoscale deployment inference --cpu-percent=70 --min=2 --max=10`
- VPA: `VerticalPodAutoscaler` в `Auto` mode для statefulsets
- Service mesh: отключен; headless services для gRPC discovery

## Architecture
```
Ingress → K8s Service → Inference Deployment (GPU)
  ├── HPA: kafka_consumer_lag > 1000 → scale up
  ├── GPU Node Pool: T4/A10G
  └── CPU Node Pool: Flink + Kafka + Memgraph
       ├── VPA: memory auto-adjust
       └── Headless Service: gRPC discovery
```

**Node pools**: 2 GPU (t4.xlarge), 3 CPU (m5.2xlarge)

## Pushback Hypotheses
1. **H1**: EKS/GKE managed control plane стоит дороже чем self-managed. *Mitigation*: Self-managed операционный долг > $50k/год; managed K8s экономит engineer time. `# ponytail:` cost vs. ops burden
2. **H2**: HPA по queue depth может вызвать flapping при нестабильном traffic. *Mitigation*: Добавить stabilization window 300s и downscale stabilization 600s. `# ponytail:` aggressive vs. conservative scaling

## Open Questions
- Какой GPU тип выбрать: T4 (дешевле) или A10G (быстрее для LLM explainer)?
- Нужен ли отдельный node pool для Memgraph с высоким RAM?

## [human] Acceptance
- p99 inference latency < 100ms на инфраструктурном уровне
- Zero service mesh components deployed
- HPA scales within 2min of queue depth increase
- VPA adjusts memory without pod restart
