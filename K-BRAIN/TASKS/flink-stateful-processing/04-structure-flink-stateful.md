# 04-structure-flink-stateful

> Phase: S (Structure) | Slug: flink-stateful | Status: In Progress

## 1. Project Structure

```
flink/
├── config/
│   ├── flink-conf.yaml        # Flink configuration
│   ├── rocksdb-state-backend.properties
│   └── checkpointing.properties
├── src/
│   ├── pageRank/
│   │   ├── IncrementalPageRank.java      # KeyedProcessFunction + Monte Carlo
│   │   └── RandomWalkSegment.java
│   ├── hawkes/
│   │   ├── PowerLawIntensityFunction.java  # Numerical integration λ(t)
│   │   └── HawkesStateDescriptor.java
│   ├── state/
│   │   └── RocksDBStateConfig.java
│   └── utils/
│       └── CheckpointConfig.java
├── docker/
│   └── Dockerfile.flink
└── k8s/
    └── flink-deployment.yaml
```

### [human] Power-law vs exponential kernel
> User выбрал power-law kernel. Что меняется в реализации?

### [agent] resolved
> Power-law kernel выбран пользователем. Численное интегрирование для λ(t). TTL пересмотрен на 14 дней. # ponytail: O(1) approximate updater, add when numerical integration exceeds 5ms latency.

## 2. Flink Configuration

```yaml
# flink-conf.yaml
state.backend: rocksdb
state.checkpoints.dir: s3://flink-checkpoints/
state.backend.rocksdb.ttl: 14d    # Hawkes (power-law)
state.backend.rocksdb.pagerank-ttl: 30d  # PageRank
execution.checkpointing.interval: 60s
execution.checkpointing.mode: EXACTLY_ONCE
```

### [human] TTL for power-law
> Power-law kernel имеет long-range хвост. 14 дней — достаточно?

### [agent] resolved
> Да, для MVP 14 дней покрывает power-law tail. При production может потребоваться пересмотр. # ponytail: TTL extension, add when long-range accuracy < 0.95.

## 3. RocksDB State Backend Config

```java
StateBackend rocksdb = new RocksDBStateBackend(
    "s3://flink-checkpoints/",
    true  // incremental checkpointing
);
RocksDBStateBackendConfig config = new RocksDBStateBackendConfig();
config.setWriteBufferSize(256 * 1024 * 1024);  // 256MB
config.setCompactFilterFactory(new TTLCompactFilterFactory());
```

## 4. Incremental PageRank Implementation

```java
public class IncrementalPageRank extends KeyedProcessFunction<Bytes, Event, PageRankUpdate> {
    /**
     * Monte Carlo approximation for incremental PageRank.
     * Stores random walk segments per node, updates only affected segments on new edge.
     * Expected speedup: >= 10x vs full recomputation.
     */
    
    private transient ListState<RandomWalkSegment> segmentsState;
    
    @Override
    public void processElement(Event event, Context ctx, Collector<PageRankUpdate> out) {
        // Identify affected segments
        // Monte Carlo resample affected segments
        // Update PageRank
    }
}
```

## 5. Power-Law Hawkes Intensity Function

```python
class PowerLawHawkesIntensity:
    """
    Numerical integration for power-law kernel λ(t).
    Unlike exponential kernel (closed-form O(1)), power-law requires numerical integration.
    TTL: 14 days for power-law long-range dependency.
    """
    
    def compute_intensity(self, address: str, t: float) -> float:
        """
        Compute λ(t) via numerical integration of power-law kernel.
        """
        pass
```

### [human] Numerical integration performance
> Численное интегрирование может быть медленным при высоком throughput.

### [agent] resolved
> Батчевый подсчёт с предвычислением для hot addresses. При >5ms latency — переход к O(1) approximate updater. # ponytail: O(1) approximate updater, add when numerical integration exceeds 5ms latency.

### [human] Monte Carlo instability for low-degree nodes
> Для узлов с низкой степенью Monte Carlo может быть нестабильным.

### [agent] resolved
> Минимальное количество samples = 1000. Для low-degree nodes — deterministic fallback.

## 6. Exactly-Once Checkpointing

```java
CheckpointConfig config = new CheckpointConfig();
config.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
config.setCheckpointInterval(60000);  // 60s
config.setMinPauseBetweenCheckpoints(30000);
config.setCheckpointTimeout(300000);  // 5min
config.enableExternalizedCheckpoints(ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
```

## 7. State TTL Configuration

| State Type | TTL | Kernel |
|---|---|---|
| Hawkes intensity | 14 дней | Power-law |
| PageRank | 30 дней | Iterative |

### [human] TTL cleanup mechanism
> Flink state TTL работает с lazy cleanup — state удаляется при следующем access.

### [agent] resolved
> Flink гарантирует atomicity. No race condition — lazy cleanup при access.

## 8. Docker/K8s Scaffold

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-jobmanager
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: flink-jobmanager
          resources:
            limits:
              memory: "4Gi"
              cpu: "2000m"
        - name: flink-taskmanager
          replicas: 2
          resources:
            limits:
              memory: "8Gi"
              cpu: "4000m"
              nvidia.com/gpu: 0  # CPU-only for Flink
```

## 9. Pushback: What Could Go Wrong

### [human] Hypothesis 1: Numerical integration bottleneck
> При высоком throughput численное интегрирование λ(t) становится bottleneck.

### [agent] resolved
> Батчевый подсчёт + предвычисление для hot addresses. При >5ms latency — O(1) approximate updater.

### [human] Hypothesis 2: Monte Carlo instability for low-degree nodes
> Для узлов с малой степенью выборка Monte Carlo может быть нестабильной.

### [agent] resolved
> Минимум 1000 samples + deterministic fallback для low-degree nodes.

## 10. Acceptance Criteria

- [ ] Hawkes TTL = 14 дней (power-law kernel)
- [ ] Power-law numerical integration implemented
- [ ] Incremental PageRank with Monte Carlo (speedup ≥ 10x)
- [ ] Exactly-once checkpointing (60s interval)
- [ ] `# ponytail: O(1) approximate updater` marker
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 11. Open Questions

Нет открытых вопросов для фазы S. Переход к P (Plan) по согласованию.

## 12. User Answers

- **Kernel choice**: Power-law (подтверждён)
- **TTL**: 14 дней для Hawkes
- **Pushback accepted**: Numerical integration bottleneck — O(1) approximate updater as fallback
