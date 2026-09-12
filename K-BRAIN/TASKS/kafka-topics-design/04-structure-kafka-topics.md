# 04-structure-kafka-topics

> Phase: S (Structure) | Slug: kafka-topics | Status: In Progress

## 1. Project Structure

```
kafka/
├── config/
│   ├── server.properties      # Kafka broker config
│   ├── schema-registry.properties
│   └── producer.properties
├── proto/
│   └── events.proto           # Protobuf schema definitions
├── src/
│   ├── producer/
│   │   └── ProducerService.java
│   ├── consumer/
│   │   └── ConsumerService.java
│   └── utils/
│       └── SchemaRegistryClient.java
├── docker/
│   └── Dockerfile.kafka
└── k8s/
    └── kafka-deployment.yaml
```

### [human] Schema cache module
> Нужен отдельный модуль для Schema Registry кэша?

### [agent] resolved
> Нет. `functools.lru_cache` вместо отдельного `schema_cache.py` для MVP. При росте >3 consumer groups — вынести в отдельный модуль. # ponytail: separate schema_cache.py, add when >3 consumer groups.

## 2. Kafka Topic Definitions

```properties
# server.properties
num.partitions=6
default.replication.factor=1
retention.ms=7776000000
log.segment.bytes=1073741824
```

- **Topic `raw-events`**: 6 partitions, key=`from_address`
- **Topic `processed-events`**: 6 partitions
- **Topic `dead-letter-events`**: 1 partition, retention=infinite

### [human] Partition count
> 6 partition — достаточно для MVP?

### [agent] resolved
> Да, для MVP 6 partition покрывает throughput. HPA до 5 consumer instances при росте.

## 3. Protobuf Schema

```protobuf
syntax = "proto3";
package kafka.events;

message OnChainEvent {
  bytes tx_hash = 1;
  bytes from_address = 2;
  bytes to_address = 3;
  double amount = 4;
  int64 timestamp = 5;
  string chain_id = 6;
}
```

### [agent] resolved
> `BACKWARD` compatibility для internal-событий (producer/consumer деплоятся вместе). Field numbers зарезервированы для будущих расширений.

## 4. Producer/Consumer Scaffold

```python
class EventProducer:
    """Protobuf-based Kafka producer for on-chain events."""
    
    def send(self, event: OnChainEvent) -> Future:
        """Send event to raw-events topic with from_address partition key."""
        pass

class EventConsumer:
    """Kafka consumer with DLQ handling."""
    
    def consume(self) -> List[OnChainEvent]:
        """Consume from processed-events, route failures to dead-letter-events."""
        pass
```

## 5. DLQ Configuration

- **Topic**: `dead-letter-events` — infinite retention
- **Circuit breaker**: при >5% DLQ rate — pause producer
- **Prometheus alert**: `kafka_dlq_rate > 0.001`
- **No auto-resume**: manual review required

### [human] Auto-resume DLQ
> Нужен автоматический возврат из DLQ?

### [agent] resolved
> Нет. Anti-pattern для AML. Ручной анализ обязателен. # ponytail: auto-resume DLQ, add when operational team confirms.

## 6. Prometheus Alert Rules

```yaml
groups:
  - name: kafka_alerts
    rules:
      - alert: KafkaDLQRate
        expr: rate(kafka_topic_partition_current_offset{topic="dead-letter-events"}[5m]) > 0.001
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "DLQ rate exceeds 0.1%"
      - alert: KafkaConsumerLag
        expr: kafka_consumer_group_lag{group="event-processor"} > 10000
        for: 5m
        labels:
          severity: warning
```

## 7. Docker/K8s Scaffold

```yaml
# k8s/kafka-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka-broker
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: kafka
          resources:
            limits:
              memory: "4Gi"
              cpu: "2000m"
```

## 8. Pushback: What Could Go Wrong

### [human] Hypothesis 1: Partition key hot-spotting
> `from_address` как partition key создаёт hot partition для exchange addresses.

### [agent] resolved
> Custom partitioner добавляет suffix при превышении threshold. Для MVP — допустимо. # ponytail: custom partitioner, add when hot partition causes >20% latency increase.

### [human] Hypothesis 2: Protobuf schema evolution break
> Поле field number может конфликтовать при эволюции схемы.

### [agent] resolved
> Schema Registry enforcement: reserved field numbers, backward compatibility policy. Breaking changes невозможны.

## 9. Acceptance Criteria

- [ ] `functools.lru_cache` вместо отдельного schema_cache.py
- [ ] `BACKWARD` compatibility для internal-событий
- [ ] 1 replica consumer для MVP, HPA до 5 при росте
- [ ] DLQ circuit breaker + Prometheus alert настроены
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 10. Open Questions

Нет открытых вопросов для фазы S. Переход к P (Plan) по согласованию.

## 11. User Answers

- **Kernel choice**: Power-law (из Flink-дока)
- **Schema cache**: `functools.lru_cache` — подтверждено
- **Auto-resume DLQ**: Нет — ручной анализ
