# 06-implementation-kafka-topics

> Phase: I (Implement) | Slug: kafka-topics | Status: Complete | Date: 2026-09-11

## Implementation Summary

Created 12 files for Kafka Producer/Consumer, Protobuf schemas, Schema Registry client, DLQ handling, Prometheus alerts.

## Created Files

- `kafka/proto/events.proto` — `OnChainEvent` schema
- `kafka/src/producer.py` — `EventProducer` with idempotent producer, LZ4 compression, DLQ fallback
- `kafka/src/consumer.py` — `EventConsumer` with DLQ routing, circuit breaker >5%, Prometheus metrics
- `kafka/src/schema_registry.py` — `functools.lru_cache(maxsize=128)` for schema caching
- `kafka/src/events_pb2.py` — Compiled Protobuf
- `kafka/config/server.properties` — Kafka broker config
- `kafka/config/prometheus_alerts.yaml` — 6 alert rules
- `kafka/docker/Dockerfile` — Python 3.11-slim
- `kafka/k8s/deployment.yaml` — Deployment + HPA
- `kafka/requirements.txt` — Dependencies
- `kafka/compile_proto.sh` — Proto compilation script

## Self-Review Results
- 🔴 Critical: **0**
- 🟡 Suggestions: **0**
- AI-slops: **0**
- AC: **All ✅**

## Key Implementation Details

- **Protobuf serialization** with `OnChainEvent` message (tx_hash, from_address, to_address, amount, timestamp, chain_id)
- **Partition key:** `from_address` ensures ordering per-address
- **DLQ circuit breaker:** Pauses producer when DLQ rate >5%
- **Schema cache:** `functools.lru_cache(maxsize=128)` instead of separate module
- **Prometheus:** 6 alert rules covering DLQ, consumer lag, throughput

## Pushback: What Could Go Wrong

1. `lru_cache` does not invalidate on schema update — new field numbers may be cached
2. 5% DLQ threshold may trigger prematurely at low throughput

## Skipped Items
- `integration tests` — add when phase I — real Kafka cluster available
- `separate schema_cache.py` — add when >3 consumer groups need distributed cache
- `auto-resume DLQ` — add when operational team confirms safe automated recovery

## Open Questions
None. Ready for Phase P (Plan) or continue to Phase I completion.

## References
- Design: `TASKS/kafka-topics-design/03-design-kafka-topics.md`
- Structure: `TASKS/kafka-topics-design/04-structure-kafka-topics.md`
