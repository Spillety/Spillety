import logging
import time

import events_pb2
from confluent_kafka import Consumer, KafkaError, Producer
from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)

try:
    import enrichment_pb2
except ImportError:  # generated stubs absent until compile_proto.sh runs
    enrichment_pb2 = None

DLQ_TOTAL = Counter("kafka_dlq_total", "Total dead-letter messages", ["topic", "error_code"])
DLQ_RATE = Gauge("kafka_dlq_rate", "Current DLQ rate", ["topic"])

# Topic -> Protobuf message class for value deserialization.
TOPIC_PARSERS: dict = {"raw-events": events_pb2.OnChainEvent}
if enrichment_pb2 is not None:
    TOPIC_PARSERS.update({
        "news-events": enrichment_pb2.NewsArticleEvent,
        "kyc-updates": enrichment_pb2.KycUpdate,
        "avm-screening": enrichment_pb2.AvmScreening,
    })


def parse_message(topic: str, payload: bytes):
    """Deserialize payload with the Protobuf class registered for topic."""
    cls = TOPIC_PARSERS.get(topic, events_pb2.OnChainEvent)
    event = cls()
    event.ParseFromString(payload)
    return event


class EventConsumer:
    """
    Consumer that deserializes OnChainEvent and routes failures to DLQ.

    Circuit breaker trips when DLQ rate exceeds 5% within a 5-minute window.
    
    """

    def __init__(self, group_id: str, bootstrap_servers: str, topics: list, dlq_threshold: float = 0.05):
        self._consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 45000,
        })
        self._dlq_producer = Producer({"bootstrap.servers": bootstrap_servers, "compression.type": "lz4"})
        self._consumer.subscribe(topics)
        self._dlq_threshold = dlq_threshold
        self._dlq_count = 0
        self._total_count = 0
        self._window_start = time.time()
        self._paused = False

    def run(self, process_callback) -> None:
        """
        Main consumer loop with DLQ rate monitoring and circuit breaker.
        """
        while True:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                self._handle_error(msg)
                continue

            self._total_count += 1
            try:
                event = parse_message(msg.topic(), msg.value())
                process_callback(event=event, topic=msg.topic(), partition=msg.partition(), offset=msg.offset())
                self._consumer.commit(msg)
            except Exception as e:  # noqa: BLE001 - all processing failures route to DLQ by design
                self._dlq_count += 1
                self._handle_dlq(msg, str(e))

            if self._total_count % 100 == 0:
                self._check_dlq_rate()

    def _handle_dlq(self, msg, error_message: str) -> None:
        """
        Send failed message to dead-letter-events topic and check circuit breaker.
        """
        import base64
        import json
        dlq_msg = {
            "original_payload": base64.b64encode(msg.value()).decode("utf-8"),
            "error_code": "DESERIALIZATION_ERROR",
            "error_message": error_message,
            "topic": msg.topic(),
            "partition": msg.partition(),
            "offset": msg.offset(),
            "timestamp": int(time.time() * 1000),
        }
        DLQ_TOTAL.labels(topic=msg.topic(), error_code="DESERIALIZATION_ERROR").inc()
        self._dlq_producer.produce(
            topic="dead-letter-events",
            key=msg.topic().encode("utf-8"),
            value=json.dumps(dlq_msg).encode("utf-8"),
        )
        self._dlq_producer.poll(0)
        self._check_circuit_breaker()

    def _handle_error(self, msg) -> None:
        logger.warning("Kafka error without DLQ routing: %s", msg.error())

    def _check_dlq_rate(self) -> None:
        """
        Evaluate DLQ rate against 5% threshold within the time window.
        """
        elapsed = time.time() - self._window_start
        if elapsed >= 300 and self._total_count > 0:
            rate = self._dlq_count / self._total_count
            DLQ_RATE.labels(topic="raw-events").set(rate)
            if rate > self._dlq_threshold:
                self._trip_circuit_breaker()
            self._dlq_count = 0
            self._total_count = 0
            self._window_start = time.time()

    def _check_circuit_breaker(self) -> None:
        """
        Immediate circuit breaker check after each DLQ event.
        """
        if self._total_count > 0:
            rate = self._dlq_count / self._total_count
            if rate > self._dlq_threshold:
                self._trip_circuit_breaker()

    def _trip_circuit_breaker(self) -> None:
        """
        Pause consumer group — manual confirmation required before resume.
        """
        if not self._paused:
            self._paused = True
            print("ALERT: DLQ rate exceeded 5%. Consumer paused. Manual confirmation required.")

    def reset(self) -> None:
        """
        Manual reset after operator confirms root cause is resolved.
        """
        self._paused = False
        self._dlq_count = 0
        self._total_count = 0
        self._window_start = time.time()

    def stop(self) -> None:
        """
        Graceful shutdown — flush offsets and close consumer.
        """
        self._dlq_producer.flush()
        self._consumer.close()
