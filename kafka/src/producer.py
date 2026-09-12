"""
EventProducer: serializes OnChainEvent to Protobuf and produces to Kafka.
"""

from confluent_kafka import Producer, KafkaError
import events_pb2
import schema_registry


class EventProducer:
    """
    Kafka producer with Protobuf serialization and DLQ routing on failure.
    """

    def __init__(self, bootstrap_servers: str, schema_registry_url: str):
        self._producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "retries": 2147483647,
            "retry.backoff.ms": 100,
            "max.in.flight.requests.per.connection": 5,
            "compression.type": "lz4",
        })
        self._schema_registry_url = schema_registry_url
        self._dlq_topic = "dead-letter-events"

    def produce(self, topic: str, event: event_pb2.OnChainEvent, partition_key: str) -> bool:
        """
        Serialize event to Protobuf bytes and produce to Kafka topic.

        On serialization failure, routes to DLQ instead of dropping.
        
        """
        try:
            payload = event.SerializeToString()
            key = partition_key.encode("utf-8")
            self._producer.produce(
                topic=topic, key=key, value=payload,
                on_commit=self._delivery_callback,
            )
            self._producer.poll(0)
            return True
        except Exception as e:
            self._send_to_dlq(topic, partition_key, e)
            return False

    def _send_to_dlq(self, source_topic: str, partition: int, error_message: str) -> None:
        """
        Send failed message metadata to dead-letter-events topic.
        """
        import base64, json, time
        dlq_msg = {
            "original_payload": "",
            "error_code": "SERIALIZATION_ERROR",
            "error_message": str(error_message),
            "topic": source_topic,
            "partition": partition,
            "offset": -1,
            "timestamp": int(time.time() * 1000),
        }
        self._producer.produce(
            topic=self._dlq_topic,
            key=source_topic.encode("utf-8"),
            value=json.dumps(dlq_msg).encode("utf-8"),
        )
        self._producer.poll(0)

    def _delivery_callback(self, err, msg) -> None:
        """
        Async delivery callback — logs failures, does not block.
        """
        if err is not None and err.code() != KafkaError._PARTITION_EOF:
            print(f"Delivery failed: {err}")

    def flush(self, timeout: int = 10) -> None:
        """
        Ensure all pending messages are delivered before shutdown.
        """
        self._producer.flush(timeout=timeout)

    def register_schema(self, subject: str, schema_str: str) -> int:
        """
        Register a Protobuf schema with Schema Registry. Returns schema ID.
        """
        return schema_registry.register_schema(subject, schema_str)
