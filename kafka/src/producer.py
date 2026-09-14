"""
EventProducer: serializes OnChainEvent to Protobuf and produces to Kafka.
"""

import schema_registry
from confluent_kafka import KafkaError, Producer
from google.protobuf.message import Message

# Canonical ingest topics, mirrors kafka/config/topics.yaml.
TOPIC_RAW_EVENTS = "raw-events"
TOPIC_NEWS_EVENTS = "news-events"
TOPIC_KYC_UPDATES = "kyc-updates"
TOPIC_AVM_SCREENING = "avm-screening"


class EventProducer:
    """
    Kafka producer with Protobuf serialization.
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

    def produce(self, topic: str, event: Message, partition_key: str) -> bool:
        """
        Serialize a Protobuf event and produce to Kafka topic.

        Serialization errors propagate to the caller.
        """
        payload = event.SerializeToString()
        key = partition_key.encode("utf-8")
        self._producer.produce(
            topic=topic, key=key, value=payload,
            on_commit=self._delivery_callback,
        )
        self._producer.poll(0)
        return True

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
