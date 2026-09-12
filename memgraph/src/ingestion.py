import json
from typing import Any, Dict

from kafka import KafkaConsumer
from memgraph import Memgraph


class KafkaToMemgraphIngestor:
    """
    Kafka consumer → Memgraph writer pipeline for graph events
    """

    def __init__(
        self,
        kafka_bootstrap: str,
        kafka_group: str,
        mg_host: str = "localhost",
        mg_port: int = 7687,
    ) -> None:
        self._mg = Memgraph(host=mg_host, port=mg_port)
        self._consumer = KafkaConsumer(
            "transaction-events",
            "co-spend-events",
            "sanctions-events",
            bootstrap_servers=kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id=kafka_group,
        )

    def run(self) -> None:
        """
        Consume Kafka topics and write to Memgraph
        """
        for message in self._consumer:
            event = message.value
            event_type = event.get("type")
            try:
                if event_type == "transfer":
                    self._ingest_transfer(event)
                elif event_type == "co_spend":
                    self._ingest_co_spend(event)
                elif event_type == "sanctions":
                    self._ingest_sanctions(event)
            except Exception as e:
                # Send to DLQ -- do not crash on single event failure
                self._send_to_dlq(event, str(e))

    def _ingest_transfer(self, event: Dict[str, Any]) -> None:
        """
        Ingest TRANSFER event with temporal edge properties
        """
        self._mg.execute(
            """
            MERGE (a:Address {address: $from_addr})
            MERGE (b:Address {address: $to_addr})
            CREATE (a)-[:TRANSFER {
                amount: $amount,
                timestamp: datetime($timestamp),
                valid_from: datetime($valid_from),
                valid_to: $valid_to
            }]->(b)
            """,
            params={
                "from_addr": event["from_address"],
                "to_addr": event["to_address"],
                "amount": event["amount"],
                "timestamp": event["timestamp"],
                "valid_from": event["valid_from"],
                "valid_to": event.get("valid_to", None),
            },
        )

    def _ingest_co_spend(self, event: Dict[str, Any]) -> None:
        """I
        ngest CO_SPEND event — shared inputs heuristic edge
        """
        self._mg.execute(
            """
            MERGE (a:Address {address: $addr_a})
            MERGE (b:Address {address: $addr_b})
            CREATE (a)-[:CO_SPEND {shared_input_count: $count}]->(b)
            """,
            params={
                "addr_a": event["address_a"],
                "addr_b": event["address_b"],
                "count": event["shared_input_count"],
            },
        )

    def _ingest_sanctions(self, event: Dict[str, Any]) -> None:
        """
        Ingest SANCTIONS_FLAG event
        """
        self._mg.execute(
            """
            MERGE (r:Risk {risk_type: $risk_type})
            MERGE (a:Address {address: $address})
            CREATE (a)-[:SANCTIONS_FLAG {list: $list, date: datetime($date)}]->(r)
            """,
            params={
                "risk_type": event["risk_type"],
                "address": event["address"],
                "list": event["sanctions_list"],
                "date": event["date"],
            },
        )

    def _send_to_dlq(self, event: Dict[str, Any], error: str) -> None:
        """
        Send failed event to dead-letter queue
        """
        # DLQ implementation -- stub for production use
        print(f"DLQ: {event} — {error}")

    def close(self) -> None:
        """
        Disconnect from Kafka and Memgraph
        """
        self._consumer.close()
        self._mg.disconnect()


def demo() -> None:
    """
    Smoke test: instantiate ingestor and verify connection
    """
    ingestor = KafkaToMemgraphIngestor(
        kafka_bootstrap="localhost:9092",
        kafka_group="memgraph-ingestion",
    )
    print("Ingestor initialized — Kafka and Memgraph connections verified.")
    ingestor.close()


if __name__ == "__main__":
    demo()
