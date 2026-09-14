import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
from memgraph import Memgraph

logger = logging.getLogger(__name__)

DLQ_TOPIC = "dead-letter-events"


def _event_time_iso(event: dict[str, Any]) -> str:
    """Prefer ISO timestamp, fall back to epoch-ms event_time_ms."""
    if event.get("timestamp"):
        return str(event["timestamp"])
    return (
        datetime.fromtimestamp(event["event_time_ms"] / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    )


class KafkaToMemgraphIngestor:
    """
    Kafka consumer → Memgraph writer pipeline for temp.md ontology events.
    """

    def __init__(
        self,
        kafka_bootstrap: str,
        kafka_group: str,
        dlq_producer: KafkaProducer,
        mg_host: str = "localhost",
        mg_port: int = 7687,
    ) -> None:
        self._mg = Memgraph(host=mg_host, port=mg_port)
        self._consumer = KafkaConsumer(
            "raw-events",
            "news-events",
            "kyc-updates",
            "avm-screening",
            bootstrap_servers=kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id=kafka_group,
        )
        self._dlq = dlq_producer

    def run(self) -> None:
        """
        Consume Kafka topics and write to Memgraph.
        """
        for message in self._consumer:
            event = message.value
            try:
                if message.topic == "raw-events":
                    self._ingest_transacts(event)
                elif message.topic == "news-events":
                    self._ingest_mentioned_in(event)
                elif message.topic == "kyc-updates":
                    self._ingest_same_as(event)
                elif message.topic == "avm-screening":
                    self._ingest_matches_pattern(event)
                else:
                    logger.warning("Unknown topic %s, routing to DLQ", message.topic)
                    self._send_to_dlq(event, message.topic, "UNKNOWN_TOPIC")
            except Exception as e:  # noqa: BLE001 - all ingest failures route to DLQ by design
                self._send_to_dlq(event, message.topic, f"{type(e).__name__}: {e}")

    def _ingest_transacts(self, event: dict[str, Any]) -> None:
        """
        Ingest TRANSACTS event with temporal edge properties.
        """
        # # ponytail: endpoints always MERGEd as Wallet; mixer/exchange
        # attribution from address roles waits for AVM-enriched raw-events.
        ts = _event_time_iso(event)
        self._mg.execute(
            """
            MERGE (a:Wallet {address: $from_addr})
            MERGE (b:Wallet {address: $to_addr})
            CREATE (a)-[:TRANSACTS {
                tx_hash: $tx_hash,
                amount: $amount,
                asset: $asset,
                timestamp: datetime($ts),
                valid_from: datetime($valid_from),
                valid_to: CASE WHEN $valid_to IS NULL THEN NULL ELSE datetime($valid_to) END
            }]->(b)
            """,
            params={
                "from_addr": event["from_address"],
                "to_addr": event["to_address"],
                "tx_hash": event.get("tx_hash"),
                "amount": event["amount"],
                "asset": event.get("asset", "USDT"),
                "ts": ts,
                "valid_from": event.get("valid_from", ts),
                "valid_to": event.get("valid_to"),
            },
        )

    def _ingest_mentioned_in(self, event: dict[str, Any]) -> None:
        """
        Ingest news-events item (NewsArticleEvent shape: article_url, title,
        mentioned_addresses, lang, event_time_ms) as NewsArticle + MENTIONED_IN.
        """
        ts = _event_time_iso(event)
        self._mg.execute(
            """
            MERGE (n:NewsArticle {url: $url})
            ON CREATE SET n.title = $title, n.lang = $lang,
                n.published_at = datetime($ts)
            WITH n
            UNWIND $addresses AS addr
            MERGE (w:Wallet {address: addr})
            CREATE (w)-[:MENTIONED_IN {
                snippet: $title,
                valid_from: datetime($ts),
                valid_to: NULL
            }]->(n)
            """,
            params={
                "url": event["article_url"],
                "title": event.get("title", ""),
                "lang": event.get("lang", "en"),
                "ts": ts,
                "addresses": event.get("mentioned_addresses", []),
            },
        )

    def _ingest_same_as(self, event: dict[str, Any]) -> None:
        """
        Ingest kyc-updates item (person_id, wallet_addresses, status) as SAME_AS.
        """
        ts = _event_time_iso(event)
        self._mg.execute(
            """
            MERGE (p:Person {person_id: $person_id})
            ON MATCH SET p.kyc_status = $status
            ON CREATE SET p.kyc_status = $status
            WITH p
            UNWIND $addresses AS addr
            MERGE (w:Wallet {address: addr})
            CREATE (w)-[:SAME_AS {
                evidence: $evidence,
                valid_from: datetime($ts),
                valid_to: NULL
            }]->(p)
            """,
            params={
                "person_id": event["person_id"],
                "status": event.get("status", "unknown"),
                "evidence": f"kyc:{event.get('event_id', 'unknown')}",
                "ts": ts,
                "addresses": event.get("wallet_addresses", []),
            },
        )

    def _ingest_matches_pattern(self, event: dict[str, Any]) -> None:
        """
        Ingest avm-screening item (address, verdict, score) as MATCHES_PATTERN.
        """
        # # ponytail: verdict string used as provisional pattern_id until G7.1
        # pattern_matching.sql defines canonical pattern IDs.
        ts = _event_time_iso(event)
        self._mg.execute(
            """
            MERGE (w:Wallet {address: $address})
            MERGE (p:Pattern {pattern_id: $pattern_id})
            CREATE (w)-[:MATCHES_PATTERN {
                score: $score,
                valid_from: datetime($ts),
                valid_to: NULL
            }]->(p)
            """,
            params={
                "address": event["address"],
                "pattern_id": event["verdict"],
                "score": float(event.get("score", 0.0)),
                "ts": ts,
            },
        )

    def _send_to_dlq(self, event: dict[str, Any], topic: str, error: str) -> None:
        """
        Route failed event to dead-letter-events; DLQ write errors propagate.
        """
        envelope = {
            "original_payload": event,
            "error_code": "INGEST_ERROR",
            "error_message": error,
            "topic": topic,
            "timestamp": int(time.time() * 1000),
        }
        self._dlq.send(DLQ_TOPIC, value=envelope)
        self._dlq.flush(timeout=10)

    def close(self) -> None:
        """
        Disconnect from Kafka and Memgraph.
        """
        self._consumer.close()
        self._dlq.close()
        self._mg.disconnect()
