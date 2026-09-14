"""Alert producer: routes scored alerts to alerts_raw/merged/p0 topics."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

TOPIC_ALERTS_RAW = "alerts_raw"
TOPIC_ALERTS_MERGED = "alerts_merged"
TOPIC_ALERTS_P0 = "alerts_p0"


class AlertProducer:
    """Serializes scored alerts to JSON and routes them by outcome/priority.

    Transport is a required injectable callable (topic, key, payload).
    # ponytail: production transport is confluent_kafka with idempotence on.
    """

    def __init__(
        self,
        transport: Callable[[str, str, bytes], bool],
        topic_raw: str = TOPIC_ALERTS_RAW,
        topic_merged: str = TOPIC_ALERTS_MERGED,
        topic_p0: str = TOPIC_ALERTS_P0,
    ) -> None:
        self._transport = transport
        self._topics = {"raw": topic_raw, "merged": topic_merged, "p0": topic_p0}

    def publish(self, result: dict[str, Any]) -> list[str]:
        """Publish a dedup result; returns the topics written to."""
        alert = result["alert"]
        key = str(alert["transaction"]["to"])
        payload = json.dumps(alert, sort_keys=True).encode("utf-8")
        outcome = result.get("outcome")
        if outcome == "dropped":
            return []
        written = []
        self._transport(self._topics["raw"], key, payload)
        written.append(self._topics["raw"])
        if outcome == "merged":
            self._transport(self._topics["merged"], key, payload)
            written.append(self._topics["merged"])
        if alert.get("priority") == "P0":
            self._transport(self._topics["p0"], key, payload)
            written.append(self._topics["p0"])
        return written
