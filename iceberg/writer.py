"""Append-only alert writer: alert JSON to an Iceberg table."""

import datetime
import json
import uuid
from typing import Any, Protocol


def _event_date(timestamp: str) -> str:
    return datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date().isoformat()


def to_record(alert: dict) -> dict:
    """Map alert JSON to archive record partitioned by event_date."""
    return {
        "alert_id": alert["alert_id"],
        "event_date": _event_date(alert["timestamp"]),
        "alert": json.dumps(alert, sort_keys=True),
    }


class IcebergTable(Protocol):
    """Minimal surface the writer needs from a live Iceberg table."""

    def append(self, record: dict[str, Any]) -> None: ...
    def contains(self, alert_id: str) -> bool: ...
    def scan(self) -> list[dict[str, Any]]: ...


class AlertIcebergWriter:
    """Append-only writer over an injected Iceberg table.

    Duplicate alert_id is refused to keep history immutable.
    # ponytail: production IcebergTable comes from catalog.yaml via pyiceberg.
    """

    def __init__(self, table: IcebergTable) -> None:
        self._table = table

    def append(self, alert: dict) -> None:
        """Persist one alert; raise FileExistsError on duplicate alert_id."""
        record = to_record(alert)
        uuid.UUID(record["alert_id"])
        if self._table.contains(record["alert_id"]):
            raise FileExistsError(f"append-only violation: {record['alert_id']} already archived")
        self._table.append(record)

    def list_records(self) -> list[dict]:
        """Read back all archived records (audit scan)."""
        return self._table.scan()
