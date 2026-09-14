import datetime
import uuid

import pytest

from iceberg.catalog import load_config, table_identifier
from iceberg.writer import AlertIcebergWriter, to_record


class _FakeTable:
    """Explicit in-memory IcebergTable double; no storage fallback."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def append(self, record: dict) -> None:
        self._rows.append(record)

    def contains(self, alert_id: str) -> bool:
        return any(r["alert_id"] == alert_id for r in self._rows)

    def scan(self) -> list[dict]:
        return list(self._rows)


def _alert() -> dict:
    return {
        "alert_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transaction": {"tx_hash": "0xabc", "from": "0x1", "to": "0x2", "amount": 125000.0, "asset": "USDT"},
        "risk_score": 0.87,
        "decision": "BLOCK",
        "explanation": {
            "counterfactual": {"removed_edge": "0x1→0x2", "score_without_edge": 0.12, "delta": -0.75, "interpretation": "x"},
            "causal_path": [{"edge": "0x1→0x2", "causal_effect": 0.34}],
            "hyperbolic_distance": {"nearest_scam_cluster": "m", "distance": 1.0, "percentile": 90},
            "hawkes_intensity": {"lambda_t": 3.0, "threshold": 2.0, "trigger": "burst_detected"},
            "regulatory_references": ["FATF Recommendation 16 (Travel Rule)"],
        },
    }


def test_record_partition_keys(tmp_path):
    record = to_record(_alert())
    assert record["event_date"] == datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    assert record["alert_id"]


def test_append_only_rejects_rewrite(tmp_path):
    writer = AlertIcebergWriter(_FakeTable())
    alert = _alert()
    writer.append(alert)
    with pytest.raises(FileExistsError):
        writer.append(alert)
    records = writer.list_records()
    assert len(records) == 1
    assert records[0]["alert_id"] == alert["alert_id"]


def test_catalog_config_shape():
    config = load_config()
    assert table_identifier(config) == "aml.alerts"
