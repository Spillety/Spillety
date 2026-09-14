"""Tests for G7.3 case orchestration."""
import pytest

from case.alert_json_builder import build_case_payload
from case.servicenow import StubServiceNowAdapter
from case.store import CaseStatus, LocalCaseStore


def _alert(score=0.87, decision="BLOCK"):
    return {
        "alert_id": "a1",
        "timestamp": "2026-09-14T00:00:00Z",
        "transaction": {"tx_hash": "0xabc", "from": "0x1", "to": "0x2", "amount": 1.0, "asset": "USDT"},
        "risk_score": score,
        "decision": decision,
        "explanation": {"causal_path": [{"edge": "0x1->0x2", "causal_effect": 0.5}]},
    }


def test_create_case_p0_auto_escalation():
    store = LocalCaseStore()
    case = store.create_case(_alert())
    assert case["priority"] == "P0"
    assert case["status"] == CaseStatus.ESCALATED


def test_create_case_low_score_stays_open():
    store = LocalCaseStore()
    case = store.create_case(_alert(score=0.1, decision="ALLOW"))
    assert case["priority"] == "P2"
    assert case["status"] == CaseStatus.OPEN


def test_status_flow_and_closed_terminal():
    store = LocalCaseStore()
    case = store.create_case(_alert(score=0.1, decision="ALLOW"))
    store.update_status(case["case_id"], CaseStatus.IN_REVIEW)
    store.update_status(case["case_id"], CaseStatus.CLOSED)
    with pytest.raises(ValueError):
        store.update_status(case["case_id"], CaseStatus.OPEN)
    with pytest.raises(ValueError):
        store.update_status(case["case_id"], "NOPE")


def test_manual_escalate():
    store = LocalCaseStore()
    case = store.create_case(_alert(score=0.1, decision="ALLOW"))
    esc = store.escalate(case["case_id"])
    assert esc["priority"] == "P0" and esc["status"] == CaseStatus.ESCALATED


def test_servicenow_stub_roundtrip():
    adapter = StubServiceNowAdapter()
    ext = adapter.push_case({"case_id": "12345678-abcd", "status": "OPEN"})
    assert adapter.fetch_status(ext) == "OPEN"
    with pytest.raises(KeyError):
        adapter.fetch_status("SNOW-unknown")


def test_builder_maps_alert_fields():
    payload = build_case_payload(_alert())
    assert payload["tx_hash"] == "0xabc"
    assert payload["evidence"]["causal_path"][0]["edge"] == "0x1->0x2"
    with pytest.raises(KeyError):
        build_case_payload({"alert_id": "x"})
