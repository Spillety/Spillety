"""Tests for G7.4 SAR/STR generation."""
import pytest

from sar.field_mapper import map_alert_to_sar
from sar.fincen_xsd import build_fincen_xml, validate_fincen_xml
from sar.renderer import render_sar
from sar.review import finalize_sar


def _alert():
    return {
        "alert_id": "a1",
        "timestamp": "2026-09-14T00:00:00Z",
        "transaction": {"tx_hash": "0xabc", "from": "0x1", "to": "0x2", "amount": 125000.0, "asset": "USDT"},
        "risk_score": 0.87,
        "decision": "BLOCK",
        "explanation": {
            "counterfactual": {"interpretation": "Edge is causally necessary"},
            "regulatory_references": ["FATF Recommendation 16 (Travel Rule)"],
        },
    }


def test_mapper_all_regimes():
    for regime in ("FinCEN", "EU", "UK"):
        fields = map_alert_to_sar(_alert(), regime=regime)
        assert fields["regime"] == regime and fields["tx_hash"] == "0xabc"
    with pytest.raises(ValueError):
        map_alert_to_sar(_alert(), regime="XX")
    with pytest.raises(KeyError):
        map_alert_to_sar({})


def test_renderer_all_regimes():
    for regime in ("FinCEN", "EU", "UK"):
        text = render_sar(map_alert_to_sar(_alert(), regime=regime))
        assert "a1" in text and "0xabc" in text


def test_fincen_xml_roundtrip():
    xml_text = build_fincen_xml(map_alert_to_sar(_alert()))
    assert validate_fincen_xml(xml_text) is True


def test_fincen_xml_rejects_bad():
    with pytest.raises(ValueError):
        validate_fincen_xml("<SAR regime='FinCEN'><AlertID>x</AlertID></SAR>")
    with pytest.raises(ValueError):
        validate_fincen_xml("<NOK regime='FinCEN'/>")
    bad_score = dict(map_alert_to_sar(_alert()), risk_score=5.0)
    with pytest.raises(ValueError):
        validate_fincen_xml(build_fincen_xml(bad_score))


def test_review_gate_blocks_unreviewed():
    with pytest.raises(PermissionError):
        finalize_sar(map_alert_to_sar(_alert()))
    with pytest.raises(PermissionError):
        finalize_sar(map_alert_to_sar(_alert()), reviewed=True, reviewer="  ")
    final = finalize_sar(map_alert_to_sar(_alert()), reviewed=True, reviewer="analyst_1")
    assert final["state"] == "FINALIZED" and final["reviewer"] == "analyst_1"
