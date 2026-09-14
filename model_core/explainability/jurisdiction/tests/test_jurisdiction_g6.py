import pytest

from model_core.explainability.jurisdiction.mapping import JurisdictionMapper, build_regulatory_references
from model_core.explainability.jurisdiction.ivms101 import schema as ivms_schema
from model_core.explainability.jurisdiction.ivms101 import message_builder, parser


def test_references_include_fatf_and_amld6():
    refs_eu = build_regulatory_references("EU")
    assert "EU AMLD6 Article 3(1)" in refs_eu
    assert "FATF Recommendation 16 (Travel Rule)" in refs_eu
    assert "AMLD5 (EU)" in refs_eu
    refs_us = build_regulatory_references("US")
    assert "FinCEN (US)" in refs_us
    assert "FATF Recommendation 16 (Travel Rule)" in refs_us


def test_legacy_single_regulation_preserved():
    mapper = JurisdictionMapper()
    assert mapper.get_regulation("US") == "FinCEN (US)"
    assert mapper.get_regulation("EU") == "AMLD5 (EU)"
    assert mapper.get_regulation("XX") == JurisdictionMapper._DEFAULT_REGULATION
    assert isinstance(mapper.get_references("XX"), list)


def test_ivms101_valid_message():
    msg = message_builder.build_message("Alice", "acc-1", "Bob", "acc-2", 5000.0, "USDT")
    assert ivms_schema.is_valid(msg) is True
    assert ivms_schema.travel_rule_applies(5000.0) is True
    assert ivms_schema.travel_rule_applies(10.0) is False


def test_ivms101_rejects_incomplete():
    assert ivms_schema.is_valid({"originator": {}}) is False
    with pytest.raises(ValueError):
        parser.parse_message({"originator": {"person_name": "A"}})
    with pytest.raises(ValueError):
        message_builder.build_from_alert({"transaction": {}})


def test_ivms101_from_alert_and_lenient():
    alert = {"transaction": {"from": "0x1", "to": "0x2", "amount": 125000.0, "asset": "USDT"}}
    msg = message_builder.build_from_alert(alert, originator_name="Alice", beneficiary_name="Bob")
    assert parser.parse_message(msg) == msg
    normalized, errors = parser.parse_lenient({"originator": {}})
    assert errors
    assert set(normalized) == {"originator", "beneficiary", "transaction"}
