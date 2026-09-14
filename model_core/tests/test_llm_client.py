"""Tests for G7.5 LLM explainer client (explains, never decides)."""
import copy

import pytest

from model_core.explainability.llm.client import LLMClient
from model_core.explainability.llm.config import LLMConfig


def _alert():
    return {
        "alert_id": "a1",
        "alert_type": "scam",
        "risk_score": 0.87,
        "decision": "BLOCK",
        "explanation": {
            "causal_path": [{"edge": "0x1->mixer", "causal_effect": 0.4}],
            "regulatory_references": ["EU AMLD6 Article 3(1)"],
        },
    }


def _client(text="EU scam narrative"):
    from unittest.mock import Mock

    engine = Mock()
    engine.generate.return_value = text
    return LLMClient(engine=engine)


def test_config_defaults_deterministic():
    cfg = LLMConfig()
    assert cfg.temperature == 0.0 and "Llama" in cfg.model_name


def test_config_rejects_nonzero_temperature():
    with pytest.raises(ValueError):
        LLMClient(LLMConfig(temperature=0.7), engine=_client()._engine)


def test_requires_engine():
    with pytest.raises(RuntimeError):
        LLMClient()


def test_vllm_source_via_mock():
    client = _client()
    assert client.source == LLMClient.SOURCE_VLLM
    result = client.explain(_alert())
    assert result["source"] == "vllm"
    assert result["model"] == LLMConfig().model_name
    assert "scam" in result["text"] or "EU" in result["text"]


def test_explain_does_not_mutate_alert():
    client = _client()
    alert = _alert()
    snapshot = copy.deepcopy(alert)
    client.explain(alert)
    assert alert == snapshot


def test_orchestrator_scores_without_llm():
    import pathlib

    orch_src = pathlib.Path("model_core/inference/orchestrator.py").read_text()
    assert "llm" not in orch_src.lower()
    assert "LLM" not in orch_src
