import torch
import numpy as np
import networkx as nx
from model_core.explainability.influence_functions import InfluenceFunctions
from model_core.explainability.path_ranking import CausalPathFinder
from model_core.explainability.hnsw_lorentz import LorentzHNSW
from model_core.explainability.embedding_store import LorentzEmbeddingStore
from model_core.explainability.schema.validator import AMLValidator
from model_core.explainability.llm.llm_explainer import LLMExplainer
from model_core.explainability.llm.narrative import NarrativeGenerator
from model_core.explainability.jurisdiction.mapping import JurisdictionMapper


def test_influence_functions():
    model = _make_dummy_model()
    inf = InfluenceFunctions(model, steps=10)
    x = torch.randn(1, 10)
    result = inf.compute_counterfactual(model, x, target=0)
    assert "counterfactual" in result
    assert "distance" in result and result["distance"] >= 0
    assert "fidelity" in result and result["fidelity"] in (0.0, 1.0)


def test_causal_path_finder():
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0, causal_effect=0.9, confounder=False)
    G.add_edge("B", "C", weight=1.0, causal_effect=0.8, confounder=False)
    G.add_edge("A", "C", weight=5.0, causal_effect=0.3, confounder=True)
    G.add_edge("A", "D", weight=1.0, causal_effect=0.5, confounder=False)
    G.add_edge("D", "C", weight=1.0, causal_effect=0.7, confounder=False)
    finder = CausalPathFinder(k=3)
    paths = finder.find_top_paths(G, "A", "C", k=3)
    assert len(paths) <= 3
    for p in paths:
        assert "path" in p and "effect" in p


def test_hnsw_lorentz():
    index = LorentzHNSW(dim=128, ef_search=50)
    data = np.random.randn(100, 128).astype(np.float32)
    index.add(data)
    query = np.random.randn(128).astype(np.float32)
    results = index.search(query, k=5)
    assert len(results) == 5
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_embedding_store():
    store = LorentzEmbeddingStore(dim=128)
    vec = np.random.randn(128).astype(np.float32)
    store.add("addr_001", vec)
    retrieved = store.get("addr_001")
    assert retrieved is not None
    assert np.allclose(retrieved, vec)
    assert store.get("nonexistent") is None


def _make_v2_alert() -> dict:
    import datetime
    return {
        "alert_id": "550e8400-e29b-41d4-a716-446655440000",
        "timestamp": datetime.datetime.now().isoformat(),
        "transaction": {"tx_hash": "0xabc", "from": "0x1", "to": "0x2", "amount": 125000.0, "asset": "USDT"},
        "risk_score": 0.87,
        "decision": "BLOCK",
        "explanation": {
            "counterfactual": {"removed_edge": "0x1→0x2", "score_without_edge": 0.12, "delta": -0.75, "interpretation": "Edge is causally necessary for high risk"},
            "causal_path": [
                {"edge": "0x1→mixer_0x", "causal_effect": 0.34},
                {"edge": "mixer_0x→0x2", "causal_effect": 0.41},
            ],
            "hyperbolic_distance": {"nearest_scam_cluster": "mixer_X", "distance": 2.34, "percentile": 97},
            "hawkes_intensity": {"lambda_t": 3.42, "threshold": 2.0, "trigger": "burst_detected"},
            "regulatory_references": ["FATF Recommendation 16 (Travel Rule)", "EU AMLD6 Article 3(1)"],
        },
    }


def test_aml_validator_v2_canonical():
    validator = AMLValidator()
    assert validator.validate(_make_v2_alert()) is True
    assert validator.last_schema_version == "v2"


def test_aml_validator_rejects_invalid():
    import jsonschema
    import pytest
    validator = AMLValidator()
    bad = _make_v2_alert()
    bad["risk_score"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)


def test_llm_explainer():
    import asyncio
    from unittest.mock import AsyncMock

    backend = AsyncMock(side_effect=lambda d: f"[LLM] Narrative for {d['alert_type']}")
    explainer = LLMExplainer(backend=backend)
    data = {"alert_type": "ransomware", "amount": 50000}

    async def run():
        r1 = await explainer.generate_narrative(data)
        r2 = await explainer.generate_narrative(data)
        assert r1 == r2
        assert "ransomware" in r1

    asyncio.run(run())


def test_narrative_generator():
    gen = NarrativeGenerator()
    r1 = gen.generate("ransomware", "A->B->C", "US")
    assert "ransomware" in r1 and "US" in r1
    r2 = gen.generate("scam", "X->Y", "EU")
    assert "scam" in r2 and "EU" in r2
    r3 = gen.generate("unknown", "Z", "UK")
    assert "unknown" in r3


def test_jurisdiction_mapper():
    mapper = JurisdictionMapper()
    assert mapper.get_regulation("US") == "FinCEN (US)"
    assert mapper.get_regulation("EU") == "AMLD5 (EU)"
    assert mapper.get_regulation("XX") == JurisdictionMapper._DEFAULT_REGULATION




def _make_dummy_model():
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 6)

        def forward(self, x):
            return self.fc(x)

    return DummyModel()
