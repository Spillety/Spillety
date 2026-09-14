import pytest

from alerting.src.dedup import DedupMerger
from alerting.src.producer import AlertProducer
from alerting.src.scorer import score_alert


def _alert(alert_id="a1", score=0.9, to="0xto", cluster="mixer_X"):
    return {
        "alert_id": alert_id,
        "timestamp": "2026-09-14T00:00:00Z",
        "transaction": {"tx_hash": "0x" + alert_id, "from": "0xfrom", "to": to, "amount": 1.0, "asset": "USDT"},
        "risk_score": score,
        "decision": "REVIEW",
        "explanation": {
            "counterfactual": {"removed_edge": "e", "score_without_edge": 0.1, "delta": -0.8, "interpretation": "i"},
            "causal_path": [{"edge": "e", "causal_effect": 0.5}],
            "hyperbolic_distance": {"nearest_scam_cluster": cluster, "distance": 1.0, "percentile": 90.0},
            "hawkes_intensity": {"lambda_t": 1.0, "threshold": 2.0, "trigger": "no_burst"},
            "regulatory_references": ["FATF Recommendation 16 (Travel Rule)"],
        },
    }


def test_score_boundaries():
    assert score_alert(_alert(score=0.96))["priority"] == "P0"
    assert score_alert(_alert(score=0.95))["priority"] == "P1"
    assert score_alert(_alert(score=0.7))["priority"] == "P1"
    assert score_alert(_alert(score=0.69))["priority"] == "P2"
    assert score_alert(_alert(score=0.5))["priority"] == "P2"
    assert score_alert(_alert(score=0.49))["priority"] == "IGNORE"
    assert score_alert(_alert(score=0.96))["action"] == "BLOCK"


def test_score_invalid():
    with pytest.raises(ValueError):
        score_alert({**_alert(), "risk_score": 1.5})
    with pytest.raises(ValueError):
        score_alert({"alert_id": "x"})


def test_dedup_merge_same_entity_pattern():
    m = DedupMerger()
    first = m.add(score_alert(_alert("a1", 0.8)))
    assert first["outcome"] == "emitted"
    second = m.add(score_alert(_alert("a2", 0.9)), now=60.0)
    assert second["outcome"] == "merged"
    assert second["merged_count"] == 2
    assert second["alert"]["risk_score"] == 0.9
    assert set(second["alert"]["merged_ids"]) == {"a1", "a2"}


def test_dedup_no_merge_different_pattern():
    m = DedupMerger()
    m.add(score_alert(_alert("a1", 0.8, cluster="mixer_X")))
    res = m.add(score_alert(_alert("a2", 0.8, cluster="mixer_Y")), now=60.0)
    assert res["outcome"] == "emitted"


def test_dedup_rate_limit():
    m = DedupMerger(rate_limit_per_entity_per_hour=2)
    m.add(score_alert(_alert("a1", 0.8)))
    m.add(score_alert(_alert("a2", 0.8)), now=10.0)
    res = m.add(score_alert(_alert("a3", 0.8)), now=20.0)
    assert res["outcome"] == "dropped"
    assert res["reason"] == "rate_limited"


def test_producer_routing():
    outbox: list[tuple[str, str, bytes]] = []
    p = AlertProducer(transport=lambda t, k, v: outbox.append((t, k, v)) or True)
    assert p.publish({"outcome": "dropped", "reason": "rate_limited", "alert": score_alert(_alert())}) == []
    assert p.publish({"outcome": "emitted", "alert": score_alert(_alert("b1", 0.8))}) == ["alerts_raw"]
    assert p.publish({"outcome": "merged", "alert": score_alert(_alert("b2", 0.8))}) == ["alerts_raw", "alerts_merged"]
    assert p.publish({"outcome": "emitted", "alert": score_alert(_alert("b3", 0.99))}) == ["alerts_raw", "alerts_p0"]
    assert len(outbox) == 5
