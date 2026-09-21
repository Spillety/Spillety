import json
import os
from unittest.mock import MagicMock, patch

import pytest

from spillety.evidence.evidence import build_evidence, evidence_hash
from spillety.evidence.gigachat import (
    _get_token,
    _validate_explanation,
    build_alert_context,
    explain_alert,
)
from spillety.evidence.merkle import merkle_proof, merkle_root, sign_root

FIXTURE_EVIDENCE = build_evidence(
    score=0.8765,
    tier="tier1",
    shap_values={
        "feat_a": 0.12345,
        "feat_b": -0.09876,
        "feat_c": 0.05555,
        "feat_d": -0.04444,
        "feat_e": 0.03333,
        "feat_f": -0.02222,
    },
    extra={"e_value": 2.3, "gamma": 1.8, "causal_passed": True},
    anchors=[
        {"wallet": "0xabc123", "source": "ofac", "distance": 0.1234},
        {"wallet": "bc1qxyz", "source": "train_illicit", "distance": 0.2345},
        {"wallet": "0xdef456", "source": "ofac", "distance": 0.3456},
    ],
    causal_path=[
        {"edge": "A->B", "effect": 0.45, "gamma": 1.2},
        {"edge": "B->C", "effect": 0.33, "gamma": 1.1},
    ],
    provenance={
        "model_version": "v1.0",
        "encoder_version": "e2",
        "hnsw_params": "M=16,ef=200",
        "calibrator": "isotonic",
        "tau": 0.5,
        "cost_ratio": 10.0,
    },
)

FIXTURE_TX = {
    "transaction_hash": "0xdeadbeef" + "0" * 56,
    "blockchain": "ETH",
    "timestamp": "2026-09-14T12:00:00Z",
    "sender": "0xabc123",
    "receiver": "0xdef456",
    "amount_crypto": 1.5,
    "amount_usd": 90000.0,
}


def test_build_alert_context_deterministic():
    ctx1 = build_alert_context(FIXTURE_EVIDENCE, FIXTURE_TX)
    ctx2 = build_alert_context(FIXTURE_EVIDENCE, FIXTURE_TX)
    assert ctx1 == ctx2
    assert "score=0.8765" in ctx1
    assert "tier=tier1" in ctx1
    assert "anchor1: wallet=0xabc123 source=ofac distance=0.1234" in ctx1
    assert "anchor2: wallet=bc1qxyz source=train_illicit distance=0.2345" in ctx1
    assert "anchor3: wallet=0xdef456 source=ofac distance=0.3456" in ctx1
    assert "causal1: edge=A->B effect=0.4500 gamma=1.2000" in ctx1
    assert "shap: feat_a=0.12345" in ctx1
    assert "provenance: model_version=v1.0" in ctx1


def test_missing_env_raises_explicit():
    with patch.dict(os.environ, {}, clear=True):
        tx = dict(FIXTURE_TX)
        tx["evidence_hash"] = evidence_hash(FIXTURE_EVIDENCE)
        with pytest.raises(RuntimeError, match="GIGACHAT_CREDENTIALS not set"):
            explain_alert(FIXTURE_EVIDENCE, tx)


def test_verify_first_blocks_on_broken_evidence():
    os.environ["GIGACHAT_CREDENTIALS"] = "dummy"
    ev = dict(FIXTURE_EVIDENCE)
    ev["score"] = 0.9999
    tx = dict(FIXTURE_TX)
    tx["evidence_hash"] = evidence_hash(FIXTURE_EVIDENCE)

    with patch("spillety.evidence.gigachat._post_chat") as mock_post:
        with pytest.raises(ValueError, match="evidence verification failed"):
            explain_alert(ev, tx)
        mock_post.assert_not_called()


def test_verify_first_blocks_on_broken_merkle():
    os.environ["GIGACHAT_CREDENTIALS"] = "dummy"
    ev = FIXTURE_EVIDENCE
    tx = dict(FIXTURE_TX)
    tx["evidence_hash"] = evidence_hash(ev)
    leaves = [b"leaf0", b"leaf1", b"leaf2"]
    root = merkle_root(leaves)
    proof = merkle_proof(leaves, 1)
    tx["merkle_proof"] = proof
    tx["merkle_root"] = root.hex()
    tx["merkle_index"] = 1

    with patch("spillety.evidence.gigachat._post_chat") as mock_post:
        with pytest.raises(ValueError, match="merkle proof verification failed"):
            explain_alert(ev, tx)
        mock_post.assert_not_called()


def test_verify_first_blocks_on_broken_root_signature():
    os.environ["GIGACHAT_CREDENTIALS"] = "dummy"
    ev = FIXTURE_EVIDENCE
    tx = dict(FIXTURE_TX)
    tx["evidence_hash"] = evidence_hash(ev)

    leaf = json.dumps(
        {
            "score": ev["score"],
            "tier": ev["tier"],
            "shap_values": ev["shap_values"],
            "e_value": ev["e_value"],
            "gamma": ev["gamma"],
            "causal_passed": ev["causal_passed"],
            "anchors": ev.get("anchors"),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    leaves = [leaf, b"leaf1"]
    root = merkle_root(leaves)
    proof = merkle_proof(leaves, 0)
    tx["merkle_proof"] = proof
    tx["merkle_root"] = root.hex()
    tx["merkle_index"] = 0
    key = b"0" * 32
    sig = sign_root(root, key)
    tx["root_signature"] = sig.hex()
    tx["root_key"] = (b"1" * 32).hex()

    with patch("spillety.evidence.gigachat._get_token", return_value="fake-token"), patch(
        "spillety.evidence.gigachat._post_chat"
    ) as mock_post:
        with pytest.raises(ValueError, match="root signature verification failed"):
            explain_alert(ev, tx)
        mock_post.assert_not_called()


def test_post_validate_rejects_foreign_address():
    os.environ["GIGACHAT_CREDENTIALS"] = "dummy"
    ev = FIXTURE_EVIDENCE
    tx = dict(FIXTURE_TX)
    tx["evidence_hash"] = evidence_hash(ev)

    mock_resp = {
        "choices": [{"message": {"content": "Адрес 0x9999999999999999999999999999999999999999 не в контексте draft."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with patch("spillety.evidence.gigachat._get_token", return_value="fake-token"), patch(
        "spillety.evidence.gigachat._post_chat", return_value=mock_resp
    ), pytest.raises(ValueError, match="post-validation failed"):
        explain_alert(ev, tx)


def test_post_validate_accepts_valid():
    os.environ["GIGACHAT_CREDENTIALS"] = "dummy"
    ev = FIXTURE_EVIDENCE
    tx = dict(FIXTURE_TX)
    tx["evidence_hash"] = evidence_hash(ev)

    context = build_alert_context(ev, tx)
    mock_resp = {
        "choices": [{"message": {"content": f"Score 0.8765, tier tier1. {context} draft."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with patch("spillety.evidence.gigachat._get_token", return_value="fake-token"), patch(
        "spillety.evidence.gigachat._post_chat", return_value=mock_resp
    ):
        result = explain_alert(ev, tx)
        assert result["status"] == "draft"
        assert result["model"] == "GigaChat-Pro"
        assert "narrative" in result
        assert "tokens" in result


def test_no_approve_sar_submit_sar_imports():
    import spillety.evidence.gigachat as giga_mod
    with open(giga_mod.__file__) as f:
        source = f.read()
    assert "approve_sar" not in source
    assert "submit_sar" not in source
    assert "from spillety.evidence.sar import" not in source


def test_validate_explanation():
    context = "score=0.8765\ntier=tier1\nanchor1: wallet=0xabc123 source=ofac distance=0.1234"
    assert _validate_explanation("Score 0.8765, wallet 0xabc123 draft.", context)
    assert not _validate_explanation("Score 0.8765, wallet 0x99999999999999999999 draft.", context)
    assert not _validate_explanation("Amount 1000 USD not in context draft.", context)


def test_token_cache():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps({
            "access_token": "test-token",
            "expires_in": 1800,
        }).encode()
        mock_urlopen.return_value = mock_resp

        token1 = _get_token("dummy-creds")
        token2 = _get_token("dummy-creds")
        assert token1 == token2 == "test-token"
        assert mock_urlopen.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])