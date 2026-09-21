import math

import pytest

from spillety.evidence.evidence import build_evidence, evidence_hash, verify_evidence
from spillety.evidence.merkle import (
    _HAS_ED25519,
    merkle_proof,
    merkle_root,
    ots_anchor,
    sign_root,
    verify_proof,
    verify_root,
)
from spillety.evidence.sar import SAR_REQUIRED, approve_sar, from_evidence, submit_sar


def _leaves(n):
    return [f"alert-{i}".encode() for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 16, 100])
def test_proof_size_is_ceil_log2n(n):
    leaves = _leaves(n)
    root = merkle_root(leaves)
    expect = 0 if n == 1 else math.ceil(math.log2(n))
    for i in (0, n // 2, n - 1):
        proof = merkle_proof(leaves, i)
        assert len(proof) == expect
        assert verify_proof(leaves[i], i, proof, root)


def test_tamper_fails_verify():
    leaves = _leaves(8)
    root = merkle_root(leaves)
    proof = merkle_proof(leaves, 0)
    assert not verify_proof(b"alert-0-tampered", 0, proof, root)
    flipped = "0" if proof[0][0] != "0" else "1"
    assert not verify_proof(leaves[0], 0, [flipped + proof[0][1:], *proof[1:]], root)
    assert not verify_proof(leaves[0], 0, proof, merkle_root(_leaves(9)))
    assert not verify_proof(leaves[0], 1, proof, root)


def test_root_signature_roundtrip_and_tamper():
    root = merkle_root(_leaves(4))
    if _HAS_ED25519:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        sk = Ed25519PrivateKey.generate()
        key = sk.private_bytes_raw()
        pub = sk.public_key().public_bytes_raw()
        assert verify_root(root, sign_root(root, key), pub)
        assert not verify_root(b"\x00" * 32, sign_root(root, key), pub)
    else:
        key = b"0" * 32
        assert verify_root(root, sign_root(root, key), key)
        assert not verify_root(b"\x00" * 32, sign_root(root, key), key)
        assert not verify_root(root, sign_root(root, key), b"1" * 32)


def test_ots_anchor_is_pending_stub():
    receipt = ots_anchor(merkle_root(_leaves(2)))
    assert receipt["status"] == "pending" and receipt["txid"] is None


def _tx(**over):
    tx = {
        "transaction_hash": "0xabc",
        "blockchain": "BTC",
        "timestamp": "2026-09-14T12:00:00Z",
        "sender": "bc1q...",
        "receiver": "bc1p...",
        "amount_crypto": 1.5,
        "amount_usd": 90000.0,
    }
    tx.update(over)
    return tx


def test_sar_has_required_fields():
    ev = build_evidence(0.87, "tier1", {"f": 0.5}, {"e_value": 2.3})
    sar = from_evidence(ev, _tx())
    for field in SAR_REQUIRED:
        assert field in sar
    assert sar["risk_score"] == 0.87 and sar["status"] == "draft"


def test_sar_missing_field_raises():
    ev = build_evidence(0.87, "tier1", {"f": 0.5})
    tx = _tx()
    del tx["sender"]
    with pytest.raises(ValueError, match="sender"):
        from_evidence(ev, tx)


def test_gate_blocks_auto_submit():
    sar = from_evidence(build_evidence(0.9, "tier1", {"f": 0.5}), _tx())
    with pytest.raises(PermissionError):
        submit_sar(sar)
    with pytest.raises(ValueError, match="reviewer"):
        approve_sar(sar, "")
    approve_sar(sar, "analyst_07")
    assert submit_sar(sar)["status"] == "filed"


def test_build_evidence_backward_compat():
    """Old poor call without anchors/causal_path/provenance still works."""
    ev = build_evidence(0.87, "tier1", {"f1": 0.5, "f2": -0.3}, {"e_value": 2.3, "gamma": 1.8})
    assert ev["score"] == 0.87
    assert ev["tier"] == "tier1"
    assert "anchors" not in ev
    assert "causal_path" not in ev
    assert "provenance" not in ev
    assert ev["e_value"] == 2.3
    assert ev["gamma"] == 1.8


def test_build_evidence_rich_call():
    """Rich call contains anchors, causal_path, provenance."""
    anchors = [{"wallet": "0xabc", "source": "ofac", "distance": 0.12}]
    causal_path = [{"effect": 0.45, "gamma": 1.2}]
    provenance = {"model_version": "v1.0", "calibrator": "isotonic", "tau": 0.5}
    ev = build_evidence(
        0.87,
        "tier1",
        {"f1": 0.5},
        {"e_value": 2.3},
        anchors=anchors,
        causal_path=causal_path,
        provenance=provenance,
    )
    assert ev["anchors"] == anchors
    assert ev["causal_path"] == causal_path
    assert ev["provenance"] == provenance


def test_verify_evidence_anchor_tamper_fails():
    """verify_evidence fails when anchor is swapped."""
    anchors = [{"wallet": "0xabc", "source": "ofac", "distance": 0.12}]
    ev = build_evidence(0.87, "tier1", {"f1": 0.5}, {}, anchors=anchors)
    digest = evidence_hash(ev)

    assert verify_evidence(ev, digest)

    tampered = dict(ev)
    tampered["anchors"] = [{"wallet": "0xdef", "source": "ofac", "distance": 0.12}]
    assert not verify_evidence(tampered, digest)


def test_verify_evidence_provenance_change_passes():
    """verify_evidence does NOT fail when provenance changes (not in canonical subset)."""
    anchors = [{"wallet": "0xabc", "source": "ofac", "distance": 0.12}]
    provenance = {"model_version": "v1.0", "calibrator": "isotonic"}
    ev = build_evidence(0.87, "tier1", {"f1": 0.5}, {}, anchors=anchors, provenance=provenance)
    digest = evidence_hash(ev)

    assert verify_evidence(ev, digest)

    tampered = dict(ev)
    tampered["provenance"] = {"model_version": "v2.0", "calibrator": "beta"}
    assert verify_evidence(tampered, digest)


def test_verify_evidence_causal_path_change_passes():
    """verify_evidence does NOT fail when causal_path changes (not in canonical subset)."""
    anchors = [{"wallet": "0xabc", "source": "ofac", "distance": 0.12}]
    causal_path = [{"effect": 0.45, "gamma": 1.2}]
    ev = build_evidence(0.87, "tier1", {"f1": 0.5}, {}, anchors=anchors, causal_path=causal_path)
    digest = evidence_hash(ev)

    assert verify_evidence(ev, digest)

    tampered = dict(ev)
    tampered["causal_path"] = [{"effect": 0.99, "gamma": 2.5}]
    assert verify_evidence(tampered, digest)


def test_canonical_subset_is_anchors():
    """Canonical subset for Merkle/OTS signature is anchors only."""
    anchors = [{"wallet": "0xabc", "source": "ofac", "distance": 0.12}]
    causal_path = [{"effect": 0.45, "gamma": 1.2}]
    provenance = {"model_version": "v1.0"}

    ev1 = build_evidence(0.87, "tier1", {"f1": 0.5}, {}, anchors=anchors, causal_path=causal_path, provenance=provenance)
    ev2 = build_evidence(0.87, "tier1", {"f1": 0.5}, {}, anchors=anchors, causal_path=[{"effect": 0.99}], provenance={"model_version": "v2.0"})
    ev3 = build_evidence(0.87, "tier1", {"f1": 0.5}, {}, anchors=[{"wallet": "0xdef"}], causal_path=causal_path, provenance=provenance)

    # Same anchors -> same hash (canonical subset)
    assert evidence_hash(ev1) == evidence_hash(ev2)

    # Different anchors -> different hash
    assert evidence_hash(ev1) != evidence_hash(ev3)
