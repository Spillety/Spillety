import math

import pytest

from spillety.evidence.evidence import build_evidence
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
