from spillety.evidence.merkle import (
    merkle_root,
    ots_anchor,
    ots_anchor_async,
    ots_verify,
)


def _root_hex(n=2):
    leaves = [f"leaf-{i}".encode() for i in range(n)]
    return merkle_root(leaves).hex()


def test_ots_anchor_async_submit_returns_submitted():
    root_hex = _root_hex()
    result = ots_anchor_async(root_hex)
    assert result["status"] in ("submitted", "error")
    if result["status"] == "submitted":
        assert result["calendar"] == "https://a.pool.opentimestamps.org"
        assert result["txid"] is None
        assert result["pending"] is True
        assert "submitted_at" in result


def test_ots_verify_roundtrip_with_fixture():
    root_hex = _root_hex()
    fake_proof = bytes.fromhex(root_hex) + b"\x00" * 64
    assert ots_verify(root_hex, fake_proof) is True
    assert ots_verify(root_hex, b"") is False
    assert ots_verify(root_hex, bytes.fromhex(root_hex)[:16]) is False
    assert ots_verify("deadbeef" + "0" * 56, fake_proof) is False


def test_ots_anchor_stub_untouched():
    root_hex = _root_hex()
    receipt = ots_anchor(bytes.fromhex(root_hex))
    assert receipt["status"] == "pending"
    assert receipt["txid"] is None
    assert receipt["proof"] is None
    assert receipt["root"] == root_hex


def test_ots_anchor_async_network_error_does_not_raise():
    result = ots_anchor_async(_root_hex(), calendar_url="https://invalid.invalid")
    assert result["status"] == "error"
    assert "error" in result