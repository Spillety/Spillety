import base64
import hashlib
import hmac
import json
import math


def canonical(ev: dict) -> str:
    return json.dumps(ev, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def evidence_hash(ev: dict) -> str:
    return hashlib.sha256(canonical(ev).encode()).hexdigest()


REQUIRED_SCHEMA = {
    "required": ["alert_id", "risk_score", "anchors", "causal_path", "shap_proxy", "provenance"],
    "anchors_required": ["distance", "source"],
    "causal_required": ["edge", "effect"],
    "provenance_required": ["model_version", "date", "source"],
}


def validate_evidence(ev: dict) -> tuple[bool, str]:
    for field in REQUIRED_SCHEMA["required"]:
        if field not in ev:
            return False, f"missing field: {field}"
    for sub in REQUIRED_SCHEMA["anchors_required"]:
        if sub not in ev["anchors"]:
            return False, f"anchors missing: {sub}"
    for sub in REQUIRED_SCHEMA["causal_required"]:
        if sub not in ev["causal_path"]:
            return False, f"causal_path missing: {sub}"
    for sub in REQUIRED_SCHEMA["provenance_required"]:
        if sub not in ev["provenance"]:
            return False, f"provenance missing: {sub}"
    if not (0 <= ev["risk_score"] <= 1):
        return False, "risk_score out of [0,1]"
    if ev.get("tier") is not None and ev["tier"] not in ("low", "medium", "high"):
        return False, "tier invalid"
    return True, "ok"


def _tier_of(score: float) -> str:
    return "high" if score > 0.8 else "medium" if score > 0.5 else "low"


def build_evidence(
    alert_id: str,
    risk_score: float,
    anchors: dict,
    causal_path: dict,
    shap: dict | None = None,
    provenance: dict | None = None,
    tier: str | None = None,
) -> dict:
    if tier is None:
        tier = _tier_of(float(risk_score))
    shap_proxy = shap if shap is not None else {}
    prov = provenance if provenance is not None else {}
    ev = {
        "alert_id": alert_id,
        "risk_score": round(float(risk_score), 4),
        "tier": tier,
        "anchors": anchors,
        "causal_path": causal_path,
        "shap_proxy": {k: round(float(v), 5) for k, v in shap_proxy.items()},
        "provenance": prov,
    }
    return ev


# HMAC-SHA256 как proxy Ed25519: без зависимости cryptography/PyNaCl, детерминированный
# контракт sign(key, hash)->sig / verify(key, hash, sig) идентичен Ed25519.
# 32B HMAC расширяем до 64B конкатенацией HMAC(key,h) || HMAC(key,HMAC) — размер как у Ed25519,
# замена drop-in: поменять hmac на ed25519 без смены интерфейса.
def sign_evidence(ev: dict, key: bytes) -> dict:
    h_hex = evidence_hash(ev)
    sig1 = hmac.new(key, h_hex.encode(), hashlib.sha256).digest()
    sig2 = hmac.new(key, sig1, hashlib.sha256).digest()
    sig64 = sig1 + sig2
    return {
        "h": h_hex,
        "sig": sig64,
        "sig_b64": base64.b64encode(sig64).decode(),
        "sig_hex": sig64.hex(),
    }


def verify_evidence(ev: dict, sig, key: bytes) -> bool:
    # sig может быть bytes / hex str / b64 str / dict из sign_evidence
    if isinstance(sig, dict):
        sig = sig.get("sig", sig.get("sig_hex", sig.get("sig_b64")))
    if isinstance(sig, str):
        # try hex (128 chars) then b64
        try:
            if len(sig) == 128:
                sig = bytes.fromhex(sig)
            else:
                sig = base64.b64decode(sig)
        except Exception:
            return False
    if not isinstance(sig, (bytes, bytearray)) or len(sig) != 64:
        return False
    h_hex = evidence_hash(ev)
    sig1 = hmac.new(key, h_hex.encode(), hashlib.sha256).digest()
    sig2 = hmac.new(key, sig1, hashlib.sha256).digest()
    expected = sig1 + sig2
    return hmac.compare_digest(expected, sig)


def build_merkle(leaves_hex: list[str]) -> tuple[list[list[str]], str]:
    if not leaves_hex:
        raise ValueError("empty leaves")
    levels: list[list[str]] = [leaves_hex[:]]
    cur = leaves_hex[:]
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            left = cur[i]
            right = cur[i + 1] if i + 1 < len(cur) else left
            parent = hashlib.sha256(bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()
            nxt.append(parent)
        levels.append(nxt)
        cur = nxt
    return levels, levels[-1][0]


def merkle_root(hashes: list[str]) -> str:
    _, root = build_merkle(hashes)
    return root


# proof size = ceil(log2 N): высота бинарного дерева = число уровней минус root,
# каждый уровень добавляет один sibling-хеш, поэтому O(log N) вместо O(N).
def merkle_proof(levels: list[list[str]], leaf_idx: int) -> list[tuple[str, bool]]:
    proof: list[tuple[str, bool]] = []
    idx = leaf_idx
    for lvl in levels[:-1]:
        is_right = idx % 2 == 1
        sib_idx = idx - 1 if is_right else idx + 1
        if sib_idx < len(lvl):
            sib = lvl[sib_idx]
        else:
            sib = lvl[idx]
        sib_is_left = is_right
        proof.append((sib, sib_is_left))
        idx //= 2
    return proof


def verify_proof(leaf_hex: str, proof: list[tuple[str, bool]], root_hex: str) -> bool:
    cur = leaf_hex
    for sib, sib_is_left in proof:
        if sib_is_left:
            cur = hashlib.sha256(bytes.fromhex(sib) + bytes.fromhex(cur)).hexdigest()
        else:
            cur = hashlib.sha256(bytes.fromhex(cur) + bytes.fromhex(sib)).hexdigest()
    return hmac.compare_digest(cur, root_hex)


# alias for spec naming
verify_merkle = verify_proof


if __name__ == "__main__":
    DEMO_KEY = b"spillety-demo-key-32-bytes!!1234"

    # build 100 evidence objects (synthetic) -> 100 hashes
    leaves = []
    evidences = []
    for i in range(100):
        ev = build_evidence(
            alert_id=f"ALT-{i:08d}-{i:03d}",
            risk_score=0.55 + (i % 10) * 0.04,
            anchors={"distance": round(float(i) * 0.1, 4), "source": "edgelist", "txId": i, "anchor_txId": i + 1000},
            causal_path={"edge": f"{i}->{i+1000}", "effect": 0.5},
            shap={"feat_1": 0.1 * (i % 5)},
            provenance={"model_version": "rf100-temporal-v1", "date": "2026-09-16", "source": "elliptic_raw", "train_range": "1..30"},
        )
        evidences.append(ev)
        leaves.append(evidence_hash(ev))

    levels, root = build_merkle(leaves)
    proof0 = merkle_proof(levels, 0)
    assert len(proof0) == 7, f"proof 7 for N=100, got {len(proof0)} (ceil(log2 100)=7)"
    assert len(proof0) == math.ceil(math.log2(100))
    assert verify_proof(leaves[0], proof0, root), "proof must verify"

    # tamper 1 char -> fail
    ev0 = evidences[0]
    sig0 = sign_evidence(ev0, DEMO_KEY)
    assert verify_evidence(ev0, sig0["sig"], DEMO_KEY)
    tampered = dict(ev0)
    tampered["risk_score"] = round(float(tampered["risk_score"]) + 0.01, 4) if tampered["risk_score"] < 0.99 else round(float(tampered["risk_score"]) - 0.01, 4)
    assert not verify_evidence(tampered, sig0["sig"], DEMO_KEY), "tamper 1 char must fail"
    assert evidence_hash(ev0) != evidence_hash(tampered)
    assert not verify_proof(evidence_hash(tampered), proof0, root), "tampered leaf must fail merkle verify"

    # tamper proof sibling
    bad_proof = proof0.copy()
    bad_proof[0] = ("00" * 32, bad_proof[0][1])
    assert not verify_proof(leaves[0], bad_proof, root)

    print(f"ok: N=100 proof={len(proof0)} root={root[:12]}... tamper test fail as expected")
