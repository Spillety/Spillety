import json
import os
import ssl
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any

from spillety.evidence.evidence import verify_evidence
from spillety.evidence.merkle import verify_proof, verify_root

_TOKEN_CACHE: dict[str, tuple[str, float]] = {}

# SSL context for self-signed certs (testing only; production should use proper CA)
_SSL_CONTEXT = None
if os.environ.get("GIGACHAT_INSECURE_SSL") == "1":
    _SSL_CONTEXT = ssl.create_default_context()
    _SSL_CONTEXT.check_hostname = False
    _SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _get_token(credentials: str) -> str:
    now = time.time()
    if credentials in _TOKEN_CACHE:
        token, expires = _TOKEN_CACHE[credentials]
        if now < expires:
            return token

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth" # TODO: hardcoded address
    data = urllib.parse.urlencode({"scope": "GIGACHAT_API_PERS"}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("RqUID", str(uuid.uuid4()))
    req.add_header("Authorization", f"Basic {credentials}")

    with urllib.request.urlopen(req, timeout=30, context=_SSL_CONTEXT) as resp:
        payload = json.load(resp)
        token = payload["access_token"]
        expires_in = payload.get("expires_in", 1800)
        _TOKEN_CACHE[credentials] = (token, now + expires_in - 60)
        return token


def build_alert_context(evidence: dict, tx: dict) -> str:
    lines = [
        f"score={evidence.get('score'):.4f}",
        f"tier={evidence.get('tier')}",
    ]

    anchors = evidence.get("anchors", [])[:3]
    for i, a in enumerate(anchors, 1):
        lines.append(
            f"anchor{i}: wallet={a.get('wallet')} source={a.get('source')} distance={a.get('distance'):.4f}"
        )

    causal_path = evidence.get("causal_path", [])
    for i, cp in enumerate(causal_path, 1):
        lines.append(
            f"causal{i}: edge={cp.get('edge')} effect={cp.get('effect'):.4f} gamma={cp.get('gamma'):.4f}"
        )

    shap = evidence.get("shap_values", {})
    top_shap = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    for name, val in top_shap:
        lines.append(f"shap: {name}={val:.5f}")

    prov = evidence.get("provenance", {})
    prov_parts = []
    for k in ("model_version", "encoder_version", "hnsw_params", "calibrator", "tau", "cost_ratio"):
        if k in prov:
            prov_parts.append(f"{k}={prov[k]}")
    if prov_parts:
        lines.append("provenance: " + ", ".join(prov_parts))

    return "\n".join(lines)


def _post_chat(token: str, context: str) -> dict[str, Any]:
    url = "https://api.giga.chat/v1/chat/completions"
    system = (
        "Перескажи только данные из контекста, не выдумывай хэши/суммы. "
        "Укажи якоря с дистанциями. Добавь дисклеймер: draft."
    )
    payload = json.dumps({
        "model": "GigaChat-Pro",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ],
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=30, context=_SSL_CONTEXT) as resp:
        return json.load(resp)


def _validate_explanation(text: str, context: str) -> bool:
    import re
    addresses = re.findall(r"(?:0x|bc1)[a-zA-Z0-9]{20,}", text)
    amounts = re.findall(r"\d+(?:\.\d+)?\s*(?:USD|BTC|ETH|crypto)", text, re.IGNORECASE)
    hashes = re.findall(r"0x[a-fA-F0-9]{64}", text)

    for addr in addresses:
        if addr not in context:
            return False
    for amt in amounts:
        if amt not in context:
            return False
    for h in hashes:
        if h not in context:
            return False
    return True


def explain_alert(
    evidence: dict,
    tx: dict,
    credentials_env: str = "GIGACHAT_CREDENTIALS",
) -> dict[str, Any]:
    credentials = os.environ.get(credentials_env)
    if not credentials:
        raise RuntimeError(f"{credentials_env} not set")

    expected_hash = tx.get("evidence_hash")
    if expected_hash is None:
        raise ValueError("tx must contain 'evidence_hash' for verify-first")
    if not verify_evidence(evidence, expected_hash):
        raise ValueError("evidence verification failed")

    proof = tx.get("merkle_proof")
    root = tx.get("merkle_root")
    index = tx.get("merkle_index")
    if proof is not None and root is not None and index is not None:
        leaf = json.dumps(
            {
                "score": evidence["score"],
                "tier": evidence["tier"],
                "shap_values": evidence["shap_values"],
                "e_value": evidence["e_value"],
                "gamma": evidence["gamma"],
                "causal_passed": evidence["causal_passed"],
                "anchors": evidence.get("anchors"),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        if not verify_proof(leaf, index, proof, bytes.fromhex(root)):
            raise ValueError("merkle proof verification failed")

    sig = tx.get("root_signature")
    key = tx.get("root_key")
    if sig is not None and key is not None and root is not None and not verify_root(bytes.fromhex(root), bytes.fromhex(sig), bytes.fromhex(key)):
        raise ValueError("root signature verification failed")

    context = build_alert_context(evidence, tx)

    token = _get_token(credentials)
    resp = _post_chat(token, context)
    text = resp["choices"][0]["message"]["content"]

    if not _validate_explanation(text, context):
        raise ValueError("post-validation failed: explanation contains data not in context")

    return {
        "narrative": text,
        "status": "draft",
        "model": "GigaChat-Pro",
        "tokens": resp.get("usage", {}),
    }