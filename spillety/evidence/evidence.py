import hashlib
import hmac
import json

_TIERS = ("tier1", "tier2", "tier3", "clear")


def build_evidence(
    score: float,
    tier: str,
    shap_values: dict,
    extra: dict | None = None,
    anchors: list[dict] | None = None,
    causal_path: list[dict] | None = None,
    provenance: dict | None = None,
) -> dict:
    """
    ## Evidence JSON for human review (§7.8.2, D6)

    Parameters
    ----------
    score : float
        Calibrated P(illicit).
    tier : str
        One of tier1 / tier2 / tier3 / clear.
    shap_values : dict
        Feature name -> TreeSHAP value; top-10 by |φ| kept.
    extra : dict | None
        Sensitivity context: e_value, gamma, causal_passed.
    anchors : list[dict] | None
        Top-K retrieval anchors: [{wallet, source, distance}, ...].
        Canonical subset for Merkle/OTS signature (provable claim: proximity to sanction anchor).
    causal_path : list[dict] | None
        DAG edges with effect/gamma. Derived, NOT in Merkle/OTS signature.
    provenance : dict | None
        Build metadata: model_version, encoder_version, hnsw_params, calibrator, tau, cost_ratio.
        Derived, NOT in Merkle/OTS signature.

    Returns
    ----------
    dict
        Evidence record with score, tier, shap_values, sensitivity, anchors, causal_path, provenance.
    """
    if tier not in _TIERS:
        raise ValueError(f"tier must be one of {_TIERS}, got {tier!r}")
    extra = extra or {}
    top = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    ev = {
        "score": round(float(score), 4),
        "tier": tier,
        "shap_values": {k: round(float(v), 5) for k, v in top},
        "e_value": None if extra.get("e_value") is None else float(extra["e_value"]),
        "gamma": None if extra.get("gamma") is None else float(extra["gamma"]),
        "causal_passed": bool(extra.get("causal_passed", False)),
    }
    if anchors is not None:
        ev["anchors"] = anchors
    if causal_path is not None:
        ev["causal_path"] = causal_path
    if provenance is not None:
        ev["provenance"] = provenance
    return ev


def evidence_hash(ev: dict) -> str:
    """
    Hash of canonical evidence subset for Merkle/OTS signing (§10.3.2, D6).

    Canonical subset = core evidence fields + anchors (provable claim).
    Excludes: causal_path, provenance (derived, reproducible from code+data).
    """
    canonical = {
        "score": ev["score"],
        "tier": ev["tier"],
        "shap_values": ev["shap_values"],
        "e_value": ev["e_value"],
        "gamma": ev["gamma"],
        "causal_passed": ev["causal_passed"],
    }
    if "anchors" in ev and ev["anchors"] is not None:
        canonical["anchors"] = ev["anchors"]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verify_evidence(ev: dict, expected: str) -> bool:
    """
    ## Hash-compare evidence record against expected digest

    Parameters
    ----------
    ev : dict
        Evidence record from build_evidence.
    expected : str
        Expected sha256 hex digest.

    Returns
    ----------
    bool
        True iff recomputed hash matches.
    """
    return hmac.compare_digest(evidence_hash(ev), expected)
