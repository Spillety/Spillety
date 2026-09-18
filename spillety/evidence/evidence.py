import hashlib
import hmac
import json

_TIERS = ("tier1", "tier2", "tier3", "clear")


def build_evidence(
    score: float,
    tier: str,
    shap_values: dict,
    extra: dict | None = None,
) -> dict:
    """
    ## Evidence JSON for human review (§7.8.2)

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

    Returns
    ----------
    dict
        Evidence record with score, tier, shap_values and sensitivity fields.
    """
    if tier not in _TIERS:
        raise ValueError(f"tier must be one of {_TIERS}, got {tier!r}")
    extra = extra or {}
    top = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    return {
        "score": round(float(score), 4),
        "tier": tier,
        "shap_values": {k: round(float(v), 5) for k, v in top},
        "e_value": None if extra.get("e_value") is None else float(extra["e_value"]),
        "gamma": None if extra.get("gamma") is None else float(extra["gamma"]),
        "causal_passed": bool(extra.get("causal_passed", False)),
    }


def evidence_hash(ev: dict) -> str:
    return hashlib.sha256(
        json.dumps(ev, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
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
