"""Map temp.md alert JSON to SAR filing fields (G7.4)."""

SUPPORTED_REGIMES = ("FinCEN", "EU", "UK")


def map_alert_to_sar(alert: dict, regime: str = "FinCEN") -> dict:
    """Map alert JSON to SAR fields for the given regime.

    Raises KeyError on missing alert fields, ValueError on bad regime.
    """
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"unsupported regime: {regime}")
    for field in ("alert_id", "timestamp", "transaction", "risk_score", "decision", "explanation"):
        if field not in alert:
            raise KeyError(f"alert missing field: {field}")
    tx = alert["transaction"]
    explanation = alert.get("explanation", {})
    counterfactual = explanation.get("counterfactual", {})
    return {
        "regime": regime,
        "alert_id": alert["alert_id"],
        "filed_at": alert["timestamp"],
        "amount": tx.get("amount"),
        "asset": tx.get("asset"),
        "originator": tx.get("from"),
        "beneficiary": tx.get("to"),
        "tx_hash": tx.get("tx_hash"),
        "risk_score": alert["risk_score"],
        "decision": alert["decision"],
        "suspicious_activity": _describe_activity(alert),
        "narrative_hint": str(counterfactual.get("interpretation", "")),
        "regulatory_references": list(explanation.get("regulatory_references", [])),
    }


def _describe_activity(alert: dict) -> str:
    refs = alert.get("explanation", {}).get("regulatory_references", [])
    joined = "; ".join(refs) if refs else "unusual transaction pattern"
    return f"score={alert.get('risk_score')} decision={alert.get('decision')} refs=[{joined}]"
