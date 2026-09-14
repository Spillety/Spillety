"""Build a case payload from a temp.md alert JSON (G7.3)."""

REQUIRED_ALERT_FIELDS = ("alert_id", "timestamp", "transaction", "risk_score", "decision", "explanation")


def build_case_payload(alert: dict) -> dict:
    """Map alert JSON (temp.md schema) to the case-creation payload.

    Raises KeyError listing missing fields so bad alerts fail fast.
    """
    missing = [f for f in REQUIRED_ALERT_FIELDS if f not in alert]
    if missing:
        raise KeyError(f"alert missing fields: {missing}")
    tx = alert["transaction"]
    explanation = alert.get("explanation", {})
    causal_path = explanation.get("causal_path", [])
    return {
        "alert_id": alert["alert_id"],
        "timestamp": alert["timestamp"],
        "risk_score": alert["risk_score"],
        "decision": alert["decision"],
        "tx_hash": tx.get("tx_hash"),
        "amount": tx.get("amount"),
        "asset": tx.get("asset"),
        "parties": {"from": tx.get("from"), "to": tx.get("to")},
        "evidence": {
            "causal_path": causal_path,
            "counterfactual": explanation.get("counterfactual"),
            "regulatory_references": explanation.get("regulatory_references", []),
        },
    }
