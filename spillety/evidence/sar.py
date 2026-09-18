SAR_REQUIRED = (
    "transaction_hash",
    "blockchain",
    "timestamp",
    "sender",
    "receiver",
    "amount_crypto",
    "amount_usd",
    "risk_score",
    "narrative",
)


_TX_FIELDS = (
    "transaction_hash",
    "blockchain",
    "timestamp",
    "sender",
    "receiver",
    "amount_crypto",
    "amount_usd",
)


def from_evidence(evidence: dict, tx: dict) -> dict:
    """
    ## Map Evidence JSON + on-chain context to a draft SAR (§10.4)

    Parameters
    ----------
    evidence : dict
        Record from build_evidence (score, tier, shap_values, sensitivity).
    tx : dict
        On-chain context: transaction_hash, blockchain, timestamp, sender,
        receiver, amount_crypto, amount_usd (+optional anchors, causal_path).

    Returns
    ----------
    dict
        Draft SAR with human_review gate unapproved; never auto-filed.
    """
    missing = [f for f in _TX_FIELDS if f not in tx]
    if "score" not in evidence and "risk_score" not in evidence and "risk_score" not in tx:
        missing.append("risk_score")
    if missing:
        raise ValueError(f"missing SAR fields: {missing}")
    anchors = evidence.get("anchors", tx.get("anchors", []))
    causal_path = evidence.get("causal_path", tx.get("causal_path", []))
    return {
        "transaction_hash": tx["transaction_hash"],
        "blockchain": tx["blockchain"],
        "timestamp": tx["timestamp"],
        "sender": tx["sender"],
        "receiver": tx["receiver"],
        "amount_crypto": tx["amount_crypto"],
        "amount_usd": tx["amount_usd"],
        "risk_score": evidence.get("risk_score", evidence.get("score", tx.get("risk_score"))),
        "confidence_tier": evidence.get("tier"),
        "anchors": anchors,
        "causal_path": causal_path,
        "shap_values": evidence.get("shap_values", {}),
        "sensitivity": {
            k: evidence.get(k) for k in ("e_value", "gamma", "causal_passed") if k in evidence
        },
        "narrative": tx.get("narrative", evidence.get("narrative", "")),
        "status": "draft",
        # Auto-filing without human review is forbidden; submit_sar enforces this.
        "human_review": {"approved": False, "reviewer": None},
    }


def approve_sar(sar: dict, reviewer: str) -> dict:
    """
    ## Human-review approval gate for SAR filing (§10.4)

    Parameters
    ----------
    sar : dict
        Draft SAR from from_evidence.
    reviewer : str
        Non-empty analyst/compliance-officer id.

    Returns
    ----------
    dict
        Same SAR with the gate opened.
    """
    if not reviewer:
        raise ValueError("reviewer must be a non-empty id")
    sar["human_review"] = {"approved": True, "reviewer": reviewer}
    return sar


def submit_sar(sar: dict) -> dict:
    """
    ## File SAR; raises unless the human-review gate is open

    Parameters
    ----------
    sar : dict
        SAR with human_review flag.

    Returns
    ----------
    dict
        Same SAR with status filed.
    """
    gate = sar.get("human_review", {})
    if not gate.get("approved") or not gate.get("reviewer"):
        raise PermissionError("SAR filing requires human review: call approve_sar first")
    sar["status"] = "filed"
    return sar
