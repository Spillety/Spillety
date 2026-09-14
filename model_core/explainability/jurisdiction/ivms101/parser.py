"""Parse and validate raw IVMS101 Travel Rule payloads."""

from model_core.explainability.jurisdiction.ivms101.schema import validate_message


def parse_message(payload: dict) -> dict:
    """Normalize raw payload into canonical originator/beneficiary/transaction shape."""
    if not isinstance(payload, dict):
        raise ValueError("IVMS101 payload must be an object")
    normalized = {
        "originator": dict(payload.get("originator", {})),
        "beneficiary": dict(payload.get("beneficiary", {})),
        "transaction": dict(payload.get("transaction", {})),
    }
    errors = validate_message(normalized)
    if errors:
        raise ValueError(f"invalid IVMS101 message: {errors}")
    return normalized


def parse_lenient(payload: dict) -> tuple[dict, list[str]]:
    """Normalize payload without raising; return (message, errors)."""
    if not isinstance(payload, dict):
        return ({}, ["message must be an object"])
    normalized = {
        "originator": dict(payload.get("originator", {})),
        "beneficiary": dict(payload.get("beneficiary", {})),
        "transaction": dict(payload.get("transaction", {})),
    }
    return (normalized, validate_message(normalized))
