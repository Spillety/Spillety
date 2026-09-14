"""IVMS101 message schema: required fields and validation.

Simplified subset of the open IVMS101 standard, sufficient for
Travel Rule originator/beneficiary data exchange checks.
"""

REQUIRED_ORIGINATOR_FIELDS = ("person_name", "account_number")
REQUIRED_BENEFICIARY_FIELDS = ("person_name", "account_number")
REQUIRED_TX_FIELDS = ("amount", "asset")

TRAVEL_RULE_THRESHOLD = 3000.0


def travel_rule_applies(amount: float, threshold: float = TRAVEL_RULE_THRESHOLD) -> bool:
    """Check whether amount triggers Travel Rule data exchange."""
    return amount >= threshold


def validate_message(message: dict) -> list[str]:
    """Validate IVMS101 message dict, return list of error strings."""
    errors: list[str] = []
    if not isinstance(message, dict):
        return ["message must be an object"]
    for section, required in (
        ("originator", REQUIRED_ORIGINATOR_FIELDS),
        ("beneficiary", REQUIRED_BENEFICIARY_FIELDS),
        ("transaction", REQUIRED_TX_FIELDS),
    ):
        node = message.get(section)
        if not isinstance(node, dict):
            errors.append(f"missing section: {section}")
            continue
        for field in required:
            if not node.get(field):
                errors.append(f"missing field: {section}.{field}")
    tx = message.get("transaction")
    if isinstance(tx, dict) and "amount" in tx:
        try:
            if float(tx["amount"]) < 0:
                errors.append("transaction.amount must be >= 0")
        except (TypeError, ValueError):
            errors.append("transaction.amount must be numeric")
    return errors


def is_valid(message: dict) -> bool:
    """Return True when message passes IVMS101 validation."""
    return not validate_message(message)
