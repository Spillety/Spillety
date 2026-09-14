"""Build IVMS101 Travel Rule messages from alert transactions."""

from model_core.explainability.jurisdiction.ivms101.schema import validate_message


def build_message(
    originator_name: str,
    originator_account: str,
    beneficiary_name: str,
    beneficiary_account: str,
    amount: float,
    asset: str,
    originator_vasp: str = "",
    beneficiary_vasp: str = "",
) -> dict:
    """Assemble IVMS101 message dict from transfer parties."""
    return {
        "originator": {
            "person_name": originator_name,
            "account_number": originator_account,
            "vasp": originator_vasp,
        },
        "beneficiary": {
            "person_name": beneficiary_name,
            "account_number": beneficiary_account,
            "vasp": beneficiary_vasp,
        },
        "transaction": {"amount": amount, "asset": asset},
    }


def build_from_alert(alert: dict, originator_name: str = "", beneficiary_name: str = "") -> dict:
    """Build IVMS101 message from alert JSON transaction block."""
    tx = alert.get("transaction", {})
    message = build_message(
        originator_name=originator_name or tx.get("from", ""),
        originator_account=tx.get("from", ""),
        beneficiary_name=beneficiary_name or tx.get("to", ""),
        beneficiary_account=tx.get("to", ""),
        amount=tx.get("amount", 0.0),
        asset=tx.get("asset", ""),
    )
    errors = validate_message(message)
    if errors:
        raise ValueError(f"invalid IVMS101 message: {errors}")
    return message
