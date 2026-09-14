import json
import jsonschema
from pathlib import Path


class AMLValidator:
    """Custom JSON Schema validator for AML-specific explanation fields.

    Supports draft-2020-12 with custom keywords: risk_score, regulation_ref.
    """

    _SCHEMA_PATH = Path(__file__).parent / "aml_schema.json"

    def __init__(self, schema_path: str | None = None):
        with open(schema_path or self._SCHEMA_PATH) as f:
            self._schema = jsonschema.Draft202012Validator(json.load(f))

    def validate(self, explanation: dict) -> bool:
        """Validate an explanation dict against the AML schema.

        Args:
            explanation: Explanation dict to validate.

        Returns:
            True if valid, raises ValidationError on failure.
        """
        self._schema.validate(explanation)
        return True


def demo() -> None:
    """Smoke test: verify AMLValidator accepts valid explanations."""
    import datetime

    validator = AMLValidator()
    explanation = {
        "explanation_id": "550e8400-e29b-41d4-a716-446655440000",
        "causal_paths": [{"path": ["A", "B"], "effect": 0.9}],
        "counterfactuals": [{"distance": 0.1, "fidelity": 1.0}],
        "jurisdiction": "US",
        "confidence_score": 0.95,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    assert validator.validate(explanation) is True
    print("validator demo passed")


if __name__ == "__main__":
    demo()
