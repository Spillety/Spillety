import yaml
from pathlib import Path


class JurisdictionMapper:
    """Hardcoded jurisdiction→regulation mapping loaded from YAML.

    Falls back to EN rule if jurisdiction is not found.
    """

    _DEFAULT_REGULATION = "FinCEN (US)"
    _YAML_PATH = Path(__file__).parent / "mapping.yaml"

    def __init__(self, yaml_path: str | None = None):
        path = yaml_path or self._YAML_PATH
        with open(path) as f:
            self._mapping = yaml.safe_load(f)

    def get_regulation(self, jurisdiction: str) -> str:
        """Get regulation reference for a jurisdiction.

        Args:
            jurisdiction: Jurisdiction code (e.g., 'US', 'EU').

        Returns:
            Regulation reference string. Falls back to EN rule.
        """
        return self._mapping.get(jurisdiction, self._DEFAULT_REGULATION)


def demo() -> None:
    """Smoke test: verify JurisdictionMapper returns correct regulations."""
    mapper = JurisdictionMapper()
    assert mapper.get_regulation("US") == "FinCEN (US)"
    assert mapper.get_regulation("EU") == "AMLD5 (EU)"
    assert mapper.get_regulation("XX") == JurisdictionMapper._DEFAULT_REGULATION
    print("jurisdiction mapping demo passed")


if __name__ == "__main__":
    demo()
