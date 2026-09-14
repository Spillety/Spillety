import yaml
from pathlib import Path


class JurisdictionMapper:
    _DEFAULT_REGULATION = "FinCEN (US)"
    _DEFAULT_REFERENCES = ["FinCEN (US)", "FATF Recommendation 16 (Travel Rule)"]
    _YAML_PATH = Path(__file__).parent / "mapping.yaml"

    def __init__(self, yaml_path: str | None = None):
        path = yaml_path or self._YAML_PATH
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        self._mapping: dict[str, list[str]] = {k: self._as_list(v) for k, v in raw.items()}

    @staticmethod
    def _as_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]

    def get_regulation(self, jurisdiction: str) -> str:
        refs = self._mapping.get(jurisdiction)
        if not refs:
            return self._DEFAULT_REGULATION
        return refs[0]

    def get_references(self, jurisdiction: str) -> list[str]:
        refs = self._mapping.get(jurisdiction)
        if not refs:
            return list(self._DEFAULT_REFERENCES)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for ref in refs:
            if ref not in seen:
                seen.add(ref)
                out.append(ref)
        return out


def build_regulatory_references(jurisdiction: str, yaml_path: str | None = None) -> list[str]:
    return JurisdictionMapper(yaml_path=yaml_path).get_references(jurisdiction)
