import json
import jsonschema
from pathlib import Path


class AMLValidator:
    _SCHEMA_V2_PATH = Path(__file__).parent / "alert_schema_v2.json"

    def __init__(self, schema_path: str | None = None):
        self.last_schema_version: str | None = None
        target = schema_path or str(self._SCHEMA_V2_PATH)
        with open(target) as f:
            self._validator = jsonschema.Draft202012Validator(json.load(f))

    def validate(self, explanation: dict) -> bool:
        self._validator.validate(explanation)
        self.last_schema_version = "v2"
        return True
