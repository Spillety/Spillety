"""Catalog config loader for the Iceberg audit trail."""

from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).parent / "catalog.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load catalog.yaml into dict."""
    with open(path or _DEFAULT_PATH) as f:
        return yaml.safe_load(f) or {}


def table_identifier(config: dict) -> str:
    """Build fully qualified table identifier from config."""
    table = config.get("table", {})
    return f"{table.get('namespace', 'aml')}.{table.get('name', 'alerts')}"
