"""
Schema Registry client with functools.lru_cache for schema caching.
"""

import json
from functools import lru_cache
from urllib.request import Request, urlopen

SCHEMA_REGISTRY_URL = "http://schema-registry:8081"


@lru_cache(maxsize=128)
def get_schema(subject: str) -> bytes:
    """
    Fetch schema from Schema Registry with local LRU caching.

    Raises on transport/decode errors or when the subject has no schema.
    """
    url = f"{SCHEMA_REGISTRY_URL}/schemas/ids/{subject}"
    req = Request(url, headers={"Accept": "application/vnd.schemaregistry.v1+json"})
    with urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    schema = data.get("schema")
    if not schema:
        raise ValueError(f"Schema not found for subject: {subject}")
    return schema.encode("utf-8")


def register_schema(subject: str, schema_str: str) -> int:
    """
    Register a new schema with Schema Registry.

    Returns the schema ID assigned by the registry.

    """
    url = f"{SCHEMA_REGISTRY_URL}/subjects/{subject}/versions"
    payload = json.dumps({"schema": schema_str}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/vnd.schemaregistry.v1+json"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["id"]
