"""
Schema Registry client with functools.lru_cache for schema caching.
"""

from functools import lru_cache
import json
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

SCHEMA_REGISTRY_URL = "http://schema-registry:8081"


@lru_cache(maxsize=128)
def get_schema(subject: str) -> Optional[bytes]:
    """
    Fetch schema from Schema Registry with local LRU caching.

    On cache miss, performs HTTP GET to Schema Registry.
    Returns None if schema not found or registry unreachable.
    
    """
    try:
        url = f"{SCHEMA_REGISTRY_URL}/schemas/ids/{subject}"
        req = Request(url, headers={"Accept": "application/vnd.schemaregistry.v1+json"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("schema", "").encode("utf-8")
    except (URLError, json.JSONDecodeError, KeyError):
        return None


def get_schema_cached(subject: str) -> bytes:
    """
    Wrapper that raises on cache miss with None to signal registration needed.
    """
    schema = get_schema(subject)
    if schema is None:
        raise ValueError(f"Schema not found for subject: {subject}")
    return schema


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
