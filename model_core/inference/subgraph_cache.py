import time
from collections import OrderedDict
from typing import Any

class SubgraphCache:
    """LRU cache 10k entries, 1h TTL for hot address embeddings."""

    def __init__(self, capacity: int = 10000, ttl_hours: float = 1.0):
        self._capacity = capacity
        self._ttl = ttl_hours * 3600
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()

    def get(self, address: str) -> dict | None:
        """Retrieve cached embedding. Returns None if expired or absent."""
        if address not in self._cache:
            return None
        embedding, ts = self._cache[address]
        if time.time() - ts > self._ttl:
            del self._cache[address]
            return None
        self._cache.move_to_end(address)
        return embedding

    def put(self, address: str, embedding: dict) -> None:
        """Store embedding with current timestamp. Evicts oldest if over capacity."""
        if address in self._cache:
            self._cache.move_to_end(address)
        self._cache[address] = (embedding, time.time())
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)
