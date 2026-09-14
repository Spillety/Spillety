import time
from collections import OrderedDict
from typing import Any

# ponytail: LRU cache 10k entries, 1h TTL for hot address embeddings.


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


def demo() -> None:
    """Smoke test: LRU cache hit/put/eviction behavior."""
    cache = SubgraphCache(capacity=3, ttl_hours=1.0)
    cache.put("addr1", {"emb": [1.0, 2.0]})
    cache.put("addr2", {"emb": [3.0, 4.0]})
    result = cache.get("addr1")
    assert result is not None
    assert result["emb"] == [1.0, 2.0]
    cache.put("addr3", {"emb": [5.0, 6.0]})
    cache.put("addr4", {"emb": [7.0, 8.0]})
    assert len(cache) == 3
    assert cache.get("addr2") is None  # evicted
    print("SubgraphCache demo passed")


if __name__ == "__main__":
    demo()
