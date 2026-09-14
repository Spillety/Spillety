import numpy as np
from typing import Optional


class LorentzEmbeddingStore:
    """Lorentz embeddings storage and retrieval.

    Stores float32 128D vectors keyed by address strings.
    Provides O(1) add and get operations.
    """

    def __init__(self, dim: int = 128):
        self._dim = dim
        self._store: dict[str, np.ndarray] = {}

    def add(self, address: str, embedding: np.ndarray) -> None:
        """Store a Lorentz embedding under the given address.

        Args:
            address: String key for the embedding.
            embedding: Float32 vector of shape (dim,) or (dim, 1).
        """
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self._dim:
            raise ValueError(f"Expected dim={self._dim}, got {emb.shape[0]}")
        self._store[address] = emb

    def get(self, address: str) -> Optional[np.ndarray]:
        """Retrieve a Lorentz embedding by address.

        Args:
            address: String key for the embedding.

        Returns:
            Float32 numpy array of shape (dim,), or None if not found.
        """
        emb = self._store.get(address)
        if emb is None:
            return None
        return emb.copy()


def demo() -> None:
    """Smoke test: verify LorentzEmbeddingStore add/get roundtrip."""
    store = LorentzEmbeddingStore(dim=128)
    vec = np.random.randn(128).astype(np.float32)
    store.add("addr_001", vec)
    retrieved = store.get("addr_001")
    assert retrieved is not None
    assert np.allclose(retrieved, vec)
    assert store.get("nonexistent") is None
    print("embedding_store demo passed")


if __name__ == "__main__":
    demo()
