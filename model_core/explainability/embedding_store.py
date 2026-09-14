import numpy as np
from typing import Optional


class LorentzEmbeddingStore:
    """Float32 vectors keyed by address."""

    def __init__(self, dim: int = 128):
        self._dim = dim
        self._store: dict[str, np.ndarray] = {}

    def add(self, address: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self._dim:
            raise ValueError(f"Expected dim={self._dim}, got {emb.shape[0]}")
        self._store[address] = emb

    def get(self, address: str) -> Optional[np.ndarray]:
        emb = self._store.get(address)
        if emb is None:
            return None
        return emb.copy()
