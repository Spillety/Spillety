import numpy as np
import hnswlib


class LorentzHNSW:
    """HNSW index over Lorentz-normalized vectors."""

    def __init__(
        self, dim: int = 128, M: int = 32, ef_construction: int = 200,
        ef_search: int = 200,
    ):
        self._dim = dim
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=100000,
            ef_construction=ef_construction,
            M=M,
        )
        self._ef_search = ef_search
        self._index.set_ef(ef_search)
        self._count = 0

    def add(self, vectors: np.ndarray) -> None:
        normed = self._lorentz_normalize(vectors)
        ids = np.arange(self._count, self._count + len(normed), dtype=np.int32)
        self._index.add_items(normed.astype(np.float32), ids)
        self._count += len(normed)

    def search(self, query: np.ndarray, k: int = 10) -> list:
        """Search for k nearest neighbors in Lorentz space.

        Args:
            query: Float32 vector of shape (dim,).
            k: Number of nearest neighbors to return.

        Returns:
            List of (label, distance) tuples.

        """
        normed = self._lorentz_normalize(query.reshape(1, -1))
        labels, distances = self._index.knn_query(
            normed.astype(np.float32), k=k
        )
        return list(zip(labels[0], distances[0]))

    @staticmethod
    def _lorentz_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vectors / norms
