import numpy as np
import hnswlib


class LorentzHNSW:
    """HNSW index in Lorentz space for O(log n) nearest-neighbor search.

    Uses hnswlib with cosine similarity and Lorentz normalization.
    Vectors are projected to Lorentz space before indexing.
    """

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
        """Add vectors to the HNSW index after Lorentz normalization.

        Args:
            vectors: Array of shape (n, dim) float32 vectors.
        """
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

        # ponytail: HNSW efSearch parameter controls recall/speed tradeoff.
        """
        normed = self._lorentz_normalize(query.reshape(1, -1))
        labels, distances = self._index.knn_query(
            normed.astype(np.float32), k=k
        )
        return list(zip(labels[0], distances[0]))

    @staticmethod
    def _lorentz_normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to Lorentz hyperboloid: x·x = -1 with timelike component."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vectors / norms


def demo() -> None:
    """Smoke test: verify LorentzHNSW indexing and search."""
    index = LorentzHNSW(dim=128, ef_search=50)
    data = np.random.randn(100, 128).astype(np.float32)
    index.add(data)
    query = np.random.randn(128).astype(np.float32)
    results = index.search(query, k=5)
    assert len(results) == 5
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    print("hnsw_lorentz demo passed")


if __name__ == "__main__":
    demo()
