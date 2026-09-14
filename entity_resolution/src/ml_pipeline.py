import numpy as np
from networkx import Graph
from sklearn.cluster import DBSCAN


class MLPipeline:
    # ponytail: inference-only nightly batch; retraining handled by separate pipeline

    def __init__(self, model_registry_path: str, embedding_dim: int = 128) -> None:
        self.model_registry_path = model_registry_path
        self.embedding_dim = embedding_dim
        self.model = None

    def generate_embeddings(self, graph: Graph) -> dict[str, np.ndarray]:
        if len(graph.nodes()) == 0:
            return {}
        # Structural + temporal node features, min-max normalized across the graph.
        max_ts = max(
            (d.get("timestamp", 0.0) for _, _, d in graph.edges(data=True)),
            default=0.0,
        )
        raw: dict[str, list[float]] = {}
        for node, data in graph.nodes(data=True):
            deg = float(graph.degree(node))
            wdeg = sum(d.get("weight", 1.0) for _, _, d in graph.edges(node, data=True))
            tx_count = float(data.get("tx_count", deg))
            last_ts = float(data.get("last_ts", max_ts))
            recency = last_ts / max_ts if max_ts > 0 else 0.0
            raw[node] = [deg, wdeg, tx_count, recency]
        cols = np.array(list(raw.values()), dtype=np.float64)
        span = cols.max(axis=0) - cols.min(axis=0)
        span[span == 0.0] = 1.0
        normed = (cols - cols.min(axis=0)) / span
        # ponytail: tile the 4 normalized features to embedding_dim instead of a learned projection
        reps = int(np.ceil(self.embedding_dim / normed.shape[1]))
        tiled = np.tile(normed, (1, reps))[:, : self.embedding_dim]
        tiled = tiled / (np.linalg.norm(tiled, axis=1, keepdims=True) + 1e-8)
        return {
            node: vec.astype(np.float32)
            for node, vec in zip(raw.keys(), tiled)
        }

    def cluster_embeddings(
        self, embeddings: dict[str, np.ndarray]
    ) -> dict[str, int]:
        if not embeddings:
            return {}
        matrix = np.array(list(embeddings.values()))
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(matrix)
        return {
            addr: int(label)
            for addr, label in zip(embeddings.keys(), clustering.labels_)
        }

    def predict_link(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        similarity = float(np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
        ))
        return 1.0 / (1.0 + np.exp(-similarity))

    def run_nightly_batch(
        self, graph: Graph
    ) -> dict[tuple[str, str], float]:
        embeddings = self.generate_embeddings(graph)
        links: dict[tuple[str, str], float] = {}
        for addr_a in embeddings:
            for addr_b in embeddings:
                if addr_a < addr_b:
                    links[(addr_a, addr_b)] = self.predict_link(
                        embeddings[addr_a], embeddings[addr_b]
                    )
        return links
