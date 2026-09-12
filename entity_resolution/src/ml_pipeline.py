import numpy as np
from sklearn.cluster import DBSCAN
from networkx import Graph


class MLPipeline:
    # ponytail: inference-only nightly batch; retraining handled by separate pipeline

    def __init__(self, model_registry_path: str, embedding_dim: int = 128) -> None:
        self.model_registry_path = model_registry_path
        self.embedding_dim = embedding_dim
        self.model = None

    def generate_embeddings(self, graph: Graph) -> dict[str, np.ndarray]:
        embeddings: dict[str, np.ndarray] = {}
        for node in graph.nodes():
            embeddings[node] = np.random.rand(self.embedding_dim).astype(np.float32)
        return embeddings

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
        clusters = self.cluster_embeddings(embeddings)
        links: dict[tuple[str, str], float] = {}
        for addr_a in embeddings:
            for addr_b in embeddings:
                if addr_a < addr_b:
                    links[(addr_a, addr_b)] = self.predict_link(
                        embeddings[addr_a], embeddings[addr_b]
                    )
        return links
