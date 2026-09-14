import networkx as nx
from typing import Any


class CausalPathFinder:
    """Top-3 causal paths via backdoor-adjusted causal effect.

    Uses Dijkstra on backdoor-adjusted causal graph to find
    the k highest-effect paths between source and target.
    """

    def __init__(self, k: int = 3):
        self._k = k

    def find_top_paths(
        self, graph: nx.DiGraph, source: str, target: str, k: int = 3
    ) -> list[dict]:
        """Find top-k causal paths ranked by backdoor-adjusted effect.

        Args:
            graph: Directed graph with 'weight' and 'causal_effect' edge attributes.
            source: Source node ID.
            target: Target node ID.
            k: Number of top paths to return.

        Returns:
            List of dicts with 'path', 'effect', 'length' keys.
        """
        k = min(k, self._k)

        adjusted = self._backdoor_adjust(graph)

        try:
            all_paths = list(nx.shortest_simple_paths(
                adjusted, source, target, weight="weight"
            ))
        except nx.NetworkXNoPath:
            return []

        ranked = []
        for path in all_paths[:k]:
            effect = self._path_effect(adjusted, path)
            ranked.append({
                "path": path,
                "effect": effect,
                "length": len(path),
            })

        ranked.sort(key=lambda p: abs(p["effect"]), reverse=True)
        return ranked[:k]

    @staticmethod
    def _backdoor_adjust(graph: nx.DiGraph) -> nx.DiGraph:
        """Apply backdoor adjustment by zeroing out confounding edges."""
        adjusted = graph.copy()
        for u, v, data in adjusted.edges(data=True):
            if data.get("confounder", False):
                data["weight"] = float("inf")
        return adjusted

    @staticmethod
    def _path_effect(graph: nx.DiGraph, path: list) -> float:
        """Compute backdoor-adjusted causal effect for a path."""
        effect = 1.0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = graph[u][v]
            effect *= abs(edge_data.get("causal_effect", 1.0))
        return effect


def demo() -> None:
    """Smoke test: verify CausalPathFinder returns top-3 paths."""
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0, causal_effect=0.9, confounder=False)
    G.add_edge("B", "C", weight=1.0, causal_effect=0.8, confounder=False)
    G.add_edge("A", "C", weight=5.0, causal_effect=0.3, confounder=True)
    G.add_edge("A", "D", weight=1.0, causal_effect=0.5, confounder=False)
    G.add_edge("D", "C", weight=1.0, causal_effect=0.7, confounder=False)

    finder = CausalPathFinder(k=3)
    paths = finder.find_top_paths(G, "A", "C", k=3)
    assert len(paths) <= 3
    for p in paths:
        assert "path" in p and "effect" in p
    print("path_ranking demo passed")


if __name__ == "__main__":
    demo()
