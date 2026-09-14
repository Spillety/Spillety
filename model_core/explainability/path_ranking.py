import networkx as nx
from typing import Any


class CausalPathFinder:
    """Top-k paths by backdoor-adjusted effect via Dijkstra."""

    def __init__(self, k: int = 3):
        self._k = k

    def find_top_paths(
        self, graph: nx.DiGraph, source: str, target: str, k: int = 3
    ) -> list[dict]:
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
        adjusted = graph.copy()
        for u, v, data in adjusted.edges(data=True):
            if data.get("confounder", False):
                data["weight"] = float("inf")
        return adjusted

    @staticmethod
    def _path_effect(graph: nx.DiGraph, path: list) -> float:
        effect = 1.0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = graph[u][v]
            effect *= abs(edge_data["causal_effect"])
        return effect
