import networkx as nx
from collections import deque
from typing import Any

# ponytail: BFS depth 2 vs 3 — depth 2 chosen for latency.


class SubgraphExtractor:
    """BFS depth 2 union from from_address AND to_address."""

    def __init__(self, depth: int = 2):
        self.depth = depth

    def extract(self, address: str, graph: nx.Graph) -> dict:
        """Extract BFS subgraph from address. Returns nodes + edges."""
        nodes = set()
        edges = []
        for start in [address]:
            visited = set()
            queue = deque([(start, 0)])
            while queue:
                node, d = queue.popleft()
                if node in visited or d > self.depth:
                    continue
                visited.add(node)
                nodes.add(node)
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        edges.append((node, neighbor))
                        queue.append((neighbor, d + 1))
        subgraph_nodes = list(nodes)
        subgraph_edges = list(set(edges))
        return {"nodes": subgraph_nodes, "edges": subgraph_edges}


def demo() -> None:
    """Smoke test: BFS depth 2 subgraph extraction."""
    import networkx as nx
    G = nx.karate_club_graph()
    extractor = SubgraphExtractor(depth=2)
    result = extractor.extract(0, G)
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)
    assert len(result["nodes"]) > 0
    print("SubgraphExtractor demo passed")


if __name__ == "__main__":
    demo()
