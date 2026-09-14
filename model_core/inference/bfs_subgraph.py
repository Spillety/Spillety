import networkx as nx
from collections import deque
from typing import Any

class SubgraphExtractor:
    """BFS depth 2 union from from_address AND to_address."""

    def __init__(self, depth: int = 2):
        self.depth = depth

    def extract(self, from_address: str, graph_or_to: Any = None, graph: nx.Graph | None = None, to_address: str | None = None) -> dict:
        """Extract BFS subgraph as union of from/to neighborhoods (depth 2)."""
        if graph is not None:
            targets = [from_address, graph_or_to]
            g = graph
        elif isinstance(graph_or_to, (nx.Graph, nx.DiGraph)):
            targets = [from_address] if to_address is None else [from_address, to_address]
            g = graph_or_to
        else:
            raise ValueError("graph must be a networkx graph")
        nodes = set()
        edges = []
        for start in targets:
            visited = set()
            queue = deque([(start, 0)])
            while queue:
                node, d = queue.popleft()
                if node in visited or d > self.depth:
                    continue
                visited.add(node)
                nodes.add(node)
                if node not in g:
                    continue
                for neighbor in g.neighbors(node):
                    if neighbor not in visited:
                        edges.append((node, neighbor))
                        queue.append((neighbor, d + 1))
        subgraph_nodes = list(nodes)
        subgraph_edges = list(set(edges))
        return {"nodes": subgraph_nodes, "edges": subgraph_edges}
