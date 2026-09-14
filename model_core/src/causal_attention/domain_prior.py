import torch
import torch.nn as nn
import numpy as np


class DomainPrior:
    """Expert graph encoding, edge type constraints.

    Defines allowed edge types: TRANSFERS -> risk, MENTIONED_IN -> risk.
    # ponytail: prior strength — soft constraint weighted prior instead of hard mask
    """

    EDGE_TYPES = {
        "TRANSFERS": 0,
        "MENTIONED_IN": 1,
        "CO_SPEND": 2,
        "SANCTIONS": 3,
        "COUNTERPARTY": 4,
        "LEGITIMATE": 5,
    }

    RISK_TYPES = {0, 1}  # TRANSFERS, MENTIONED_IN are risk edges

    def __init__(self, num_edge_types: int = 6) -> None:
        self.num_edge_types = num_edge_types
        self.edge_type_to_idx = self.EDGE_TYPES
        self._build_risk_mask()

    def _build_risk_mask(self) -> None:
        """Build binary mask where risk edge types are 1.0."""
        self.risk_mask = torch.zeros(self.num_edge_types)
        for t in self.RISK_TYPES:
            self.risk_mask[t] = 1.0

    def get_edge_mask(self) -> torch.Tensor:
        """Return risk edge type mask as tensor.

        Returns:
            Tensor of shape (num_edge_types,) with 1.0 for risk types
        """
        return self.risk_mask

    def is_risk_edge(self, edge_type: int) -> bool:
        """Check if an edge type is classified as risk."""
        return edge_type in self.RISK_TYPES

    def encode_expert_graph(
        self, edges: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Encode expert graph edges into adjacency tensor.

        Args:
            edges: List of (src, dst, edge_type_str) tuples

        Returns:
            Adjacency tensor of shape (n, n, num_edge_types)
        """
        nodes = set()
        for src, dst, _ in edges:
            nodes.add(int(src))
            nodes.add(int(dst))
        n = max(nodes) + 1 if nodes else 1
        adj = torch.zeros(n, n, self.num_edge_types)
        for src, dst, etype in edges:
            idx = self.edge_type_to_idx.get(etype, -1)
            if idx >= 0:
                adj[int(src)][int(dst)][idx] = 1.0
        return adj


def demo() -> None:
    """Smoke test: DomainPrior with assert tests."""
    prior = DomainPrior(num_edge_types=6)
    mask = prior.get_edge_mask()
    assert mask.shape == (6,), f"Mask shape mismatch: {mask.shape}"
    assert mask[0] == 1.0 and mask[1] == 1.0, "TRANSFERS and MENTIONED_IN must be risk"
    assert mask[2] == 0.0, "CO_SPEND must not be risk"
    assert prior.is_risk_edge(0) is True, "TRANSFERS is risk"
    assert prior.is_risk_edge(4) is False, "COUNTERPARTY is not risk"
    # Encode simple expert graph
    edges = [("0", "1", "TRANSFERS"), ("1", "2", "CO_SPEND")]
    adj = prior.encode_expert_graph(edges)
    assert adj.shape[2] == 6, "Last dim must be num_edge_types"
    assert adj[0][1][0] == 1.0, "TRANSFERS edge must be encoded"
    print("DomainPrior demo passed")


if __name__ == "__main__":
    demo()
