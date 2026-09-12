import json
from dataclasses import dataclass
from typing import Any, Dict, List

from memgraph import Memgraph


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[Dict[str, Any]]


class MemgraphQueryService:
    """
    Python query service for Memgraph graph database.
    """

    def __init__(self, host: str = "localhost", port: int = 7687) -> None:
        self._mg = Memgraph(host=host, port=port)

    def lookup_address(self, address: str) -> QueryResult:
        """
        Look up an address by hash index (O(1) lookup).
        """
        result = self._mg.execute(
            "MATCH (a:Address {address: $addr}) RETURN a",
            params={"addr": address},
        )
        return self._to_result(result)

    def traverse_transfers(
        self, address: str, depth: int = 3, limit: int = 100
    ) -> QueryResult:
        """
        BFS traversal via TRANSFER edges from an address.
        """
        result = self._mg.execute(
            "MATCH (a:Address {address: $addr})-[:TRANSFER*1..$depth]->(b:Address) "
            "RETURN b.address AS address, b.cluster_id AS cluster_id "
            "LIMIT $limit",
            params={"addr": address, "depth": depth, "limit": limit},
        )
        return self._to_result(result)

    def temporal_query(
        self, start_date: str, end_date: str, limit: int = 1000
    ) -> QueryResult:
        """
        Find active transfers within a temporal window.
        """
        result = self._mg.execute(
            "MATCH (a:Address)-[t:TRANSFER]->(b:Address) "
            "WHERE t.valid_from <= datetime($end) "
            "AND (t.valid_to IS NULL OR t.valid_to >= datetime($start)) "
            "RETURN a.address AS from_addr, b.address AS to_addr, "
            "t.amount AS amount, t.timestamp AS timestamp "
            "ORDER BY t.timestamp DESC LIMIT $limit",
            params={"start": start_date, "end": end_date, "limit": limit},
        )
        return self._to_result(result)

    def vector_search(
        self, query_vector: List[float], limit: int = 10
    ) -> QueryResult:
        """
        Nearest neighbor search via HNSW vector index.
        """
        result = self._mg.execute(
            "MATCH (a:Address) WHERE a.hyperbolic_embedding IS NOT NULL "
            "RETURN a.address AS address, a.cluster_id AS cluster_id "
            "ORDER BY a.hyperbolic_embedding <-> $vector LIMIT $limit",
            params={"vector": query_vector, "limit": limit},
        )
        return self._to_result(result)

    def co_spend_clustering(
        self, min_shared: int = 1, limit: int = 50
    ) -> QueryResult:
        """
        Find co-spending relationships via CO_SPEND edges.
        """
        result = self._mg.execute(
            "MATCH (a:Address)-[c:CO_SPEND]->(b:Address) "
            "WHERE c.shared_input_count >= $min_shared "
            "RETURN a.address AS addr_a, b.address AS addr_b, "
            "c.shared_input_count AS shared_input_count "
            "ORDER BY shared_input_count DESC LIMIT $limit",
            params={"min_shared": min_shared, "limit": limit},
        )
        return self._to_result(result)

    def sanctions_query(
        self, sanctions_list: str, limit: int = 100
    ) -> QueryResult:
        """
        Find addresses flagged by a sanctions list.
        """
        result = self._mg.execute(
            "MATCH (a:Address)-[s:SANCTIONS_FLAG]->(r:Risk) "
            "WHERE s.list = $list "
            "RETURN a.address AS address, r.risk_type AS risk_type, s.date AS date "
            "ORDER BY s.date DESC LIMIT $limit",
            params={"list": sanctions_list, "limit": limit},
        )
        return self._to_result(result)

    def close(self) -> None:
        """
        Disconnect from Memgraph.
        """
        self._mg.disconnect()

    @staticmethod
    def _to_result(result: Any) -> QueryResult:
        return QueryResult(
            columns=[col.name for col in result.columns],
            rows=[dict(row.items()) for row in result],
        )


def demo() -> None:
    """
    Smoke test: instantiate service and run a lookup.
    """
    svc = MemgraphQueryService()
    try:
        r = svc.lookup_address("0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb")
        assert isinstance(r, QueryResult)
        print(f"Columns: {r.columns}, Rows: {len(r.rows)}")
    finally:
        svc.close()


if __name__ == "__main__":
    demo()
