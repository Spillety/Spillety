from dataclasses import dataclass
from typing import Any

from memgraph import Memgraph


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]


class MemgraphQueryService:
    """
    Python query service for temp.md ontology in Memgraph.
    """

    def __init__(self, host: str = "localhost", port: int = 7687) -> None:
        self._mg = Memgraph(host=host, port=port)

    def lookup_wallet(self, address: str) -> QueryResult:
        """
        Look up a wallet by point-lookup index (O(1) lookup).
        """
        result = self._mg.execute(
            "MATCH (w:Wallet {address: $addr}) RETURN w",
            params={"addr": address},
        )
        return self._to_result(result)

    def traverse_transacts(
        self, address: str, depth: int = 2, limit: int = 100
    ) -> QueryResult:
        """
        Local subgraph via TRANSACTS from a wallet (lazy inference input).
        """
        result = self._mg.execute(
            "MATCH (w:Wallet {address: $addr})-[:TRANSACTS*1..$depth]-(n) "
            "RETURN labels(n) AS labels, n.address AS address, "
            "n.cluster_id AS cluster_id LIMIT $limit",
            params={"addr": address, "depth": depth, "limit": limit},
        )
        return self._to_result(result)

    def temporal_transacts(
        self, start_date: str, end_date: str, limit: int = 1000
    ) -> QueryResult:
        """
        Find active TRANSACTS within a temporal window.
        """
        result = self._mg.execute(
            "MATCH (a)-[t:TRANSACTS]->(b) "
            "WHERE t.valid_from <= datetime($end) "
            "AND (t.valid_to IS NULL OR t.valid_to >= datetime($start)) "
            "RETURN a.address AS from_addr, b.address AS to_addr, "
            "t.tx_hash AS tx_hash, t.amount AS amount, t.timestamp AS timestamp "
            "ORDER BY t.timestamp DESC LIMIT $limit",
            params={"start": start_date, "end": end_date, "limit": limit},
        )
        return self._to_result(result)

    def nearest_scam_cluster(
        self, query_vector: list[float], limit: int = 10
    ) -> QueryResult:
        """
        Nearest neighbor search over HNSW vector indexes on Wallet/Mixer/Exchange
        hyperbolic embeddings; feeds explanation.hyperbolic_distance.
        """
        result = self._mg.execute(
            "MATCH (w:Wallet) WHERE w.hyperbolic_embedding IS NOT NULL "
            "RETURN w.address AS node_id, 'Wallet' AS kind, "
            "w.hyperbolic_embedding <-> $vector AS distance "
            "UNION ALL "
            "MATCH (m:Mixer) WHERE m.hyperbolic_embedding IS NOT NULL "
            "RETURN m.mixer_id AS node_id, 'Mixer' AS kind, "
            "m.hyperbolic_embedding <-> $vector AS distance "
            "UNION ALL "
            "MATCH (e:Exchange) WHERE e.hyperbolic_embedding IS NOT NULL "
            "RETURN e.exchange_id AS node_id, 'Exchange' AS kind, "
            "e.hyperbolic_embedding <-> $vector AS distance "
            "ORDER BY distance LIMIT $limit",
            params={"vector": query_vector, "limit": limit},
        )
        return self._to_result(result)

    def wallets_of_person(self, person_id: str, limit: int = 50) -> QueryResult:
        """
        Find wallets linked to a person via SAME_AS edges.
        """
        result = self._mg.execute(
            "MATCH (w:Wallet)-[s:SAME_AS]->(p:Person {person_id: $pid}) "
            "RETURN w.address AS address, s.evidence AS evidence, "
            "s.valid_from AS valid_from LIMIT $limit",
            params={"pid": person_id, "limit": limit},
        )
        return self._to_result(result)

    def mentions_of_wallet(self, address: str, limit: int = 100) -> QueryResult:
        """
        Find news articles mentioning a wallet via MENTIONED_IN edges.
        """
        result = self._mg.execute(
            "MATCH (w:Wallet {address: $addr})-[m:MENTIONED_IN]->(n:NewsArticle) "
            "RETURN n.url AS url, n.title AS title, m.snippet AS snippet, "
            "m.valid_from AS valid_from "
            "ORDER BY m.valid_from DESC LIMIT $limit",
            params={"addr": address, "limit": limit},
        )
        return self._to_result(result)

    def pattern_matches(
        self, min_score: float = 0.5, limit: int = 50
    ) -> QueryResult:
        """
        Find wallets matching risk patterns above score threshold.
        """
        result = self._mg.execute(
            "MATCH (w:Wallet)-[m:MATCHES_PATTERN]->(p:Pattern) "
            "WHERE m.score >= $min_score "
            "RETURN w.address AS address, p.pattern_id AS pattern_id, "
            "m.score AS score "
            "ORDER BY score DESC LIMIT $limit",
            params={"min_score": min_score, "limit": limit},
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
