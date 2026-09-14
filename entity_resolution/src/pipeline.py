from entity_resolution.src.analyst_queue import AnalystQueue
from entity_resolution.src.anomaly_detection import OverMergingDetector
from entity_resolution.src.exchange_filter import ExchangeFilter
from entity_resolution.src.ml_pipeline import MLPipeline
from entity_resolution.src.union_find import (
    UnionFind,
    process_co_spend,
    process_fee_pattern,
    process_timing,
)


class EntityResolutionPipeline:
    def __init__(
        self,
        exchange_filter: ExchangeFilter | None = None,
        ml_pipeline: MLPipeline | None = None,
        over_merging_detector: OverMergingDetector | None = None,
        analyst_queue: AnalystQueue | None = None,
        heuristic_weight: float = 0.4,  # ponytail: mirrors merge.heuristic_weight; no runtime YAML load
        ml_weight: float = 0.6,  # ponytail: mirrors merge.ml_weight; no runtime YAML load
    ) -> None:
        self.uf = UnionFind()
        self.exchange_filter = exchange_filter or ExchangeFilter()
        self.ml_pipeline = ml_pipeline
        self.anomaly_detector = over_merging_detector or OverMergingDetector()
        self.analyst_queue = analyst_queue or AnalystQueue()
        self.heuristic_weight = heuristic_weight
        self.ml_weight = ml_weight

    @staticmethod
    def fused_confidence(
        heuristic_conf: float, ml_conf: float,
        heuristic_weight: float = 0.4, ml_weight: float = 0.6,
    ) -> float:
        total = heuristic_weight + ml_weight
        return (heuristic_weight * heuristic_conf + ml_weight * ml_conf) / total

    def _fuse(self, confidence: float, ml_score: float | None) -> float:
        if ml_score is None:
            return confidence
        return self.fused_confidence(
            confidence, ml_score, self.heuristic_weight, self.ml_weight
        )

    def process_transaction(
        self,
        input_addresses: list[str],
        confidence: float = 0.5,
        ml_score: float | None = None,
    ) -> str | None:
        root = process_co_spend(self.uf, input_addresses, self.exchange_filter)
        if root is None:
            self.analyst_queue.enqueue(
                {"size": len(input_addresses)}, self._fuse(confidence, ml_score)
            )
            return None
        self._handle_merge(root, self._fuse(confidence, ml_score))
        return root

    def process_timed_spends(
        self,
        events: list[tuple[str, float]],
        window_sec: float = 3600.0,  # ponytail: mirrors heuristics.timing_window_sec; no runtime YAML load
        confidence: float = 0.5,
        ml_score: float | None = None,
    ) -> list[str]:
        roots = process_timing(self.uf, events, window_sec, self.exchange_filter)
        for root in roots:
            self._handle_merge(root, self._fuse(confidence, ml_score))
        return roots

    def process_fee_cluster(
        self,
        address_fees: dict[str, float],
        tolerance: float = 0.05,  # ponytail: mirrors heuristics.fee_tolerance; no runtime YAML load
        confidence: float = 0.5,
        ml_score: float | None = None,
    ) -> list[str]:
        roots = process_fee_pattern(
            self.uf, address_fees, tolerance, self.exchange_filter
        )
        for root in roots:
            self._handle_merge(root, self._fuse(confidence, ml_score))
        return roots

    def _handle_merge(self, root: str, confidence: float) -> None:
        cluster_size = self.uf.cluster_size[root]
        if self.anomaly_detector.check_cluster(cluster_size):
            self.analyst_queue.enqueue(
                {"cluster_root": root, "size": cluster_size}, confidence
            )

    def get_cluster_info(self, address: str) -> dict:
        root = self.uf.find(address)
        cluster = self.uf.get_cluster(address)
        return {
            "root": root,
            "size": len(cluster),
            "is_exchange": self.exchange_filter.is_exchange_cluster(root),
        }
