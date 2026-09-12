from entity_resolution.src.union_find import UnionFind, process_co_spend
from entity_resolution.src.exchange_filter import ExchangeFilter
from entity_resolution.src.ml_pipeline import MLPipeline
from entity_resolution.src.anomaly_detection import OverMergingDetector
from entity_resolution.src.analyst_queue import AnalystQueue


class EntityResolutionPipeline:
    def __init__(
        self,
        exchange_filter: ExchangeFilter | None = None,
        ml_pipeline: MLPipeline | None = None,
        over_merging_detector: OverMergingDetector | None = None,
        analyst_queue: AnalystQueue | None = None,
    ) -> None:
        self.uf = UnionFind()
        self.exchange_filter = exchange_filter or ExchangeFilter()
        self.ml_pipeline = ml_pipeline
        self.anomaly_detector = over_merging_detector or OverMergingDetector()
        self.analyst_queue = analyst_queue or AnalystQueue()

    def process_transaction(
        self, input_addresses: list[str], confidence: float = 0.5
    ) -> str | None:
        root = process_co_spend(self.uf, input_addresses, self.exchange_filter)
        if root is None:
            self.analyst_queue.enqueue(
                {"size": len(input_addresses)}, confidence
            )
            return None
        cluster_size = self.uf.cluster_size[root]
        if self.anomaly_detector.check_cluster(cluster_size):
            self.analyst_queue.enqueue(
                {"cluster_root": root, "size": cluster_size}, confidence
            )
        return root

    def get_cluster_info(self, address: str) -> dict:
        root = self.uf.find(address)
        cluster = self.uf.get_cluster(address)
        return {
            "root": root,
            "size": len(cluster),
            "is_exchange": self.exchange_filter.is_exchange_cluster(root),
        }
