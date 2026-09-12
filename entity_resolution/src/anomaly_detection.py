class OverMergingDetector:
    # ponytail: fixed 1000 threshold; adaptive percentile when labeled data available

    def __init__(self, cluster_size_max: int = 1000) -> None:
        self.cluster_size_max = cluster_size_max

    def check_cluster(self, cluster_size: int) -> bool:
        return cluster_size > self.cluster_size_max

    def flag_cluster(self, cluster_id: str, size: int) -> dict:
        is_anomalous = self.check_cluster(size)
        return {
            "cluster_id": cluster_id,
            "size": size,
            "anomalous": is_anomalous,
        }
