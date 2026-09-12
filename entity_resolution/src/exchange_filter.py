class ExchangeFilter:
    # ponytail: exchange detection heuristic, add when exchange coverage < 95%

    def __init__(self) -> None:
        self._exchange_addresses: set[str] = set()
        self._is_exchange_cluster: dict[str, bool] = {}

    def load_from_labels(self, labeled_addresses: set[str]) -> None:
        self._exchange_addresses = labeled_addresses
        for addr in labeled_addresses:
            self._is_exchange_cluster[addr] = True

    def is_exchange(self, address: str) -> bool:
        return address in self._exchange_addresses

    def is_exchange_cluster(self, cluster_root: str) -> bool:
        return self._is_exchange_cluster.get(cluster_root, False)

    def mark_cluster(self, cluster_root: str) -> None:
        self._exchange_addresses.add(cluster_root)
        self._is_exchange_cluster[cluster_root] = True
