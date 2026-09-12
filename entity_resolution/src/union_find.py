class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        self.cluster_size: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.cluster_size[x] = 1
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> str:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return root_x
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        self.cluster_size[root_x] += self.cluster_size[root_y]
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return root_x

    def get_cluster(self, x: str) -> set[str]:
        root = self.find(x)
        return {addr for addr in self.parent if self.find(addr) == root}

    def is_connected(self, x: str, y: str) -> bool:
        return self.find(x) == self.find(y)


def process_co_spend(
    uf: UnionFind,
    input_addresses: list[str],
    exchange_filter: "ExchangeFilter",
) -> str | None:
    if any(exchange_filter.is_exchange(addr) for addr in input_addresses):
        return None
    root = uf.find(input_addresses[0])
    for addr in input_addresses[1:]:
        root = uf.union(root, addr)
    return root
