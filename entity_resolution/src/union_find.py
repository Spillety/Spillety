from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entity_resolution.src.exchange_filter import ExchangeFilter


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


def process_timing(
    uf: UnionFind,
    events: list[tuple[str, float]],
    window_sec: float,
    exchange_filter: "ExchangeFilter | None" = None,
) -> list[str]:
    # ponytail: unions only time-adjacent pairs; transitivity via UnionFind chains the full window cluster
    if exchange_filter is not None:
        events = [e for e in events if not exchange_filter.is_exchange(e[0])]
    ordered = sorted(events, key=lambda e: e[1])
    roots: list[str] = []
    for (prev_addr, prev_ts), (addr, ts) in pairwise(ordered):
        if ts - prev_ts <= window_sec:
            roots.append(uf.union(prev_addr, addr))
    return roots


def process_fee_pattern(
    uf: UnionFind,
    address_fees: dict[str, float],
    tolerance: float,
    exchange_filter: "ExchangeFilter | None" = None,
) -> list[str]:
    # ponytail: relative tolerance against the lower fee; adjacent-pair unioning relies on UnionFind transitivity
    if exchange_filter is not None:
        address_fees = {
            a: f for a, f in address_fees.items()
            if not exchange_filter.is_exchange(a)
        }
    ordered = sorted(address_fees.items(), key=lambda kv: kv[1])
    roots: list[str] = []
    for (prev_addr, prev_fee), (addr, fee) in pairwise(ordered):
        base = prev_fee if prev_fee > 0 else fee
        if base > 0 and abs(fee - prev_fee) / base <= tolerance or base == 0 and fee == 0:
            roots.append(uf.union(prev_addr, addr))
    return roots
