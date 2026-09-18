from collections.abc import Hashable, Sequence

from spillety.entity.unionfind import UnionFind

__all__ = ["cluster_cioh"]


def cluster_cioh(
    transactions: Sequence[Sequence[Hashable]],
    *,
    is_coinjoin: Sequence[bool] | None = None,
    is_exchange_hot: Sequence[bool] | None = None,
    is_taproot: Sequence[bool] | None = None,
) -> list[frozenset]:
    """
    ## Cluster addresses by co-spending inputs (§2.3.1, §5.2.1)

    Parameters
    ----------
    transactions : Sequence[Sequence[Hashable]]
        One entry per transaction, listing its input addresses.
    is_coinjoin : Sequence[bool] | None
        Per-transaction CoinJoin flag; flagged transactions are not merged.
    is_exchange_hot : Sequence[bool] | None
        Per-transaction exchange-hot-wallet flag; flagged are not merged.
    is_taproot : Sequence[bool] | None
        Per-transaction Taproot flag; flagged are not merged.

    Returns
    ----------
    list[frozenset]
        Clusters, deterministically ordered by sorted member repr.
    """
    dsu = UnionFind()
    n = len(transactions)
    flags = _as_flag_lists(n, is_coinjoin, is_exchange_hot, is_taproot)
    for tx, (cj,xh, tr) in zip(transactions, flags):
        addrs = [a for a in tx if a is not None]
        for a in addrs:
            dsu.add(a)
        if cj or xh or tr or len(addrs) < 2:
            continue
        first = addrs[0]
        for other in addrs[1:]:
            dsu.union(first, other)
    # ponytail: change-heuristic outputs merged here, add when §5.2.2 lands.
    return dsu.clusters()


def _as_flag_lists(
    n: int,
    *flag_seqs: Sequence[bool] | None,
) -> list[tuple[bool, bool, bool]]:
    out: list[tuple[bool, bool, bool]] = []
    cols: list[list[bool]] = []
    for seq in flag_seqs:
        if seq is None:
            cols.append([False] * n)
        else:
            vals = [bool(v) for v in seq]
            if len(vals) != n:
                raise ValueError(f"flag length {len(vals)} != transactions {n}")
            cols.append(vals)
    for i in range(n):
        out.append((cols[0][i], cols[1][i], cols[2][i]))
    return out
