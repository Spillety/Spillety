from collections.abc import Hashable
from typing import Any

import numpy as np

__all__ = ["UnionFind", "select_tau_star"]

MAX_CLUSTER_SIZE = 1_000_000


class UnionFind:
    """
    ## Incremental disjoint-set union with cluster cap (§5.6.2, §5.5.3)

    Parameters
    ----------
    max_cluster_size : int
        Refuse merges that would exceed this size.
    """

    def __init__(self, max_cluster_size: int = MAX_CLUSTER_SIZE) -> None:
        self._parent: dict[Any, Any] = {}
        self._rank: dict[Any, int] = {}
        self._size: dict[Any, int] = {}
        self.max_cluster_size = max_cluster_size

    def __len__(self) -> int:
        return len(self._parent)

    def add(self, x: Hashable) -> None:
        """
        ## Register a singleton cluster (§5.6.2 step 1)

        Parameters
        ----------
        x : Hashable
            New address id; no-op when already known.
        """
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._size[x] = 1

    def find(self, x: Hashable) -> Hashable:
        """
        ## Root of the cluster holding x (§5.6.2)

        Parameters
        ----------
        x : Hashable
            Address id, auto-registered when unknown.

        Returns
        ----------
        Hashable
            Cluster root.
        """
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression flattens both hops.
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> bool:
        """
        ## Merge clusters of a and b unless capped (§5.5.3)

        Parameters
        ----------
        a : Hashable
            First address id.
        b : Hashable
            Second address id.

        Returns
        ----------
        bool
            True when merged (or already together), False when refused by cap.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self._size[ra] + self._size[rb] > self.max_cluster_size:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True

    def size_of(self, x: Hashable) -> int:
        """
        ## Size of the cluster holding x

        Parameters
        ----------
        x : Hashable
            Address id.

        Returns
        ----------
        int
            Member count of its cluster.
        """
        return self._size[self.find(x)]

    def clusters(self) -> list[frozenset]:
        """
        ## Snapshot of all clusters, deterministically ordered

        Returns
        ----------
        list[frozenset]
            One frozenset per cluster, ordered by sorted member repr.
        """
        groups: dict[Any, set] = {}
        for x in list(self._parent):
            groups.setdefault(self.find(x), set()).add(x)
        out = [frozenset(m) for m in groups.values()]
        out.sort(key=lambda c: sorted(repr(m) for m in c))
        return out


def select_tau_star(
    y_true: np.ndarray,
    y_score: np.ndarray,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
) -> tuple[float, float]:
    """
    ## Cost-optimal operating point τ* = argmin[C_FP·FP + C_FN·FN] (§5.5.2)

    Parameters
    ----------
    y_true : np.ndarray
        Binary pair labels.
    y_score : np.ndarray
        Fusion posterior per pair.
    c_fp : float
        Cost of a false merge.
    c_fn : float
        Cost of a false split.

    Returns
    ----------
    tuple[float, float]
        (τ*, min cost); ties resolve toward the larger τ.
    """
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("y_true is empty")
    # ponytail: full PR-grid + max_cluster_size constraint added with §5.5.4.
    best_tau, best_cost = 1.0, np.inf
    for tau in sorted(set(np.unique(s)) | {1.0}, reverse=True):
        pred = (s >= tau).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        cost = float(c_fp * fp + c_fn * fn)
        if cost < best_cost:
            best_tau, best_cost = float(tau), cost
    return best_tau, best_cost
