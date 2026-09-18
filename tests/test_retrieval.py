import importlib.util

import numpy as np
import pytest

from spillety.retrieval.hnsw import (
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_EF_SEARCH,
    DEFAULT_M,
    brute_query,
    build_index,
    query_index,
    recall_at_k,
)
from spillety.retrieval.pq import (
    PQ_K,
    PQ_M,
    compression_ratio,
    pq_bytes,
    raw_bytes,
    should_use_pq,
)

needs_hnswlib = pytest.mark.skipif(
    importlib.util.find_spec("hnswlib") is None, reason="hnsw path needs spillety[hnsw]"
)


def test_brute_query_exact_against_naive():
    rng = np.random.default_rng(72)
    anchors = rng.standard_normal((40, 8))
    queries = rng.standard_normal((10, 8))
    dist, idx = brute_query(anchors, queries, k=5)
    naive = np.sqrt(((queries[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2))
    assert dist.shape == (10, 5) and idx.shape == (10, 5)
    assert np.allclose(dist, np.sort(naive, axis=1)[:, :5])
    assert np.allclose(dist, naive[np.arange(10)[:, None], idx])
    assert (np.diff(dist, axis=1) >= 0).all()
    self_hit = brute_query(anchors, anchors[:3], k=1)
    assert np.allclose(self_hit[0][:, 0], 0.0)
    assert (self_hit[1][:, 0] == [0, 1, 2]).all()


def test_brute_query_validation():
    rng = np.random.default_rng(72)
    a = rng.standard_normal((6, 4))
    with pytest.raises(ValueError, match="dim mismatch"):
        brute_query(a, rng.standard_normal((2, 5)), k=2)
    with pytest.raises(ValueError, match="k must be"):
        brute_query(a, rng.standard_normal((2, 4)), k=7)
    with pytest.raises(ValueError, match="2D"):
        brute_query(rng.standard_normal(4), rng.standard_normal((2, 4)), k=1)


def test_recall_at_k_edges_and_partial():
    exact = np.array([[0, 1, 2], [3, 4, 5]])
    assert recall_at_k(exact, exact, k=3) == 1.0
    assert recall_at_k(np.array([[7, 8, 9], [0, 1, 2]]), exact, k=3) == 0.0
    approx = np.array([[0, 9, 9], [3, 4, 9]])
    assert recall_at_k(approx, exact, k=2) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="share one 2D shape"):
        recall_at_k(exact, exact[:, :2], k=2)


def test_pq_footprint_ratio_and_bytes():
    assert (PQ_M, PQ_K) == (16, 256)
    assert compression_ratio(128) == pytest.approx(32.0)
    assert compression_ratio(128, 16) == raw_bytes(1000, 128) / pq_bytes(1000)
    assert raw_bytes(80_000_000, 128) == 40_960_000_000
    assert pq_bytes(80_000_000) == 1_280_000_000
    with pytest.raises(ValueError, match="dim"):
        compression_ratio(0)


def test_should_use_pq_both_branches():
    assert should_use_pq(0.80, 0.795, 0.95) is True
    assert should_use_pq(0.80, 0.79, 0.95) is False
    assert should_use_pq(0.80, 0.795, 0.93) is False
    assert should_use_pq(0.80, 0.81, 0.95) is True


@needs_hnswlib
def test_hnsw_recall_matches_brute_on_smoke():
    assert (DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH) == (24, 128, 100)
    rng = np.random.default_rng(72)
    anchors = rng.standard_normal((300, 16)).astype(np.float32)
    queries = rng.standard_normal((30, 16)).astype(np.float32)
    _, exact_idx = brute_query(anchors, queries, k=10)
    index = build_index(anchors)
    dist, idx = query_index(index, queries, k=10)
    assert dist.shape == (30, 10) and idx.shape == (30, 10)
    assert recall_at_k(idx, exact_idx, k=10) > 0.95
    with pytest.raises(ValueError, match="m must be"):
        build_index(anchors, m=8)
