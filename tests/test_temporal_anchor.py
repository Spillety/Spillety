import numpy as np
import pytest

from spillety.embeddings.pairs import temporal_anchor_split
from spillety.retrieval.ofac_anchors import build_anchor_pool_temporal


def test_temporal_anchor_split():
    np.random.default_rng(42)
    n = 49
    indices = np.arange(n)
    times = np.arange(1, n + 1)
    split = temporal_anchor_split(indices, times, era_split=30)
    assert set(split.keys()) == {"train", "valid", "test"}
    assert len(split["train"]) == 30
    assert len(split["valid"]) == 10
    assert len(split["test"]) == 9


def test_temporal_anchor_split_disjoint():
    np.random.default_rng(42)
    n = 49
    indices = np.arange(n)
    times = np.arange(1, n + 1)
    split = temporal_anchor_split(indices, times, era_split=30)
    train_s, valid_s, test_s = set(split["train"]), set(split["valid"]), set(split["test"])
    assert train_s.isdisjoint(valid_s)
    assert train_s.isdisjoint(test_s)
    assert valid_s.isdisjoint(test_s)
    assert train_s | valid_s | test_s == set(indices)


def test_temporal_anchor_split_custom_split():
    indices = np.arange(20)
    times = np.arange(1, 21)
    split = temporal_anchor_split(indices, times, era_split=10)
    assert len(split["train"]) == 10
    assert len(split["valid"]) == 10
    assert len(split["test"]) == 0


def test_temporal_anchor_split_empty():
    indices = np.array([], dtype=int)
    times = np.array([], dtype=int)
    split = temporal_anchor_split(indices, times, era_split=30)
    assert all(len(v) == 0 for v in split.values())


def test_build_anchor_pool_temporal():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((49, 8))
    labels = np.array([1] * 25 + [0] * 24)
    times = np.arange(1, 50)
    ofac = rng.standard_normal((5, 8)) + 5.0
    pools = build_anchor_pool_temporal(emb, labels, times, ofac, era_split=30)
    assert "train" in pools
    assert "valid" in pools
    assert "test" in pools
    for name, pool in pools.items():
        assert pool.era == name
        assert pool.kind in ("ofac", "mixed", "train_illicit")


def test_build_anchor_pool_temporal_disjoint():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((49, 4))
    labels = np.array([1] * 15 + [0] * 10 + [1] * 10 + [0] * 4 + [1] * 5 + [0] * 5)
    times = np.arange(1, 50)
    pools = build_anchor_pool_temporal(emb, labels, times, None, era_split=30)
    total = sum(pool.matrix.shape[0] for pool in pools.values())
    assert total == 30  # only illicit anchors are kept in the pool
    assert set(pools.keys()) == {"train", "valid", "test"}


def test_build_index_temporal():
    rng = np.random.default_rng(42)
    anchors = {
        "train": rng.standard_normal((20, 8)),
        "test": rng.standard_normal((10, 8)),
    }
    try:
        from spillety.retrieval.hnsw import build_index_temporal
        indices = build_index_temporal(anchors)
        assert set(indices.keys()) == {"train", "test"}
        for idx in indices.values():
            assert idx is not None
    except ImportError:
        pytest.skip("hnswlib not installed")


def test_build_index_temporal_empty_era():
    rng = np.random.default_rng(42)
    anchors = {
        "train": rng.standard_normal((20, 8)),
        "valid": np.zeros((0, 8)),
    }
    try:
        from spillety.retrieval.hnsw import build_index_temporal
        indices = build_index_temporal(anchors)
        assert set(indices.keys()) == {"train"}
        assert "valid" not in indices
    except ImportError:
        pytest.skip("hnswlib not installed")


def test_ofac_pool_coverage_metric():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((49, 4))
    labels = np.array([1] * 25 + [0] * 24)
    ofac = rng.standard_normal((5, 4)) + 5.0
    pools = build_anchor_pool_temporal(emb, labels, np.arange(1, 50), ofac, era_split=30)
    for p in pools.values():
        assert isinstance(p.coverage, float)
        assert 0.0 <= p.coverage <= 1.0
