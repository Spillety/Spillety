import numpy as np

from spillety.retrieval.ofac_anchors import (
    build_anchor_pool,
    load_ofac_pool,
    query_with_fallback,
)


def _synthetic(seed=72):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((20, 4))
    labels = np.array([1] * 10 + [0] * 10)
    ofac = rng.standard_normal((5, 4)) + 5.0
    return emb, labels, ofac


def test_load_empty_and_missing_csv(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert load_ofac_pool(str(empty)) == []
    assert load_ofac_pool(str(tmp_path / "nope.csv")) == []


def test_build_kinds_and_coverage():
    emb, labels, ofac = _synthetic()
    mixed = build_anchor_pool(emb, labels, ofac)
    assert mixed.kind == "mixed"
    assert mixed.coverage == 1.0
    assert mixed.meta == {"n_ofac": 5, "n_train": 10}
    train_only = build_anchor_pool(emb, labels, None)
    assert train_only.kind == "train_illicit"
    assert train_only.coverage == 0.0
    ofac_only = build_anchor_pool(emb, np.zeros(20, dtype=int), ofac)
    assert ofac_only.kind == "ofac"
    assert ofac_only.coverage == 1.0


def test_query_fallback_branches():
    emb, labels, ofac = _synthetic()
    queries = emb[:4]
    dist, _, used = query_with_fallback(build_anchor_pool(emb, labels, ofac), queries, k=3)
    assert used == "ofac"
    assert dist.shape == (4, 3)
    assert (np.diff(dist, axis=1) >= 0).all()
    _, _, used_fb = query_with_fallback(build_anchor_pool(emb, labels, None), queries, k=3)
    assert used_fb == "train_illicit"


def test_fallback_indices_global_into_pool_matrix():
    emb, labels, _ = _synthetic()
    pool = build_anchor_pool(emb, labels, None)
    _, idx, _ = query_with_fallback(pool, emb[:4], k=3)
    assert (idx >= 0).all() and (idx < pool.matrix.shape[0]).all()


def test_deterministic_build_and_query():
    emb, labels, ofac = _synthetic()
    first = build_anchor_pool(emb, labels, ofac)
    second = build_anchor_pool(emb, labels, ofac)
    assert np.array_equal(first.matrix, second.matrix)
    assert first.coverage == second.coverage
    out1 = query_with_fallback(first, emb[:3], k=2)
    out2 = query_with_fallback(second, emb[:3], k=2)
    assert np.array_equal(out1[0], out2[0]) and np.array_equal(out1[1], out2[1])
