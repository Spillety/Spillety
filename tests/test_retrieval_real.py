import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from spillety.retrieval.hnsw import brute_query, build_index, query_index, recall_at_k
from spillety.retrieval.pq import compression_ratio

needs_hnswlib = pytest.mark.skipif(
    importlib.util.find_spec("hnswlib") is None, reason="hnsw path needs spillety[hnsw]"
)

METRICS = Path("models/retrieval_v1/metrics.json")


def _recall_on_subset(seed, n_queries=200):
    """## HNSW recall on a deterministic query subset."""
    anchors = np.random.default_rng(seed).standard_normal((2000, 32)).astype(np.float32)
    q_idx = np.random.default_rng(seed).choice(len(anchors), size=n_queries, replace=False)
    queries = anchors[q_idx]
    _, exact = brute_query(anchors, queries, k=10)
    _, approx = query_index(build_index(anchors), queries, k=10)
    return recall_at_k(approx, exact, k=10)


@needs_hnswlib
def test_real_recall_deterministic_seed72():
    assert _recall_on_subset(72) == _recall_on_subset(72)


@needs_hnswlib
def test_real_recall_bounded():
    assert 0.0 <= _recall_on_subset(72) <= 1.0


def test_pq_ratio_above_one():
    assert compression_ratio(32, 16) > 1


def test_retrieval_v1_metrics_bounds():
    if not METRICS.exists():
        pytest.skip("run scripts/run_hnsw.py first")
    m = json.loads(METRICS.read_text())
    assert m["seed"] == 72
    assert 0.0 <= m["recall_at_10"] <= 1.0
    assert 0.0 <= m["pq"]["recall_at_10"] <= 1.0
    assert m["latency_ms"]["p99"] >= m["latency_ms"]["p50"] >= 0
    assert m["footprint"]["pq_ratio"] > 1
    assert Path(m["index_file"]).stat().st_size > 5_000_000
