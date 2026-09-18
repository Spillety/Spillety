"""## H2 decision-path golden suite on seed-72 synthetic (fit -> predict -> evaluate)."""

import math

import numpy as np
from numpy.testing import assert_array_equal

from spillety.pipeline.pipeline import SpilletyPipeline

GOLDEN_PR_AUC = 0.9761904761904762
GOLDEN_BRIER = 0.019999743170400933
GOLDEN_ECE = 0.019999871584375113
BUDGET = 5


def _synthetic_frames(rng, n, n_pos, k=5):
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    perm = rng.permutation(n)
    y = y[perm]
    distances = rng.standard_normal((n, k))
    distances[y == 1] -= 2.0
    freqs = rng.standard_normal((n, 2))
    freqs[y == 1, 0] += 1.5
    min_dist = rng.standard_normal((n, 3))
    min_dist[y == 1] -= 1.0
    sensitivity = np.column_stack(
        [
            rng.uniform(0.5, 1.0, n),
            np.where(y == 1, 3.0, 1.0),
            np.where(y == 1, 2.0, 1.0),
            rng.uniform(0.0, 0.3, n),
        ]
    )
    graph = rng.standard_normal((n, 3))
    temporal = rng.standard_normal((n, 2))
    context = rng.standard_normal((n, 2))
    frames = {
        "distances": distances,
        "anchor_type_freqs": freqs,
        "anchor_min_dist": min_dist,
        "sensitivity": sensitivity,
        "graph": graph,
        "temporal": temporal,
        "context": context,
    }
    return frames, y


def _split(frames, y, rng):
    n = len(y)
    idx = rng.permutation(n)
    train_idx, valid_idx, test_idx = idx[:300], idx[300:400], idx[400:]
    sub = lambda d, i: {k: np.asarray(v)[i] for k, v in d.items()}
    return (
        sub(frames, train_idx),
        y[train_idx],
        sub(frames, valid_idx),
        y[valid_idx],
        sub(frames, test_idx),
        y[test_idx],
    )


def _fresh_split():
    rng = np.random.default_rng(72)
    frames, y = _synthetic_frames(rng, 500, 40)
    return _split(frames, y, rng)


def _fit(config=None):
    f_train, y_train, f_valid, y_valid, f_test, y_test = _fresh_split()
    pipe = SpilletyPipeline(
        config={"random_state": 72, "n_boot": 100, **(config or {})}
    )
    pipe.fit(f_train, y_train, f_valid, y_valid)
    return pipe, f_test, y_test


def test_golden_metrics():
    pipe, f_test, y_test = _fit()
    scores, _, _ = pipe.predict(f_test)
    metrics = pipe.evaluate(y_test, scores)
    assert abs(metrics["pr_auc"] - GOLDEN_PR_AUC) < 1e-6
    assert abs(metrics["brier"] - GOLDEN_BRIER) < 1e-6
    assert abs(metrics["ece"] - GOLDEN_ECE) < 1e-6


def test_thresholds_and_model_choice():
    pipe, _, _ = _fit()
    assert pipe.tau1_ >= pipe.tau2_ >= pipe.tau3_
    assert pipe.num_leaves_ in {31, 63, 127}
    assert pipe.calibrator_name_ in {"isotonic", "beta"}


def test_determinism_bitwise():
    pipe_a, f_test, _ = _fit()
    pipe_b, _, _ = _fit()
    scores_a = pipe_a.predict_proba(f_test)
    scores_b = pipe_b.predict_proba(f_test)
    assert_array_equal(scores_a, scores_b)


def test_budget_binds():
    free, _, _ = _fit()
    pipe, f_test, y_test = _fit(config={"budget": BUDGET})
    scores, _, _ = pipe.predict(f_test)
    metrics = pipe.evaluate(y_test, scores)
    assert metrics["alerts"] <= BUDGET
    assert math.isfinite(metrics["cost"])
    assert metrics["ece"] < 0.1
    assert pipe.tau1_ >= free.tau1_


def test_operating_asserts():
    pipe, f_test, y_test = _fit()
    scores, _, _ = pipe.predict(f_test)
    metrics = pipe.evaluate(y_test, scores)
    assert 0.0 <= metrics["ece"] < 0.1
    assert math.isfinite(metrics["cost"]) and metrics["cost"] >= 0.0
    assert metrics["alerts"] == int((scores >= pipe.tau1_).sum())
