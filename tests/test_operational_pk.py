import numpy as np

from spillety.metrics.operational import precision_at_k, savings_vs_baseline


def test_precision_at_k_on_fixed_arrays():
    y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    prec, (lo, hi) = precision_at_k(y_true, scores, k=4, n_bootstrap=200, seed=72)
    assert prec == 0.75
    assert lo <= prec <= hi


def test_precision_at_k_ci_covers_true_on_synthetic():
    rng = np.random.default_rng(72)
    n = 5000
    y_true = (rng.random(n) < 0.1).astype(int)
    scores = y_true * rng.beta(5, 1, n) + (1 - y_true) * rng.beta(1, 5, n)
    _, (lo, hi) = precision_at_k(y_true, scores, k=100, n_bootstrap=500, seed=72)
    true_prec_at_k = float(np.mean(y_true[np.argsort(scores)[::-1][:100]]))
    assert lo <= true_prec_at_k <= hi


def test_precision_at_k_k_exceeds_n():
    y_true = np.array([1, 0, 1])
    scores = np.array([0.9, 0.5, 0.1])
    prec, (_, _) = precision_at_k(y_true, scores, k=10, n_bootstrap=50, seed=72)
    assert prec == 2.0 / 3.0


def test_savings_vs_baseline_on_fixture():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    scores_model = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.1, 0.15, 0.05])
    scores_baseline = np.array([0.6, 0.55, 0.7, 0.4, 0.65, 0.3, 0.2, 0.1])
    out = savings_vs_baseline(y_true, scores_model, scores_baseline, c_fp=1.0, c_fn=10.0)
    assert out["gain"] > 0
    assert out["fp_prevented"] >= 0
    assert out["n_model"] >= 0
    assert out["n_baseline"] >= 0
    assert "cost_fp_model" in out
    assert "cost_fn_model" in out
    assert "cost_baseline" in out


def test_savings_vs_baseline_equal_scores_edge_case():
    y_true = np.array([1, 0, 1, 0])
    scores_model = np.array([0.5, 0.5, 0.5, 0.5])
    scores_baseline = np.array([0.5, 0.5, 0.5, 0.5])
    out = savings_vs_baseline(y_true, scores_model, scores_baseline, c_fp=1.0, c_fn=10.0)
    assert out["gain"] == 0.0
    assert out["fp_prevented"] == 0
    assert out["n_model"] == out["n_baseline"]


def test_precision_at_k_determinism_seed():
    rng = np.random.default_rng(72)
    y_true = (rng.random(200) < 0.2).astype(int)
    scores = rng.random(200)
    p1, ci1 = precision_at_k(y_true, scores, k=50, n_bootstrap=100, seed=72)
    p2, ci2 = precision_at_k(y_true, scores, k=50, n_bootstrap=100, seed=72)
    assert p1 == p2
    assert ci1 == ci2