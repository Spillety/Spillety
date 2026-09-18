import numpy as np

from spillety.causal.filter import causal_filter
from spillety.causal.select import power_gate, select_dag
from spillety.causal.validate import ci_test, falsification_test


def _synthetic_dag(rng, n=2000):
    c = rng.standard_normal(n)
    w = c + 0.5 * rng.standard_normal(n)
    spurious = c + 0.5 * rng.standard_normal(n)
    causal = w + 0.5 * rng.standard_normal(n)
    return c, w, np.column_stack([spurious, causal])


def test_spurious_excluded_causal_kept():
    rng = np.random.default_rng(72)
    c, w, anchors = _synthetic_dag(rng)
    out = causal_filter(anchors, w, c.reshape(-1, 1))
    assert out["mask"].tolist() == [False, True]
    assert out["pass_rate"] == 0.5


def test_pass_rate_and_sensitivity_bounds():
    rng = np.random.default_rng(72)
    c, w, anchors = _synthetic_dag(rng)
    out = causal_filter(anchors, w, c.reshape(-1, 1))
    assert 0.0 <= out["pass_rate"] <= 1.0
    assert out["mask"].shape == (2,)
    for key in ("e_value", "gamma", "p_values", "partial_corr"):
        assert out[key].shape == (2,)
        assert np.all(np.isfinite(out[key]))
    assert np.all(out["e_value"] >= 1.0) and np.all(out["gamma"] >= 1.0)
    assert out["e_value"][1] > out["e_value"][0]
    assert ci_test(anchors[:, 1], w, c.reshape(-1, 1))["reject"]
    assert not ci_test(anchors[:, 0], w, c.reshape(-1, 1))["reject"]


def test_falsification_catches_substitution():
    rng = np.random.default_rng(72)
    n = 500
    controls = rng.standard_normal(n)
    x_indep = rng.standard_normal(n)
    y_clean = 2.0 * controls + 0.5 * rng.standard_normal(n)
    assert not falsification_test(y_clean, x_indep, controls.reshape(-1, 1))["reject"]
    y_swapped = 2.0 * x_indep + 0.5 * rng.standard_normal(n)
    assert falsification_test(y_swapped, x_indep, controls.reshape(-1, 1))["reject"]


def test_select_dag_parsimony_and_power_gate():
    out = select_dag(
        {
            "DAG-A": {"rejections": 0, "num_edges": 5},
            "DAG-B": {"rejections": 0, "num_edges": 3},
            "DAG-C": {"rejections": 2, "num_edges": 2},
        }
    )
    assert out["best"] == "DAG-B" and out["status"] == "ok"
    gate = power_gate(20, 0.3)
    assert gate["status"] == "insufficient data" and not gate["enough"]
    empty = select_dag({"DAG-D": {"rejections": 0, "num_edges": 1, "n": 20, "effect": 0.3}})
    assert empty["best"] is None and empty["status"] == "insufficient data"
