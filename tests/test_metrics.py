import numpy as np

from spillety.metrics.fairness import equalized_odds
from spillety.metrics.operational import (
    alert_to_sar_rate,
    bootstrap_ci,
    cost_per_alert,
    fp_rate,
    latency_p99,
    ttd,
)


def test_operational_formulas_on_fixed_arrays():
    assert list(ttd([0.0, 10.0], [5.0, 20.0])) == [5.0, 10.0]
    assert alert_to_sar_rate(3, 12) == 0.25
    assert alert_to_sar_rate(0, 0) == 0.0
    assert fp_rate([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5
    assert fp_rate([1, 0], [0, 0]) == 0.0
    assert cost_per_alert(900.0, 100.0, 50) == 20.0
    assert cost_per_alert(1.0, 1.0, 0) == 0.0
    lat = np.arange(100, dtype=float)
    assert latency_p99(lat) == float(np.percentile(lat, 99))


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(72)
    values = (rng.random(2000) < 0.3).astype(float)
    ci = bootstrap_ci(values, np.mean, n_boot=1000, random_state=72)
    assert ci["lower"] <= 0.3 <= ci["upper"]
    assert ci["n_boot"] == 1000


def test_equalized_odds_gaps_on_fixed_groups():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    groups = np.array(["a"] * 4 + ["b"] * 4)
    out = equalized_odds(y_true, y_pred, groups, n_boot=200, random_state=72)
    assert out["tpr"]["a"] == 0.5
    assert out["tpr"]["b"] == 1.0
    assert out["fpr"]["a"] == 0.0
    assert out["fpr"]["b"] == 0.5
    assert out["delta_tpr"] == 0.5
    assert out["delta_fpr"] == 0.5


def test_equalized_odds_ci_covers_zero_when_fair():
    rng = np.random.default_rng(72)
    n = 800
    y_true = (rng.random(n) < 0.3).astype(int)
    y_pred = (rng.random(n) < 0.3).astype(int)
    groups = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    out = equalized_odds(y_true, y_pred, groups, n_boot=500, random_state=72)
    assert out["delta_tpr"] < 0.08
    assert out["delta_fpr"] < 0.08
    assert out["ci"]["delta_tpr_lower"] <= out["delta_tpr"] <= out["ci"]["delta_tpr_upper"]
    assert out["ci"]["delta_fpr_lower"] <= out["delta_fpr"] <= out["ci"]["delta_fpr_upper"]
