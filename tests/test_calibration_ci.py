import numpy as np
import pytest

from spillety.metrics.dashboard import ece_bootstrap_ci
from spillety.metrics.operational import bootstrap_ci
from spillety.temporal.leadtime import LeadTimeResult, evaluate_lead_time


def test_ece_bootstrap_ci_structure():
    rng = np.random.default_rng(72)
    y_true = (rng.random(500) < 0.1).astype(int)
    y_prob = y_true * rng.beta(5, 1, 500) + (1 - y_true) * rng.beta(1, 5, 500)
    res = ece_bootstrap_ci(y_true, y_prob, n_bins=10, n_boot=100, random_state=72)
    assert isinstance(res, dict)
    assert "ece" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert "n_boot" in res
    assert res["n_boot"] == 100
    assert isinstance(res["ece"], float)
    assert isinstance(res["ci_lower"], float)
    assert isinstance(res["ci_upper"], float)


def test_ece_bootstrap_ci_contains_true_ece():
    rng = np.random.default_rng(72)
    y_true = (rng.random(500) < 0.1).astype(int)
    y_prob = y_true * rng.beta(5, 1, 500) + (1 - y_true) * rng.beta(1, 5, 500)
    res = ece_bootstrap_ci(y_true, y_prob, n_bins=10, n_boot=500, random_state=72)
    assert res["ci_lower"] <= res["ece"] <= res["ci_upper"]


def test_ece_bootstrap_ci_ci_lower_leq_upper():
    rng = np.random.default_rng(72)
    y_true = (rng.random(300) < 0.1).astype(int)
    y_prob = y_true * rng.beta(5, 1, 300) + (1 - y_true) * rng.beta(1, 5, 300)
    res = ece_bootstrap_ci(y_true, y_prob, n_bins=10, n_boot=100, random_state=72)
    assert res["ci_lower"] <= res["ci_upper"]


def test_ece_bootstrap_ci_reproducibility():
    rng = np.random.default_rng(72)
    y_true = (rng.random(300) < 0.1).astype(int)
    y_prob = y_true * rng.beta(5, 1, 300) + (1 - y_true) * rng.beta(1, 5, 300)
    r1 = ece_bootstrap_ci(y_true, y_prob, n_boot=100, random_state=72)
    r2 = ece_bootstrap_ci(y_true, y_prob, n_boot=100, random_state=72)
    assert r1["ece"] == r2["ece"]
    assert r1["ci_lower"] == r2["ci_lower"]
    assert r1["ci_upper"] == r2["ci_upper"]


def test_evaluate_lead_time_synthetic():
    alert_steps = {1: 10, 2: 15, 3: 20, 4: 25}
    sanction_steps = {1: 13, 2: 20, 3: 28, 4: 35}
    res = evaluate_lead_time(alert_steps, sanction_steps)
    assert isinstance(res, LeadTimeResult)
    assert res.lead_steps == [3, 5, 8, 10]
    assert res.median == pytest.approx(6.5)
    assert res.p90 == pytest.approx(9.4)
    assert res.censored_count == 0
    assert res.recall_at_k_new >= 0.0


def test_evaluate_lead_time_from_temporal_data():
    alert_steps = {101: 38, 102: 40, 103: 44}
    sanction_steps = {101: 42, 102: 45, 103: 48}
    res = evaluate_lead_time(alert_steps, sanction_steps, k=100, era_split=43)
    assert isinstance(res, LeadTimeResult)
    assert len(res.lead_steps) == 3
    assert all(s > 0 for s in res.lead_steps)
    assert res.censored_count == 0
    assert "pre_43" in res.by_era
    assert "post_43" in res.by_era


def test_evaluate_lead_time_censored():
    alert_steps = {1: 10}
    sanction_steps = {1: 15, 2: 25, 3: 30}
    res = evaluate_lead_time(alert_steps, sanction_steps)
    assert res.censored_count == 2
    assert isinstance(res, LeadTimeResult)
    assert len(res.lead_steps) == 1


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(72)
    values = (rng.random(2000) < 0.3).astype(float)
    ci = bootstrap_ci(values, np.mean, n_boot=1000, random_state=72)
    assert ci["lower"] <= 0.3 <= ci["upper"]
    assert ci["n_boot"] == 1000
