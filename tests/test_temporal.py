import numpy as np
import pytest

from spillety.data.canonical import is_finalized
from spillety.temporal.drift import cohen_d, is_drift, ks_stats, median_distance, tau_d
from spillety.temporal.policy import (
    classify_regime,
    hnsw_action,
    mann_kendall,
    regime_action,
    retrain_gate,
    walk_forward_retrain,
)
from spillety.temporal.sampling import audit_range, power_n

RNG = np.random.default_rng(72)


def test_median_distance_exact():
    z_old = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert median_distance(np.array([0.0, 0.0]), z_old) == pytest.approx(2.5)


def test_tau_d_quantiles():
    hist = np.arange(100, dtype=float)
    assert tau_d(hist) == pytest.approx(94.05)
    assert tau_d(hist, strict=True) == pytest.approx(98.01)


def test_is_drift_all_branches():
    assert is_drift(0.01, 0.20, 0.50) is True
    assert is_drift(0.10, 0.20, 0.50) is False
    assert is_drift(0.01, 0.05, 0.50) is False
    assert is_drift(0.01, 0.20, 0.10) is False
    assert is_drift(0.01, 0.20, -0.50) is True


def test_ks_stats_and_cohen_d_separate_shift_from_noise():
    a = RNG.normal(0, 1, 2000)
    same = RNG.normal(0, 1, 2000)
    shifted = RNG.normal(1.5, 1, 2000)
    d0, p0 = ks_stats(a, same)
    assert is_drift(p0, d0, cohen_d(a, same)) is False
    d1, p1 = ks_stats(a, shifted)
    assert is_drift(p1, d1, cohen_d(a, shifted)) is True


def test_classify_regime_table():
    assert classify_regime(True, 0.10) == "drift"
    assert classify_regime(False, 0.10) == "novel"
    assert classify_regime(True, 0.50) == "noise"
    assert classify_regime(False, 0.50) == "normal"
    assert regime_action("drift") == "retrain"
    assert regime_action("novel") == "review"
    assert regime_action("noise") == "ignore"
    assert regime_action("normal") == "incremental"


def test_hnsw_action_branches():
    assert hnsw_action(0.95, 100) == "incremental"
    assert hnsw_action(0.80, 100) == "rebuild"
    assert hnsw_action(0.95, 20000) == "rebuild"


def test_mann_kendall_trends():
    s_down, p_down = mann_kendall(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    assert s_down < 0 and p_down < 0.05
    s_up, p_up = mann_kendall(np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    assert s_up > 0 and p_up < 0.05
    _, p_flat = mann_kendall(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    assert p_flat == 1.0


def test_walk_forward_and_retrain_gate():
    assert walk_forward_retrain(0.80, 0.60) is True
    assert walk_forward_retrain(0.80, 0.76) is False
    assert retrain_gate(0.80, 0.60, 5, 0.30) is True
    assert retrain_gate(0.80, 0.79, -15, 0.01) is True
    assert retrain_gate(0.80, 0.79, 5, 0.30) is False


def test_power_n_reference():
    assert power_n(0.05, 0.01) == 1825


def test_audit_range_rates():
    assert audit_range(100_000, "clear") == (100, 1000)
    assert audit_range(100_000, "block") == (1000, 5000)
    with pytest.raises(KeyError):
        audit_range(100, "tier2")


def test_is_finalized_boundaries():
    assert is_finalized("btc", 6) is True
    assert is_finalized("btc", 5) is False
    assert is_finalized("btc", 1, required=1) is True
    assert is_finalized("btc", 0, required=0) is True
    assert is_finalized("btc", 2, required=3) is False
    assert is_finalized("eth", 32) is True
    assert is_finalized("eth", 31) is False
    assert is_finalized("eth", 12, required=12) is True
    assert is_finalized("eth", 63, required=64) is False
    with pytest.raises(ValueError):
        is_finalized("btc", 6, required=2)
    with pytest.raises(ValueError):
        is_finalized("eth", 12, required=6)
    with pytest.raises(ValueError):
        is_finalized("sol", 10)
