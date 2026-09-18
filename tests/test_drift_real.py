## Real-slice drift tests: determinism on seed-72 cuts, regime branches.

import numpy as np

from spillety.temporal.drift import cohen_d, is_drift, ks_stats, median_distance, tau_d
from spillety.temporal.policy import classify_regime, regime_action

SEED = 72


def _slices():
    """## Fixed seed-72 cuts shared by determinism checks."""
    rng = np.random.default_rng(SEED)
    z_old = rng.normal(0, 1, (200, 8))
    z_new = rng.normal(0.5, 1, (50, 8))
    return z_old, z_new


def test_median_distance_deterministic():
    """## Same seed-72 cut gives identical median distance on repeat."""
    z_old, z_new = _slices()
    first = [median_distance(z, z_old) for z in z_new]
    z_old2, z_new2 = _slices()
    second = [median_distance(z, z_old2) for z in z_new2]
    assert first == second
    assert first[0] == median_distance(z_new2[0], z_old2)


def test_is_drift_deterministic_on_slices():
    """## KS triple on fixed cuts is stable and separates shift from noise."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(0, 1, 2000)
    same = rng.normal(0, 1, 2000)
    shifted = rng.normal(2.0, 1, 2000)
    d0, p0 = ks_stats(a, same)
    d1, p1 = ks_stats(a, shifted)
    assert is_drift(p0, d0, cohen_d(a, same)) is False
    assert is_drift(p1, d1, cohen_d(a, shifted)) is True
    d0b, p0b = ks_stats(a, same)
    assert (d0, p0) == (d0b, p0b)
    assert tau_d(np.arange(100, dtype=float)) == tau_d(np.arange(100, dtype=float))


def test_classify_regime_branches():
    """## Full decision table maps to the four regimes and actions."""
    assert classify_regime(True, 0.10) == "drift"
    assert classify_regime(False, 0.10) == "novel"
    assert classify_regime(True, 0.50) == "noise"
    assert classify_regime(False, 0.50) == "normal"
    assert regime_action("drift") == "retrain"
    assert regime_action("novel") == "review"
    assert regime_action("noise") == "ignore"
    assert regime_action("normal") == "incremental"
