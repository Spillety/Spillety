"""## Direct unit tests for five uncovered core functions (seed 72 everywhere)."""

import numpy as np
import pandas as pd

from spillety.causal.select import power_n_min
from spillety.causal.validate import partial_corr_pvalue
from spillety.evidence.merkle import leaf_hash
from spillety.features.temporal import time_since_last_illicit_series
from spillety.metrics.dashboard import pr_auc_score


def test_power_n_min_boundaries():
    """## Larger effects need fewer samples; minimum is 4."""
    assert power_n_min(1.0) == 4
    assert power_n_min(0.5) < power_n_min(0.1) < power_n_min(0.0)
    for effect in (0.0, 0.1, 0.5, 0.99, 1.0):
        assert power_n_min(effect) >= 4


def test_partial_corr_pvalue_collinear():
    """## Collinear pair gives r near 1 with a significant p-value."""
    x = np.arange(200, dtype=float)
    y = 3.0 * x + 7.0
    r, p = partial_corr_pvalue(x, y)
    assert r > 0.99
    assert p < 0.05


def test_leaf_hash_determinism_and_hex():
    """## Same payload hashes equal; digest is 32 bytes (64 hex chars)."""
    h1 = leaf_hash(b"evidence-leaf")
    assert h1 == leaf_hash(b"evidence-leaf")
    assert len(h1) == 32
    assert len(h1.hex()) == 64
    assert h1 != leaf_hash(b"evidence-leaf-tampered")


def test_pr_auc_perfect_and_constant():
    """## Perfect ranking scores 1.0; constant scores match the base rate."""
    y = np.array([0, 0, 1, 1])
    assert pr_auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    rng = np.random.default_rng(72)
    yc = (rng.uniform(0, 1, 1000) < 0.3).astype(int)
    assert abs(pr_auc_score(yc, np.full(1000, 0.5)) - yc.mean()) < 0.05


def test_tsil_monotone_on_synthetic_series():
    """## Time-since-last-illicit is 0 at the illicit step, then non-decreasing."""
    df = pd.DataFrame(
        {
            "time_step": list(range(1, 9)),
            "class": ["2", "2", "1", "2", "2", "2", "2", "2"],
        }
    )
    s = time_since_last_illicit_series(df, pd.Index(range(1, 9)))
    assert s.loc[3] == 0.0
    tail = s.loc[3:].to_numpy()
    assert bool((np.diff(tail) >= 0).all())
    assert (s.loc[1:2].to_numpy() == 49.0).all()
