import numpy as np
from sklearn.metrics import average_precision_score

from spillety.models.calibration import brier_score, calibrate
from spillety.models.gbdt import CANDIDATE_NUM_LEAVES, select_num_leaves, train_gbdt


def _synthetic(seed=72):
    rng = np.random.default_rng(seed)
    n, d = 600, 20
    x = rng.standard_normal((n, d))
    y = np.array([1] * 60 + [0] * (n - 60))
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    x[y == 1] += 2.0
    return x[:400], y[:400], x[400:500], y[400:500], x[500:], y[500:]


def test_gbdt_selects_leaves_and_beats_baseline():
    x_train, y_train, x_hold, y_hold, _, _ = _synthetic()
    selected, pr_auc = select_num_leaves(
        x_train, y_train, x_hold, y_hold, n_boot=100, random_state=72
    )
    assert selected in CANDIDATE_NUM_LEAVES
    assert set(pr_auc) == set(CANDIDATE_NUM_LEAVES)

    booster = train_gbdt(x_train, y_train, x_hold, y_hold, random_state=72, n_boot=100)
    proba = np.asarray(booster.predict(x_hold), dtype=float)
    assert average_precision_score(y_hold, proba) > y_hold.mean()


def test_calibration_pick_and_brier_below_trivial():
    x_train, y_train, x_valid, y_valid, x_test, y_test = _synthetic()
    booster = train_gbdt(x_train, y_train, x_valid, y_valid, random_state=72, n_boot=100)
    p_valid = np.asarray(booster.predict(x_valid), dtype=float)
    p_test = np.asarray(booster.predict(x_test), dtype=float)

    result = calibrate(p_valid, y_valid, p_test, y_test, n_boot=100, random_state=72)
    assert result["best"] in ("isotonic", "beta")
    assert result["bootstrap"]["n_boot"] == 100

    q = result["best_estimator"].predict_proba(p_test)[:, 1]
    trivial = float(np.mean((y_test - y_test.mean()) ** 2))
    assert brier_score(y_test, q) < trivial
