import numpy as np

from spillety.entity.dbscan import choose_dbscan_params, cluster_dbscan, select_eps

SEED = 72


def test_select_eps_deterministic():
    rng = np.random.default_rng(SEED)
    X = rng.normal(0, 1, (100, 4))
    eps1 = select_eps(X, min_pts=5, q=0.95)
    eps2 = select_eps(X, min_pts=5, q=0.95)
    assert eps1 == eps2
    assert eps1 > 0.0


def test_cluster_dbscan_size_cap_splits_giant():
    rng = np.random.default_rng(SEED)
    # One tight blob of 200 points
    blob = rng.normal(0, 0.05, (200, 2))
    # Sparse noise
    noise = rng.uniform(-10, 10, (20, 2))
    X = np.vstack([blob, noise])
    # Large eps, small min_pts -> one giant cluster
    labels = cluster_dbscan(X, eps=1.0, min_pts=3, max_size=50)
    # Should split the 200-point blob into clusters <= 50
    for cid in np.unique(labels):
        if cid == -1:
            continue
        assert (labels == cid).sum() <= 50
    # Noise stays -1
    assert (labels == -1).sum() >= 10


def test_choose_dbscan_params_cost_selection():
    rng = np.random.default_rng(SEED)
    # Two well-separated blobs
    c1 = rng.normal(-3, 0.3, (80, 2))
    c2 = rng.normal(3, 0.3, (80, 2))
    noise = rng.uniform(-8, 8, (20, 2))
    X = np.vstack([c1, c2, noise])
    y = np.array([0] * 80 + [1] * 80 + [-1] * 20)
    # Cost: FP expensive, FN cheap -> prefer tighter clusters
    res = choose_dbscan_params(X, y, cost_fp=10.0, cost_fn=1.0)
    assert "eps" in res and "min_pts" in res
    assert res["cost"] >= 0.0
    assert 0.0 <= res["noise_frac"] <= 1.0
    assert res["n_clusters"] >= 1
    # Deterministic on same seed
    res2 = choose_dbscan_params(X, y, cost_fp=10.0, cost_fn=1.0)
    assert res["eps"] == res2["eps"]
    assert res["min_pts"] == res2["min_pts"]


def test_edge_cases_empty_and_singleton():
    # Empty
    X_empty = np.empty((0, 2))
    labels = cluster_dbscan(X_empty, eps=1.0, min_pts=3, max_size=10)
    assert labels.shape == (0,)

    # Single point
    X_one = np.array([[0.0, 0.0]])
    labels = cluster_dbscan(X_one, eps=1.0, min_pts=3, max_size=10)
    assert labels.shape == (1,)
    assert labels[0] == -1  # noise because min_pts=3

    # All noise
    rng = np.random.default_rng(SEED)
    X_sparse = rng.uniform(-100, 100, (30, 3))
    labels = cluster_dbscan(X_sparse, eps=0.1, min_pts=5, max_size=10)
    assert (labels == -1).all()

    # choose_dbscan_params with trivial labels (all noise ground truth)
    y_trivial = np.array([-1] * 30)
    res = choose_dbscan_params(X_sparse, y_trivial, cost_fp=1.0, cost_fn=1.0)
    assert "eps" in res
    # When ground truth is all noise, best params may still find spurious clusters;
    # the cost breakdown should reflect FP/FN tradeoff.
    assert "cost_breakdown" in res
    assert res["cost"] >= 0.0