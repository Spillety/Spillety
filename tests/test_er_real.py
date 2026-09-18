"""## ER on real Elliptic data: determinism and metric bounds (seed 72)."""

import importlib.util
import os

import numpy as np

from spillety.entity.cioh import cluster_cioh
from spillety.entity.fusion import (
    copula_proba,
    estimate_prior,
    fit_copula_theta,
    fit_logistic,
    pr_auc,
    select_fusion,
    to_pseudo_obs,
)

SEED = 72
SUB_EDGES = 5000
N_CLUSTERS_5K = 4347
N_SCORED_5K = 783
N_PURE_5K = 741

_spec = importlib.util.spec_from_file_location(
    "run_er", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_er.py")
)
run_er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_er)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "elliptic_raw")


def _subsample():
    """
    ## Deterministic 5k-edge slice plus CIOH clusters

    Returns
    ----------
    tuple
        (sub-edges, clusters, label map, time map).
    """
    edges, label, tmap = run_er.load_inputs(DATA_DIR)
    sub = edges[
        np.random.default_rng(SEED).choice(len(edges), SUB_EDGES, replace=False)
    ]
    txs = [[a, b] for a, b in sub.tolist() if a != b]
    return sub, cluster_cioh(txs), label, tmap


def test_clustering_deterministic_fixed_counts():
    """## Same seed gives identical clusters with fixed counts and sane precision."""
    sub, first, label, _ = _subsample()
    txs = [[a, b] for a, b in sub.tolist() if a != b]
    assert cluster_cioh(txs) == first
    assert len(first) == N_CLUSTERS_5K
    pure, weighted, n_scored, n_pure = run_er.cluster_precision(first, label)
    assert (n_scored, n_pure) == (N_SCORED_5K, N_PURE_5K)
    assert 0.5 <= pure <= 1.0
    assert 0.5 <= weighted <= 1.0


def test_pair_sampling_and_fusion_deterministic():
    """## Pair signals and logistic-vs-copula ranking are reproducible and bounded."""
    sub, clusters, label, tmap = _subsample()
    cof = {m: i for i, c in enumerate(clusters) for m in sorted(c)}
    x1, y1, t1 = run_er.sample_pairs(sub, cof, label, tmap, SEED, 2000)
    x2, y2, t2 = run_er.sample_pairs(sub, cof, label, tmap, SEED, 2000)
    assert (x1 == x2).all() and (y1 == y2).all() and (t1 == t2).all()
    assert len(y1) <= 2000 and len(np.unique(y1)) == 2
    tr = t1 <= run_er.STEP_CUT
    if tr.sum() < 50 or (~tr).sum() < 50:
        tr = np.zeros(len(y1), bool)
        tr[: len(y1) // 2] = True
    prior = estimate_prior(y1[tr])
    cands = {"logistic": fit_logistic(x1[tr], y1[tr]).predict_proba(x1[~tr])[:, 1]}
    utr, uva = to_pseudo_obs(x1[tr][:, :2]), to_pseudo_obs(x1[~tr][:, :2])
    th1 = fit_copula_theta(utr[y1[tr] == 1], "clayton")
    th0 = fit_copula_theta(utr[y1[tr] == 0], "clayton")
    cands["clayton"] = copula_proba(uva, "clayton", th1, th0, prior)
    sel = select_fusion(y1[~tr], cands)
    assert sel["best"] in cands
    for name, p in cands.items():
        assert 0.0 <= pr_auc(y1[~tr], p) <= 1.0
        assert sel["metrics"][name]["pr_auc"] <= sel["metrics"][sel["best"]]["pr_auc"]
