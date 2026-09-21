import numpy as np

from spillety.entity.cioh import cluster_cioh
from spillety.entity.dbscan import choose_dbscan_params, cluster_dbscan
from spillety.entity.fusion import (
    brier_score,
    copula_proba,
    ece_score,
    estimate_prior,
    fit_copula_theta,
    fit_logistic,
    pr_auc,
    select_fusion,
    to_pseudo_obs,
)
from spillety.entity.unionfind import UnionFind, select_tau_star

SEED = 72


def _owners_and_txs(seed=SEED):
    rng = np.random.default_rng(seed)
    owners: dict[str, int] = {}
    txs: list[list[str]] = []
    for owner in range(10):
        addrs = [f"o{owner}a{i}" for i in range(4)]
        for a in addrs:
            owners[a] = owner
        for _ in range(3):
            txs.append([str(x) for x in rng.choice(addrs, size=2, replace=False)])
    cj = ["o0a0", "o1a0", "o2a0"]
    xh = ["o3a0", "o4a0"]
    tr = ["o5a0", "o6a0"]
    txs += [cj, xh, tr]
    flags = (
        [False] * (len(txs) - 3) + [True, False, False],
        [False] * (len(txs) - 3) + [False, True, False],
        [False] * (len(txs) - 3) + [False, False, True],
    )
    return owners, txs, flags


def _pair_precision(owners, clusters):
    cid = {}
    for i, c in enumerate(clusters):
        for a in c:
            cid[a] = i
    tp = fp = 0
    addrs = sorted(owners)
    for i in range(len(addrs)):
        for j in range(i + 1, len(addrs)):
            a, b = addrs[i], addrs[j]
            if cid.get(a) == cid.get(b):
                if owners[a] == owners[b]:
                    tp += 1
                else:
                    fp += 1
    return tp / (tp + fp) if tp + fp else 1.0


def test_cioh_precision_smoke_and_exclusions():
    owners, txs, (cj, xh, tr) = _owners_and_txs()
    clusters = cluster_cioh(txs, is_coinjoin=cj, is_exchange_hot=xh, is_taproot=tr)
    assert _pair_precision(owners, clusters) >= 0.90
    again = cluster_cioh(txs, is_coinjoin=cj, is_exchange_hot=xh, is_taproot=tr)
    assert clusters == again
    mixed = cluster_cioh(txs)
    assert _pair_precision(owners, mixed) <= _pair_precision(owners, clusters)


def _synthetic_signals(seed=SEED, n_pos=200, n_neg=200):
    rng = np.random.default_rng(seed)
    latent = rng.uniform(0, 1, n_pos)
    pos = np.column_stack(
        [
            np.clip(latent + rng.normal(0, 0.08, n_pos), 0, 1),
            np.clip(latent + rng.normal(0, 0.08, n_pos), 0, 1),
            rng.uniform(0.4, 1.0, n_pos),
        ]
    )
    neg = np.column_stack(
        [
            rng.uniform(0, 1, n_neg),
            rng.uniform(0, 1, n_neg),
            rng.uniform(0, 1, n_neg),
        ]
    )
    x = np.vstack([pos, neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def _fusion_candidates(x, y):
    n = len(y)
    xtr, ytr, xva, yva = x[: n // 2], y[: n // 2], x[n // 2 :], y[n // 2 :]
    prior = estimate_prior(ytr)
    assert prior == float(np.clip(ytr.mean(), 1e-6, 1 - 1e-6))
    model = fit_logistic(xtr, ytr)
    cands = {"logistic": model.predict_proba(xva)[:, 1]}
    uva = to_pseudo_obs(xva[:, :2])
    utr = to_pseudo_obs(xtr[:, :2])
    for fam in ("clayton", "gumbel", "frank"):
        t1 = fit_copula_theta(utr[ytr == 1], fam)
        t0 = fit_copula_theta(utr[ytr == 0], fam)
        cands[fam] = copula_proba(uva, fam, t1, t0, prior)
    return yva, cands


def test_fusion_selection_deterministic_and_ranked():
    x, y = _synthetic_signals()
    yva, cands = _fusion_candidates(x, y)
    first = select_fusion(yva, cands)
    second = select_fusion(yva, cands)
    assert first["best"] == second["best"]
    assert set(first["metrics"]) == {"logistic", "clayton", "gumbel", "frank"}
    best = first["best"]
    for name, row in first["metrics"].items():
        assert row["pr_auc"] <= first["metrics"][best]["pr_auc"] + 1e-12
        assert row["pr_auc"] == pr_auc(yva, cands[name])
        assert row["ece"] == ece_score(yva, cands[name])
        assert row["brier"] == brier_score(yva, cands[name])
    leaders = [
        n
        for n, r in first["metrics"].items()
        if abs(r["pr_auc"] - first["metrics"][best]["pr_auc"]) <= 1e-12
    ]
    assert first["metrics"][best]["ece"] <= min(first["metrics"][n]["ece"] for n in leaders) + 1e-12


def test_unionfind_invariants_and_cap():
    dsu = UnionFind(max_cluster_size=3)
    for v in ("a", "b", "c", "d"):
        dsu.add(v)
    assert dsu.union("a", "b") is True
    assert dsu.union("b", "c") is True
    assert dsu.find("a") == dsu.find("c")
    assert dsu.size_of("a") == 3
    assert dsu.union("a", "d") is False
    assert dsu.find("d") != dsu.find("a")
    assert dsu.union("a", "c") is True
    clusters = dsu.clusters()
    assert clusters == sorted(clusters, key=lambda c: sorted(repr(m) for m in c))
    flat = [m for c in clusters for m in c]
    assert sorted(flat) == ["a", "b", "c", "d"]
    assert {len(c) for c in clusters} == {1, 3}


def test_tau_star_minimizes_cost():
    rng = np.random.default_rng(SEED)
    y = (rng.uniform(0, 1, 500) < 0.3).astype(int)
    s = np.clip(
        y * rng.normal(0.6, 0.2, 500) + (1 - y) * rng.normal(0.4, 0.2, 500), 0, 1
    )
    tau, cost = select_tau_star(y, s, c_fp=2.0, c_fn=1.0)
    assert 0.0 <= tau <= 1.0
    grid = np.linspace(0, 1, 1001)
    brute = min(
        2.0 * int((((s >= t).astype(int) == 1) & (y == 0)).sum())
        + 1.0 * int((((s >= t).astype(int) == 0) & (y == 1)).sum())
        for t in grid
    )
    assert cost <= brute + 1e-9
    _, cost_mid = select_tau_star(y, s)
    pred_mid = (s >= 0.5).astype(int)
    mid = float(((pred_mid == 1) & (y == 0)).sum() + ((pred_mid == 0) & (y == 1)).sum())
    assert cost_mid <= mid


def test_choose_dbscan_params_returns_valid_dict():
    rng = np.random.default_rng(SEED)
    X = rng.normal(0, 1, (100, 4))
    y = rng.integers(0, 2, 100)
    res = choose_dbscan_params(X, y, cost_fp=10.0, cost_fn=1.0)
    assert isinstance(res, dict)
    for key in ("eps", "min_pts", "cost", "cost_breakdown", "n_clusters", "noise_frac"):
        assert key in res, f"missing key {key}"
    assert "fp" in res["cost_breakdown"]
    assert "fn" in res["cost_breakdown"]
    assert "cost_fp" in res["cost_breakdown"]
    assert "cost_fn" in res["cost_breakdown"]
    assert isinstance(res["eps"], float)
    assert isinstance(res["min_pts"], int)
    assert res["n_clusters"] >= 1
    assert 0.0 <= res["noise_frac"] <= 1.0


def test_cluster_dbscan_produces_valid_labels():
    rng = np.random.default_rng(SEED)
    blob = rng.normal(0, 0.1, (50, 2))
    noise = rng.uniform(-5, 5, (10, 2))
    X = np.vstack([blob, noise])
    labels = cluster_dbscan(X, eps=0.5, min_pts=3, max_size=100)
    assert labels.ndim == 1
    assert len(labels) == len(X)
    assert labels.dtype == np.int64 or labels.dtype == np.int32
    for cid in np.unique(labels):
        if cid == -1:
            continue
        assert (labels == cid).sum() <= 100


def test_dbscan_cluster_count_reasonable_on_validation_embeddings():
    rng = np.random.default_rng(SEED)
    c1 = rng.normal(-3, 0.3, (60, 2))
    c2 = rng.normal(3, 0.3, (60, 2))
    noise = rng.uniform(-8, 8, (20, 2))
    X = np.vstack([c1, c2, noise])
    labels = cluster_dbscan(X, eps=0.8, min_pts=5, max_size=50)
    n_clusters = len([c for c in np.unique(labels) if c != -1])
    assert 1 <= n_clusters <= 10, f"unexpected cluster count {n_clusters}"
