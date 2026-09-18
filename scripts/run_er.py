"""## ER on Elliptic real data: CIOH clusters plus logistic vs copula fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spillety.entity.cioh import cluster_cioh
from spillety.entity.fusion import (
    copula_proba,
    estimate_prior,
    fit_copula_theta,
    fit_logistic,
    select_fusion,
    to_pseudo_obs,
)

SEED = 72
MAX_PAIRS = 50_000
STEP_CUT = 35

# ponytail: co-spend is a directed-flow proxy, upgrade with UTXO inputs when available.
ASSUMPTION = (
    "edgelist edge (txId1->txId2) is treated as one co-spending transaction "
    "[txId1, txId2], i.e. address≈txId; clusters = transitive closure (connected "
    "components) of this CIOH proxy. Clustering is unsupervised, so no temporal "
    "leakage into cluster precision; fusion uses a temporal split "
    f"(train max(step)<= {STEP_CUT}, valid above)."
)


def load_inputs(data_dir: str) -> tuple[np.ndarray, dict[str, str], dict[str, int]]:
    """
    ## Load edgelist, label and time maps from Elliptic CSVs

    Parameters
    ----------
    data_dir : str
        Directory holding elliptic_txs_*.csv files.

    Returns
    ----------
    tuple[np.ndarray, dict[str, str], dict[str, int]]
        Edge array (M, 2) of str ids, labeled-class map, time-step map.
    """
    edges = pd.read_csv(
        os.path.join(data_dir, "elliptic_txs_edgelist.csv"), dtype=str
    ).to_numpy()
    classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"), dtype=str)
    ids = classes["txId"].astype(str).tolist()
    cls = classes["class"].astype(str).tolist()
    label = {i: c for i, c in zip(ids, cls) if c in ("1", "2")}
    feats = pd.read_csv(
        os.path.join(data_dir, "elliptic_txs_features.csv"), header=None, usecols=[0, 1]
    )
    tmap = dict(zip(feats[0].astype(str), feats[1].astype(int)))
    return edges, label, tmap


def cluster_precision(
    clusters: list[frozenset], label: dict[str, str]
) -> tuple[float, float, int, int]:
    """
    ## Fraction of label-pure clusters, plain and labeled-size-weighted

    Parameters
    ----------
    clusters : list[frozenset]
        CIOH clusters.
    label : dict[str, str]
        Labeled-class map ('1' illicit, '2' licit).

    Returns
    ----------
    tuple[float, float, int, int]
        (pure_rate, weighted_rate, n_scored, n_pure).
    """
    n_scored = n_pure = w_all = w_pure = 0
    for c in clusters:
        labs = [label[m] for m in c if m in label]
        if len(labs) < 2:
            continue
        n_scored += 1
        w_all += len(labs)
        if len(set(labs)) == 1:
            n_pure += 1
            w_pure += len(labs)
    if n_scored == 0:
        return 1.0, 1.0, 0, 0
    return n_pure / n_scored, w_pure / w_all, n_scored, n_pure


def sample_pairs(
    edges: np.ndarray,
    clusters_of: dict[str, int],
    label: dict[str, str],
    tmap: dict[str, int],
    seed: int = SEED,
    max_pairs: int = MAX_PAIRS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ## Sample same/different-cluster labeled pairs with 3 ER signals

    Parameters
    ----------
    edges : np.ndarray
        Edge array (M, 2) for the CIOH adjacency signal.
    clusters_of : dict[str, int]
        Node -> cluster index.
    label : dict[str, str]
        Labeled-class map.
    tmap : dict[str, int]
        Node -> time step.
    seed : int
        RNG seed.
    max_pairs : int
        Cap on sampled pairs.

    Returns
    ----------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (X signals [cioh, temporal, deg_sim], y same-class labels, tmax steps).
    """
    rng = np.random.default_rng(seed)
    rows = edges.tolist()
    eset = {(a, b) for a, b in rows} | {(b, a) for a, b in rows}
    deg: dict[str, int] = {}
    for a, b in rows:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    dmax = max(deg.values()) if deg else 1
    nodes = np.array(sorted(n for n in clusters_of if n in label and n in tmap))
    half = max_pairs // 2
    same: list[tuple] = []
    diff: list[tuple] = []
    order = rng.permutation(len(nodes))
    # ponytail: O(budget) rejection sampling, replace with indexed per-cluster draws at scale.
    for _ in range(max_pairs * 20):
        if len(same) >= half and len(diff) >= half:
            break
        i, j = rng.integers(0, len(nodes), 2)
        if i == j:
            continue
        a, b = nodes[order[i]], nodes[order[j]]
        bucket = same if clusters_of[a] == clusters_of[b] else diff
        if len(bucket) >= half:
            continue
        bucket.append((a, b))
    pairs = same + diff
    X = np.array(
        [
            [
                1.0 if (a, b) in eset else 0.0,
                1.0 if abs(tmap[a] - tmap[b]) <= 1 else 0.0,
                1.0 - abs(deg.get(a, 0) - deg.get(b, 0)) / dmax,
            ]
            for a, b in pairs
        ]
    )
    y = np.array([1 if label[a] == label[b] else 0 for a, b in pairs], dtype=int)
    t = np.array([max(tmap[a], tmap[b]) for a, b in pairs], dtype=int)
    return X, y, t


def run(
    data_dir: str = "data/elliptic_raw",
    out_dir: str = "models/er_v1",
    seed: int = SEED,
    max_pairs: int = MAX_PAIRS,
) -> dict:
    """
    ## Run CIOH clustering plus logistic vs copula fusion comparison

    Parameters
    ----------
    data_dir : str
        Input CSV directory.
    out_dir : str
        Directory for metrics.json and README.md.
    seed : int
        RNG seed for pair sampling.
    max_pairs : int
        Cap on sampled pairs.

    Returns
    ----------
    dict
        Metrics payload also written to metrics.json.
    """
    t0 = time.time()
    edges, label, tmap = load_inputs(data_dir)
    txs = [[a, b] for a, b in edges.tolist() if a != b]
    clusters = cluster_cioh(txs)
    clusters_of = {m: i for i, c in enumerate(clusters) for m in c}
    sizes = np.array([len(c) for c in clusters])
    pure, weighted, n_scored, n_pure = cluster_precision(clusters, label)

    X, y, t = sample_pairs(edges, clusters_of, label, tmap, seed, max_pairs)
    tr, va = t <= STEP_CUT, t > STEP_CUT
    if va.sum() < 100 or len(np.unique(y[va])) < 2 or tr.sum() < 100:
        idx = np.random.default_rng(seed).permutation(len(y))
        tr = np.zeros(len(y), bool)
        tr[idx[: len(y) // 2]] = True
        va = ~tr
    prior = estimate_prior(y[tr])
    cands = {"logistic": fit_logistic(X[tr], y[tr]).predict_proba(X[va])[:, 1]}
    utr, uva = to_pseudo_obs(X[tr][:, :2]), to_pseudo_obs(X[va][:, :2])
    for fam in ("clayton", "gumbel", "frank"):
        th1 = fit_copula_theta(utr[y[tr] == 1], fam)
        th0 = fit_copula_theta(utr[y[tr] == 0], fam)
        cands[fam] = copula_proba(uva, fam, th1, th0, prior)
    sel = select_fusion(y[va], cands)

    metrics = {
        "seed": seed,
        "n_nodes_clustered": len(clusters_of),
        "n_edges": len(edges),
        "n_clusters": len(clusters),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_mean": float(sizes.mean()),
        "size_max": int(sizes.max()),
        "n_singletons": int((sizes == 1).sum()),
        "cluster_precision_unweighted": pure,
        "cluster_precision_weighted": weighted,
        "n_scored_clusters": n_scored,
        "n_pure_clusters": n_pure,
        "n_pairs": len(y),
        "n_pairs_train": int(tr.sum()),
        "n_pairs_valid": int(va.sum()),
        "prior_train": prior,
        "fusion_pr_auc": {k: v["pr_auc"] for k, v in sel["metrics"].items()},
        "fusion_ece": {k: v["ece"] for k, v in sel["metrics"].items()},
        "fusion_brier": {k: v["brier"] for k, v in sel["metrics"].items()},
        "fusion_best": sel["best"],
        "assumption": ASSUMPTION,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ddir = "data/derived"
    os.makedirs(ddir, exist_ok=True)
    pq = os.path.join(ddir, "er_v1_clusters.parquet")
    pd.DataFrame(
        sorted((m, i) for i, c in enumerate(clusters) for m in c),
        columns=["txId", "cluster_id"],
    ).to_parquet(pq, index=False)
    size_mb = os.path.getsize(pq) / 1e6
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "# ER v1 — CIOH + fusion on Elliptic real data\n\n"
            f"Assumption: {ASSUMPTION}\n\n"
            f"Clusters: {len(clusters)} over {len(clusters_of)} nodes "
            f"({int((sizes == 1).sum())} singletons), "
            f"size median {float(np.median(sizes)):.0f} / mean {float(sizes.mean()):.1f} "
            f"/ max {int(sizes.max())}.\n\n"
            f"Cluster precision (≥2 labeled): {pure:.4f} unweighted, "
            f"{weighted:.4f} labeled-size-weighted "
            f"({n_pure}/{n_scored} pure).\n\n"
            f"Fusion PR-AUC on {int(va.sum())} temporal-valid pairs: "
            + ", ".join(f"{k}={v['pr_auc']:.4f}" for k, v in sel["metrics"].items())
            + f". Selected: {sel['best']}.\n\n"
            f"Clusters parquet: {pq} ({size_mb:.1f} MB).\n"
        )
    metrics["clusters_parquet_mb"] = round(size_mb, 2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic_raw")
    ap.add_argument("--out-dir", default="models/er_v1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    args = ap.parse_args()
    m = run(args.data_dir, args.out_dir, args.seed, args.max_pairs)
    print(json.dumps({k: v for k, v in m.items() if k != "assumption"}, indent=2))
