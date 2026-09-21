#!/usr/bin/env python3
"""## Ensemble v1+v2 release: walk-forward report, tiers, artifacts."""

import argparse
import glob
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from spillety.cost.operating import alerts_at, assign_tier, cost_at, select_tau
from spillety.features.build import build_feature_matrix
from spillety.models import calibration as cal
from spillety.data.loader import load_elliptic
from spillety.features.graph import add_graph_features
from spillety.features.temporal import add_temporal_features
from spillety.retrieval.hnsw import brute_query
from spillety.causal.filter import causal_filter
from spillety.metrics.operational import precision_at_k
from spillety.models.calibration import ece_score

SEED = 72
K = 10
N_PCA = 32
LABEL = {"1": 1, "2": 0}
W_V1 = 0.25


def load_enriched(root):
    _f, _c, edgelist, merged = load_elliptic(root)
    labeled = merged[merged["class"].isin(["1", "2"])].copy()
    y = labeled["class"].map(LABEL).astype(int).to_numpy()
    steps = labeled["time_step"].to_numpy()
    enriched = add_temporal_features(add_graph_features(labeled, edgelist), time_col="time_step")
    raw_cols = [c for c in enriched.columns if c.startswith("feat_")]
    X_raw = enriched[raw_cols].fillna(0.0).to_numpy(dtype=np.float64)
    graph_cols = [c for c in enriched.columns if c not in raw_cols
                  and c not in ("txId", "time_step", "class", "y")]
    X_ctx = enriched[raw_cols + graph_cols].fillna(0.0).to_numpy(dtype=np.float64)
    return X_raw, X_ctx, raw_cols + graph_cols, y, steps


def retrieval_frames(X_raw, y, steps):
    tr = steps <= 30
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(X_raw[tr])
    Z = pca.transform(X_raw)
    Z_tr, y_tr = Z[tr], y[tr]
    anchors_ill = Z_tr[y_tr == 1]
    rng = np.random.default_rng(SEED)
    lic_idx = rng.choice(int((y_tr == 0).sum()), size=min(3000, int((y_tr == 0).sum())), replace=False)
    anchors_lic = Z_tr[y_tr == 0][lic_idx]
    D_ill, I_ill = brute_query(anchors_ill, Z, K)
    D_lic, _ = brute_query(anchors_lic, Z, K)
    combined = np.sort(np.column_stack([D_ill, D_lic]), axis=1)[:, :K]
    frac_ill = (D_ill <= combined[:, -1:]).sum(axis=1).clip(max=K) / K
    freqs = np.column_stack([frac_ill, 1.0 - frac_ill])
    return Z, D_ill, D_lic, I_ill, anchors_ill, freqs


def sensitivity_block(Z, D_ill_idx, anchors_ill):
    n = len(Z)
    out = np.zeros((n, 4))
    for i in range(n):
        a = anchors_ill[D_ill_idx[i]].T
        r = causal_filter(a, Z[i], None)
        out[i, 0] = r["pass_rate"]
        out[i, 1] = float(r["e_value"].max())
        out[i, 2] = float(r["gamma"].min())
    return out


def build_v1_features(root):
    """Rebuild v1 features: tabular + graph + temporal (same as train_decision_path.py)."""
    _f, _c, edgelist, merged = load_elliptic(root)
    labeled = merged[merged["class"].isin(["1", "2"])].copy()
    y = labeled["class"].map(LABEL).astype(int).to_numpy()
    steps = labeled["time_step"].to_numpy()
    enriched = add_temporal_features(add_graph_features(labeled, edgelist), time_col="time_step")
    feat_cols = [c for c in enriched.columns if c.startswith("feat_") or c in (
        "degree", "in_degree", "out_degree", "pagerank", "burstiness",
        "hawkes_lambda", "time_since_last_illicit")]
    feat_cols = [c for c in feat_cols if c in enriched.columns]
    X = enriched[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    return X, y, steps, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/elliptic_raw")
    ap.add_argument("--out", default="models/elliptic_v3")
    opt = ap.parse_args()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    model_dir = f"{opt.root}/../../models"

    # --- Build v1 features (tabular + graph + temporal) ---
    X1, y, steps, feat_names = build_v1_features(opt.root)
    tr = steps <= 30
    va = (steps > 30) & (steps <= 40)
    te1 = steps > 40

    # --- Build v2 features (retrieval + causal + context) ---
    X_raw, X_ctx, ctx_names, y2, steps2 = load_enriched(opt.root)
    Z, D_ill, D_lic, I_ill, a_ill, freqs = retrieval_frames(X_raw, y2, steps2)
    sens = sensitivity_block(Z, I_ill, a_ill)
    dist_full = np.column_stack([D_ill, D_lic])
    frames = {"distances": dist_full, "anchor_type_freqs": freqs,
              "sensitivity": sens, "context": X_ctx}
    X2, ctx_names2 = build_feature_matrix(
        frames, names={"anchor_types": ["illicit", "licit"], "context": ctx_names})
    te2 = steps2 > 40

    # Load models
    b1 = lgb.Booster(model_file=f"{opt.root}/../../models/elliptic_v1/gbdt.txt")
    c1 = joblib.load(f"{opt.root}/../../models/elliptic_v1/calibrator_beta.joblib")
    q1 = c1.predict_proba(b1.predict(X1[te1]))[:, 1]

    import glob
    cj = glob.glob(f"{opt.root}/../../models/elliptic_v2/calibrator_*.joblib")[0]
    c2 = joblib.load(cj)
    b2 = lgb.Booster(model_file=f"{opt.root}/../../models/elliptic_v2/gbdt.txt")
    q2 = c2.predict_proba(b2.predict(X2[te2]))[:, 1]

    # Ensemble
    q = W_V1 * q1 + (1 - W_V1) * q2

    # Metrics on test set (steps > 40)
    yte = y[te1]
    # Align q2 to te1
    q2_aligned = q2  # already same length
    q_ens = W_V1 * q1 + (1 - W_V1) * q2_aligned

    pr_auc = average_precision_score(yte, q_ens)
    roc_auc = roc_auc_score(yte, q_ens)
    brier = brier_score_loss(yte, q_ens)
    ece = ece_score(yte, q_ens)

    # Walk-forward
    wf = []
    for step in range(31, 50):
        mask = steps == step
        if mask.sum() == 0:
            continue
        pr = average_precision_score(y[mask], q_ens[mask])
        base = y[mask].mean()
        wf.append({"step": int(step), "n": int(mask.sum()), "base_rate": float(base), "pr_auc": float(pr)})

    # Precision@K
    pk = {}
    for k in [100, 500, 1000]:
        p, (lo, hi) = precision_at_k(yte, q_ens, k=k, n_bootstrap=600, seed=72)
        pk[str(k)] = {"precision": float(p), "ci_lower": float(lo), "ci_upper": float(hi)}

    # Best F1
    best_f1, best_tau = 0, 0
    for tau in np.linspace(0, 1, 1001):
        f1 = f1_score(yte, q_ens >= tau)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    prec = precision_score(yte, q_ens >= best_tau)
    rec = recall_score(yte, q_ens >= best_tau)

    metrics = {
        "test": {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "brier": float(brier),
            "ece": float(ece),
            "precision_at": pk,
            "best_f1": float(best_f1),
            "best_f1_tau": float(best_tau),
            "precision_at_best_f1": float(prec),
            "recall_at_best_f1": float(rec),
        },
        "walkforward": wf,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {out}/metrics.json")


if __name__ == "__main__":
    main()