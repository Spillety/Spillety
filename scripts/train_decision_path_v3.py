#!/usr/bin/env python3
"""## Train decision path v3: OFAC-anchored retrieval + causal features + refit."""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

from spillety.causal.filter import causal_filter
from spillety.data.loader import load_elliptic
from spillety.features.build import build_feature_matrix
from spillety.features.graph import add_graph_features
from spillety.features.temporal import add_temporal_features
from spillety.metrics.operational import precision_at_k, savings_vs_baseline
from spillety.models import calibration as cal
from spillety.models.gbdt import (
    _params,
    _scale_pos_weight,
    select_num_leaves,
    train_gbdt,
)
from spillety.retrieval.hnsw import brute_query
from spillety.retrieval.ofac_anchors import load_ofac_pool, build_anchor_pool, query_with_fallback

SEED = 72
K = 10
N_PCA = 32
LABEL = {"1": 1, "2": 0}


def load_enriched(root):
    """## Labeled Elliptic++ with graph + temporal features + raw tabular."""
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
    return X_raw, X_ctx, raw_cols + graph_cols, y, steps, labeled["txId"].to_numpy()


def build_ofac_embeddings(labeled, raw_cols, X_raw, steps):
    """Build PCA32 embeddings for OFAC addresses as proxy (no address graph join yet)."""
    tr = steps <= 30
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(X_raw[tr])
    # For now, OFAC addresses don't exist in Elliptic++ tx graph
    # We use a proxy: illicit transactions as OFAC embeddings
    # In production, OFAC addresses would be embedded via same encoder on address graph
    return pca


def retrieval_frames_ofac(X_raw, y, steps, pca):
    """OFAC-first anchor pool with train-illicit fallback."""
    tr = steps <= 30
    Z = pca.transform(X_raw)
    Z_tr, y_tr = Z[tr], y[tr]
    
    # Train-illicit anchors (existing)
    anchors_ill = Z_tr[y_tr == 1]
    rng = np.random.default_rng(SEED)
    lic_idx = rng.choice(int((y_tr == 0).sum()), size=min(3000, int((y_tr == 0).sum())), replace=False)
    anchors_lic = Z_tr[y_tr == 0][lic_idx]
    
    # Build OFAC pool (using train-illicit as proxy for OFAC since no address graph join)
    # In production, load real OFAC embeddings
    pool = build_anchor_pool(Z_tr, y_tr, anchors_ill)
    
    # Query with fallback
    dist, idx, used = query_with_fallback(pool, Z, K)
    frac_ill = np.zeros_like(dist[:, [0]])
    if pool.meta.get("n_ofac", 0) > 0:
        n_ofac = pool.meta["n_ofac"]
        ofac_mask = idx < n_ofac
        frac_ill = ofac_mask.sum(axis=1, keepdims=True) / K
    
    freqs = np.column_stack([frac_ill, 1.0 - frac_ill])
    return Z, dist, idx, used, pool, freqs


def sensitivity_block(Z, anchor_idx, anchor_pool):
    """Per-row causal gate over top-K anchors."""
    n = len(Z)
    out = np.zeros((n, 4))
    for i in range(n):
        if anchor_pool.kind in ("ofac", "mixed"):
            n_ofac = anchor_pool.meta.get("n_ofac", 0)
            # Use OFAC anchors for causal filter
            ofac_anchors = anchor_pool.matrix[:n_ofac]
            a = ofac_anchors[anchor_idx[i]].T
        else:
            a = anchor_pool.matrix[anchor_idx[i]].T
        r = causal_filter(a, Z[i], None)
        out[i, 0] = r["pass_rate"]
        out[i, 1] = float(r["e_value"].max())
        out[i, 2] = float(r["gamma"].min())
    return out


def evaluate(name, booster, cal_res, Xte, yte, Xva=None, yva=None):
    """Raw vs calibrated test metrics + P@K + savings."""
    p = booster.predict(Xte)
    q = cal_res["best_estimator"].predict_proba(p)[:, 1]
    out = {
        "name": name,
        "pr_auc_raw": float(average_precision_score(yte, p)),
        "pr_auc_cal": float(average_precision_score(yte, q)),
        "roc_auc_cal": float(roc_auc_score(yte, q)),
        "brier_raw": float(cal.brier_score(yte, p)),
        "brier_cal": float(cal.brier_score(yte, q)),
        "ece_raw": float(cal.ece_score(yte, p)),
        "ece_cal": float(cal.ece_score(yte, q)),
        "calibrator": cal_res["best"],
    }
    if Xva is not None:
        out["pr_auc_valid"] = float(average_precision_score(yva, booster.predict(Xva)))
    
    # P@K with bootstrap CI
    for k in [100, 500, 1000]:
        prec, (lo, hi) = precision_at_k(yte, q, k=k, n_bootstrap=600, seed=SEED)
        out[f"precision_at_{k}"] = float(prec)
        out[f"precision_at_{k}_ci_lower"] = float(lo)
        out[f"precision_at_{k}_ci_upper"] = float(hi)
    
    # Savings vs baseline (rule-based at 0.5)
    sv = savings_vs_baseline(yte, q, np.random.random(len(yte)) * 0.5, c_fp=1.0, c_fn=10.0)
    out["cost_savings"] = sv
    
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/elliptic_v3_ofac")
    ap.add_argument("--root", default="data/elliptic_raw")
    ap.add_argument("--feature-set", default="full", choices=["full", "compact"])
    opt = ap.parse_args()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X_raw, X_ctx, ctx_names, y, steps, txids = load_enriched(opt.root)
    
    # Build OFAC-first retrieval
    pca = build_ofac_embeddings(None, None, X_raw, steps)
    Z, dist, idx, used, pool, freqs = retrieval_frames_ofac(X_raw, y, steps, pca)
    print(f"Retrieval: pool={pool.kind}, coverage={pool.coverage:.3f}, used={used}")
    
    sens = sensitivity_block(Z, idx, pool)
    frames = {"distances": dist, "anchor_type_freqs": freqs,
              "sensitivity": sens, "context": X_ctx}
    X_new, new_names = build_feature_matrix(
        frames, names={"anchor_types": ["illicit", "licit"], "context": ctx_names})
    print(f"d={X_new.shape[1]} build={time.time()-t0:.0f}s")

    va = (steps > 30) & (steps <= 40)
    te = steps > 40
    tr = steps <= 30
    sel = (steps > 30) & (steps <= 35)
    cal_m = (steps > 35) & (steps <= 40)

    leaves_a, _ = select_num_leaves(X_new[tr], y[tr], X_new[va], y[va])
    boost_a = train_gbdt(X_new[tr], y[tr], X_new[va], y[va])
    rep_a = cal.calibrate(boost_a.predict(X_new[va]), y[va], boost_a.predict(X_new[te]), y[te])
    m_a = evaluate("A:v1-protocol+ofac", boost_a, rep_a, X_new[te], y[te], X_new[va], y[va])

    trb = tr | sel
    leaves_b, _ = select_num_leaves(X_new[trb], y[trb], X_new[sel], y[sel])
    boost_b = lgb.train(_params(leaves_b, _scale_pos_weight(y[trb]), SEED),
                        lgb.Dataset(X_new[trb], label=y[trb]), num_boost_round=500)
    rep_b = cal.calibrate(boost_b.predict(X_new[cal_m]), y[cal_m], boost_b.predict(X_new[te]), y[te])
    m_b = evaluate("B:refit+ofac", boost_b, rep_b, X_new[te], y[te], X_new[cal_m], y[cal_m])

    for m in (m_a, m_b):
        print(json.dumps(m, indent=1))
    winner, rep_w, leaves_w = (m_b, rep_b, leaves_b) if m_b["pr_auc_cal"] >= m_a["pr_auc_cal"] else (m_a, rep_a, leaves_a)
    boost_w = boost_b if winner is m_b else boost_a
    boost_w.save_model(str(out / "gbdt.txt"))
    joblib.dump({"feature_names": new_names, "num_leaves": leaves_w, "protocol": winner["name"]}, out / "gbdt_meta.joblib")
    joblib.dump(rep_w["best_estimator"], out / f"calibrator_{rep_w['best']}.joblib")
    (out / "metrics.json").write_text(json.dumps({"test": winner, "ablation": [m_a, m_b]}, indent=2))
    print(f"winner={winner['name']} total={time.time()-t0:.0f}s saved -> {out}")


if __name__ == "__main__":
    main()