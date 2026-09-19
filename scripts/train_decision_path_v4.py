#!/usr/bin/env python3
"""Train v4: base features + wavelet features + focal loss GBDT.

Usage: python3 scripts/train_decision_path_v4.py [--out models/elliptic_v4] [--root data/elliptic_raw]
"""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from spillety.cost.operating import alerts_at, assign_tier, cost_at, select_tau
from spillety.data.loader import load_elliptic
from spillety.features.graph import add_graph_features
from spillety.features.temporal import add_temporal_features
from spillety.features.wavelet import wavelet_features
from spillety.models import calibration as cal
from spillety.models.focal import focal_loss_objective, threshold_moving_fbeta

SEED = 72
LABEL = {"1": 1, "2": 0}


def build_matrix(root="data/elliptic_raw"):
    _features, _classes, edgelist, merged = load_elliptic(root)
    labeled = merged[merged["class"].isin(["1", "2"])].copy()
    labeled["y"] = labeled["class"].map(LABEL).astype(int)
    enriched = add_temporal_features(add_graph_features(labeled, edgelist), time_col="time_step")
    feat_cols = [c for c in enriched.columns if c.startswith("feat_") or c in (
        "degree", "in_degree", "out_degree", "pagerank", "burstiness",
        "hawkes_lambda", "time_since_last_illicit")]
    feat_cols = [c for c in feat_cols if c in enriched.columns]
    X = enriched[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    return X, labeled["y"].to_numpy(), labeled["time_step"].to_numpy(), feat_cols


def split_temporal(X, y, steps):
    tr = steps <= 30
    va = (steps > 30) & (steps <= 40)
    te = steps > 40
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--out", default="models/elliptic_v4")
    args.add_argument("--root", default="data/elliptic_raw")
    args.add_argument("--gamma", type=float, default=2.0)
    args.add_argument("--alpha", type=float, default=0.25)
    args.add_argument("--n-estimators", type=int, default=500)
    args.add_argument("--num-leaves", type=int, default=63)
    opt = args.parse_args()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    X_raw, y, steps, names = build_matrix(opt.root)
    X = wavelet_features(X_raw)
    print(f"features: {X_raw.shape[1]} raw -> {X.shape[1]} with wavelet")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_temporal(X, y, steps)
    print(f"shapes: train={Xtr.shape} valid={Xva.shape} test={Xte.shape}")
    print(f"base rates: train={ytr.mean():.4f} valid={yva.mean():.4f} test={yte.mean():.4f}")

    params = {
        "objective": focal_loss_objective(opt.gamma, opt.alpha),
        "metric": "binary_logloss",
        "num_leaves": opt.num_leaves,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": SEED,
        "deterministic": True,
    }

    ds_train = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    ds_valid = lgb.Dataset(Xva, label=yva, free_raw_data=False)
    booster = lgb.train(
        params, ds_train,
        num_boost_round=opt.n_estimators,
        valid_sets=[ds_valid],
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )
    booster.save_model(str(out / "gbdt.txt"))

    p_va = booster.predict(Xva)
    p_te = booster.predict(Xte)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_va, yva)
    q_va = iso.predict(p_va)
    q_te = iso.predict(p_te)

    tau_star = float(select_tau(yva, q_va, c_fp=1.0, c_fn=100.0))

    order = np.argsort(-q_te)
    prec_at = {k: float(yte[order[:k]].mean()) for k in (100, 500, 1000)}
    tiers = [assign_tier(s, tau_star, tau_star / 2, tau_star / 4, True, 3.0, 2.0) for s in q_te]

    wf = []
    for t in range(31, 50):
        idx = np.where(steps == t)[0]
        if len(idx) == 0:
            continue
        yy = y[idx]
        if yy.sum() > 0 and (1 - yy).sum() > 0:
            qq = iso.predict(booster.predict(X[idx]))
            wf.append({"step": t, "n": len(idx), "base_rate": float(yy.mean()),
                        "pr_auc": float(average_precision_score(yy, qq))})

    best_thresh = threshold_moving_fbeta(yva, q_va, beta=2.0)
    pred_test = (q_te >= best_thresh).astype(int)
    from sklearn.metrics import fbeta_score
    f2 = fbeta_score(yte, pred_test, beta=2.0, zero_division=0)
    prec_t = float(pred_test[yte == 1].sum() / max(1, pred_test.sum())) if pred_test.sum() > 0 else 0
    rec_t = float(pred_test[yte == 1].sum() / max(1, yte.sum()))

    metrics = {
        "seed": SEED,
        "n_features_raw": X_raw.shape[1],
        "n_features_wavelet": X.shape[1],
        "gamma": opt.gamma,
        "alpha": opt.alpha,
        "test": {
            "pr_auc": float(average_precision_score(yte, q_te)),
            "roc_auc": float(roc_auc_score(yte, q_te)),
            "brier": float(cal.brier_score(yte, q_te)),
            "ece": float(cal.ece_score(yte, q_te)),
            "precision_at": prec_at,
            "tau_star_1_100": tau_star,
            "alerts_at_tau": int(alerts_at(q_te, tau_star)),
            "cost_at_tau": float(cost_at(yte, q_te, tau_star, 1.0, 100.0)),
            "tier_counts": pd.Series(tiers).value_counts().to_dict(),
            "f2_threshold": float(best_thresh),
            "f2_test": float(f2),
            "precision_at_f2": prec_t,
            "recall_at_f2": rec_t,
        },
        "walkforward": wf,
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(wf).to_csv(out / "walkforward.csv", index=False)
    joblib.dump(iso, out / "calibrator_isotonic.joblib")
    joblib.dump({"feature_names": names + [f"wavelet_{i}" for i in range(X_raw.shape[1])],
                 "wavelet": True, "focal_gamma": opt.gamma, "focal_alpha": opt.alpha},
                out / "gbdt_meta.joblib")

    print(json.dumps(metrics["test"], indent=2))
    print(f"walk-forward: {[(w['step'], round(w['pr_auc'], 3)) for w in wf]}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
