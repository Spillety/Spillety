#!/usr/bin/env python3
"""## Train GBDT decision path on Elliptic++ and store model + metrics.

Usage: python3 scripts/train_decision_path.py [--out models/elliptic_v1]
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from spillety.data.loader import load_elliptic
from spillety.features.graph import add_graph_features
from spillety.features.temporal import add_temporal_features
from spillety.models import calibration as cal
from spillety.models.gbdt import select_num_leaves, train_gbdt

SEED = 72
LABEL = {"1": 1, "2": 0}


def build_matrix(root="data/elliptic_raw"):
    """## Load Elliptic++, label licit/illicit, add graph + temporal features."""
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
    """## Temporal split 1..30 / 31..40 / 41..49 without shuffling."""
    tr = steps <= 30
    va = (steps > 30) & (steps <= 40)
    te = steps > 40
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def main():
    """## Fit select-train-calibrate-evaluate, dump booster + metrics JSON."""
    args = argparse.ArgumentParser()
    args.add_argument("--out", default="models/elliptic_v1")
    args.add_argument("--root", default="data/elliptic_raw")
    opt = args.parse_args()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, steps, names = build_matrix(opt.root)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_temporal(X, y, steps)
    print(f"shapes train/valid/test: {Xtr.shape} {Xva.shape} {Xte.shape}, base rates: "
          f"{ytr.mean():.4f} {yva.mean():.4f} {yte.mean():.4f}, d={X.shape[1]}")

    leaves, pr_table = select_num_leaves(Xtr, ytr, Xva, yva)
    print(f"num_leaves: {leaves} {pr_table}")
    booster = train_gbdt(Xtr, ytr, Xva, yva)
    booster.save_model(str(out / "gbdt.txt"))
    joblib.dump({"feature_names": names, "num_leaves": leaves}, out / "gbdt_meta.joblib")

    p_va = booster.predict(Xva)
    p_te = booster.predict(Xte)
    report = cal.calibrate(p_va, yva, p_te, yte)
    joblib.dump(report["best_estimator"], out / f"calibrator_{report['best']}.joblib")

    q = report["best_estimator"].predict_proba(p_te)[:, 1]
    metrics = {
        "seed": SEED,
        "n_features": X.shape[1],
        "num_leaves": leaves,
        "pr_auc_by_leaves": pr_table,
        "test": {
            "pr_auc_raw": float(average_precision_score(yte, p_te)),
            "pr_auc_cal": float(average_precision_score(yte, q)),
            "roc_auc_cal": float(roc_auc_score(yte, q)),
            "brier_raw": float(cal.brier_score(yte, p_te)),
            "brier_cal": float(cal.brier_score(yte, q)),
            "brier_trivial": float(cal.brier_score(yte, np.full_like(q, yte.mean()))),
            "ece_raw": float(cal.ece_score(yte, p_te)),
            "ece_cal": float(cal.ece_score(yte, q)),
        },
        "calibration": {
            "best": report["best"],
            "metrics": {k: {m: float(v) for m, v in d.items()} for k, d in report["metrics"].items()},
            "bootstrap": {k: float(v) for k, v in report["bootstrap"].items()},
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test"], indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
