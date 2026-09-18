#!/usr/bin/env python3
"""## Ensemble v1+v2 release: walk-forward report, tiers, artifacts."""

import glob
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from train_decision_path import build_matrix, split_temporal
from train_decision_path_v2 import load_enriched, retrieval_frames, sensitivity_block

from spillety.cost.operating import alerts_at, assign_tier, cost_at, select_tau
from spillety.features.build import build_feature_matrix
from spillety.models import calibration as cal

SEED = 72
W_V1 = 0.25


def main():
    """## Score ensemble, walk-forward curve, tier table, dump v3."""
    out = Path("models/elliptic_v3")
    out.mkdir(parents=True, exist_ok=True)

    X1, y1, s1, _ = build_matrix("data/elliptic_raw")
    (_, _), (_, _), (_, yte1) = split_temporal(X1, y1, s1)
    va_m, te_m = (s1 > 30) & (s1 <= 40), s1 > 40
    b1 = lgb.Booster(model_file="models/elliptic_v1/gbdt.txt")
    c1 = joblib.load("models/elliptic_v1/calibrator_beta.joblib")

    X_raw, X_ctx, ctx_names, y, steps = load_enriched("data/elliptic_raw")
    Z, D_ill, D_lic, I_ill, a_ill, freqs = retrieval_frames(X_raw, y, steps)
    sens = sensitivity_block(Z, I_ill, a_ill)
    X2, _ = build_feature_matrix(
        {"distances": np.column_stack([D_ill, D_lic]), "anchor_type_freqs": freqs,
         "sensitivity": sens, "context": X_ctx},
        names={"anchor_types": ["illicit", "licit"], "context": ctx_names})
    te2 = steps > 40

    q1 = c1.predict_proba(b1.predict(X1[te_m]))[:, 1]
    cj = glob.glob("models/elliptic_v2/calibrator_*.joblib")[0]
    c2 = joblib.load(cj)
    b2 = lgb.Booster(model_file="models/elliptic_v2/gbdt.txt")
    q2 = c2.predict_proba(b2.predict(X2[te2]))[:, 1]
    q = W_V1 * q1 + (1 - W_V1) * q2

    wf = []
    for t in range(31, 50):
        idx = np.where(steps == t)[0]
        yy = y[idx]
        if yy.sum() > 0 and (1 - yy).sum() > 0:
            qq1 = c1.predict_proba(b1.predict(X1[idx]))[:, 1]
            qq2 = c2.predict_proba(b2.predict(X2[idx]))[:, 1]
            wf.append({"step": t, "n": len(idx), "base_rate": float(yy.mean()),
                       "pr_auc": float(average_precision_score(yy, W_V1 * qq1 + (1 - W_V1) * qq2))})
    pd.DataFrame(wf).to_csv(out / "walkforward.csv", index=False)

    q_va = c1.predict_proba(b1.predict(X1[va_m]))[:, 1]
    tau_star = float(select_tau(y1[va_m], q_va, c_fp=1.0, c_fn=100.0))
    order = np.argsort(-q)
    prec_at = {k: float(yte1[order[:k]].mean()) for k in (100, 500, 1000)}
    tiers = [assign_tier(s, tau_star, tau_star / 2, tau_star / 4, True, 3.0, 2.0) for s in q]
    metrics = {
        "ensemble_w_v1": W_V1,
        "test": {
            "pr_auc": float(average_precision_score(yte1, q)),
            "brier": float(cal.brier_score(yte1, q)),
            "ece": float(cal.ece_score(yte1, q)),
            "precision_at": prec_at,
            "tau_star_1_100": tau_star,
            "alerts_at_tau": int(alerts_at(q, tau_star)),
            "cost_at_tau": float(cost_at(yte1, q, tau_star, 1.0, 100.0)),
            "tier_counts": pd.Series(tiers).value_counts().to_dict(),
        },
        "walkforward": wf,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"w_v1": W_V1, "v1": "models/elliptic_v1", "v2": "models/elliptic_v2"},
                out / "ensemble.joblib")
    print(json.dumps(metrics["test"], indent=1))
    print("walk-forward pr_auc:", [(w["step"], round(w["pr_auc"], 3)) for w in wf])
    print("saved ->", out)


if __name__ == "__main__":
    main()
