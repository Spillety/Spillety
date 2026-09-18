#!/usr/bin/env python3
"""## Render README metric plots from saved models (PR, reliability, walk-forward, tiers)."""

import json
from itertools import pairwise
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # E402: backend must be set before pyplot import
from sklearn.metrics import average_precision_score, precision_recall_curve

OUT = Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    """## Score test split, draw four PNGs."""
    import sys

    sys.path.insert(0, "scripts")
    from train_decision_path import build_matrix, split_temporal

    X1, y1, _s1, _ = build_matrix("data/elliptic_raw")
    (_, _), (_, _), (Xte, yte) = split_temporal(X1, y1, _s1)
    b1 = lgb.Booster(model_file="models/elliptic_v1/gbdt.txt")
    c1 = joblib.load("models/elliptic_v1/calibrator_beta.joblib")
    q1 = c1.predict_proba(b1.predict(Xte))[:, 1]

    wf = pd_wf()
    with open("models/elliptic_v3/metrics.json") as f:
        v3 = json.load(f)["test"]

    plt.figure(figsize=(7, 5))
    for q, name in ((b1.predict(Xte), "v1 raw"), (q1, "v1 calibrated")):
        prec, rec, _ = precision_recall_curve(yte, q)
        plt.step(rec, prec, where="post", label=f"{name} (AP={average_precision_score(yte, q):.3f})")
    plt.axhline(float(yte.mean()), color="gray", linestyle="--", label=f"base {yte.mean():.3f}")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("PR curve, Elliptic++ test (steps 41-49)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pr_curve.png", dpi=110)
    plt.close()

    plt.figure(figsize=(6, 6))
    bins = np.quantile(q1, np.linspace(0, 1, 11))
    cx, cy = [], []
    for lo, hi in pairwise(bins):
        m = (q1 >= lo) & (q1 <= hi)
        if m.sum():
            cx.append(float(q1[m].mean()))
            cy.append(float(yte[m].mean()))
    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    plt.plot(cx, cy, "o-", label=f"v1 calibrated (ECE={v3_ece():.3f})")
    plt.xlabel("mean predicted")
    plt.ylabel("empirical rate")
    plt.title("Reliability diagram, test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "reliability.png", dpi=110)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot([w["step"] for w in wf], [w["pr_auc"] for w in wf], "o-")
    plt.axvline(42.5, color="red", linestyle="--", label="regime change ~43")
    plt.xlabel("time step")
    plt.ylabel("PR-AUC")
    plt.title("Walk-forward PR-AUC by step (v3 ensemble)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "walkforward.png", dpi=110)
    plt.close()

    plt.figure(figsize=(7, 4))
    names = ["v1 PR-AUC 0.647", "v3 PR-AUC 0.655"]
    vals = [0.6472, v3["pr_auc"]]
    bars = plt.bar(names, vals, color=["steelblue", "darkorange"])
    plt.axhline(float(yte.mean()), color="gray", linestyle="--")
    plt.ylabel("PR-AUC")
    plt.title("Test PR-AUC vs base rate")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUT / "metrics_bars.png", dpi=110)
    plt.close()
    print("plots ->", sorted(p.name for p in OUT.glob("*.png")))


def pd_wf():
    """## Walk-forward rows from v3 metrics."""
    with open("models/elliptic_v3/metrics.json") as f:
        return json.load(f)["walkforward"]


def v3_ece():
    """## v3 ECE for the reliability caption."""
    with open("models/elliptic_v3/metrics.json") as f:
        return json.load(f)["test"]["ece"]


if __name__ == "__main__":
    main()
