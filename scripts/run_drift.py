## Epoch drift monitor: PCA32 distances, feature KS, regimes, retrain gates (§8/§9).

from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT / "scripts"))
_sys.path.insert(0, str(_ROOT))
from train_decision_path_v2 import load_enriched

from spillety.temporal.drift import cohen_d, is_drift, ks_stats, tau_d
from spillety.temporal.policy import (
    classify_regime,
    mann_kendall,
    regime_action,
    retrain_gate,
    walk_forward_retrain,
)

SEED = 72
BASE_MAX = 30
EPOCHS = range(31, 50)
POOL_START = 43
N_PCA = 32
N_ANCHORS = 2000
N_HIST_REF = 2000
N_BASE_FEAT = 5000
TOP_K = 10


def epoch_median_distances(Zt: np.ndarray, Za: np.ndarray) -> np.ndarray:
    """## Per-row median Euclidean distance to base anchors, vectorized."""
    return np.median(cdist(Zt, Za), axis=1)


def base_epoch_history(Z: np.ndarray, Za: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """## One median distance per base step; past-only reference without leakage."""
    return np.array(
        [float(np.median(epoch_median_distances(Z[steps == t], Za))) for t in range(1, BASE_MAX + 1)]
    )


def silhouette_proxy(Zt: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> float:
    """## Nearest-base-centroid cohesion as silhouette stand-in (0.0 on one cluster)."""
    # ponytail: centroid assignment only, upgrade to HNSW-cluster silhouette when index gates rebuild.
    lab = (np.linalg.norm(Zt - c1, axis=1) < np.linalg.norm(Zt - c0, axis=1)).astype(int)
    if len(np.unique(lab)) < 2 or len(Zt) <= 2:
        return 0.0
    return float(silhouette_score(Zt, lab))


def feature_drift_table(
    X_ctx: np.ndarray, names: list[str], base: np.ndarray, pool: np.ndarray
) -> list[dict]:
    """## KS+D+Cohen d per feature, pooled window vs base, ranked by |Cohen d|."""
    rng = np.random.default_rng(SEED)
    bi = rng.choice(base, size=min(N_BASE_FEAT, len(base)), replace=False)
    rows = []
    for j, name in enumerate(names):
        d, p = ks_stats(X_ctx[bi, j], X_ctx[pool, j])
        cd = cohen_d(X_ctx[bi, j], X_ctx[pool, j])
        rows.append(
            {
                "feature": name,
                "ks_D": float(d),
                "ks_p": float(p),
                "cohen_d": float(cd),
                "drift": bool(is_drift(p, d, cd)),
            }
        )
    rows.sort(key=lambda r: -abs(r["cohen_d"]))
    return rows


def main() -> None:
    """## Run epoch drift scan and dump models/drift_v1/report.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/elliptic_raw")
    ap.add_argument("--metrics", default="models/elliptic_v3/metrics.json")
    ap.add_argument("--out", default="models/drift_v1")
    opt = ap.parse_args()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    X_raw, X_ctx, names, y, steps = load_enriched(opt.root)
    base = np.where(steps <= BASE_MAX)[0]
    rng = np.random.default_rng(SEED)

    pca = PCA(n_components=N_PCA, random_state=SEED).fit(X_raw[base])
    Z = pca.transform(X_raw)
    anchors = rng.choice(base, size=min(N_ANCHORS, len(base)), replace=False)
    Za = Z[anchors]
    history = base_epoch_history(Z, Za, steps)
    tau = float(tau_d(history))
    base_per = epoch_median_distances(
        Z[rng.choice(base, size=min(N_HIST_REF, len(base)), replace=False)], Za
    )

    y = y.astype(int)
    c1 = Za[y[anchors] == 1].mean(axis=0)
    c0 = Za[y[anchors] == 0].mean(axis=0)

    wf = {w["step"]: w["pr_auc"] for w in json.loads(Path(opt.metrics).read_text())["walkforward"]}
    pr_first = float(wf[min(wf)])
    pr_series = np.array([wf[t] for t in sorted(wf)], dtype=float)
    mk_s, mk_p = mann_kendall(pr_series)

    epochs = []
    for t in EPOCHS:
        idx = np.where(steps == t)[0]
        per = epoch_median_distances(Z[idx], Za)
        emed = float(np.median(per))
        d, p = ks_stats(base_per, per)
        cd = cohen_d(base_per, per)
        kd = bool(is_drift(p, d, cd))
        sil = silhouette_proxy(Z[idx], c0, c1)
        regime = classify_regime(kd, sil)
        pr = float(wf[t])
        epochs.append(
            {
                "step": int(t),
                "n": len(idx),
                "median_distance": emed,
                "over_tau": bool(emed > tau),
                "ks_D": float(d),
                "ks_p": float(p),
                "cohen_d": float(cd),
                "ks_drift": kd,
                "silhouette": sil,
                "regime": regime,
                "action": regime_action(regime),
                "pr_auc": pr,
                "wf_retrain_cum": bool(walk_forward_retrain(pr_first, pr)),
            }
        )

    pool = np.where(steps >= POOL_START)[0]
    feats = feature_drift_table(X_ctx, list(names), base, pool)
    pr_last = float(wf[max(wf)])
    report = {
        "seed": SEED,
        "base": f"steps<={BASE_MAX}",
        "tau_d": tau,
        "n_anchors": len(anchors),
        "epochs": epochs,
        "walkforward": {
            "pr_first": pr_first,
            "pr_last": pr_last,
            "rel_drop": float((pr_first - pr_last) / pr_first),
            "walk_forward_retrain": bool(walk_forward_retrain(pr_first, pr_last)),
            "mann_kendall_S": int(mk_s),
            "mann_kendall_p": float(mk_p),
            "retrain_gate": bool(retrain_gate(pr_first, pr_last, mk_s, mk_p)),
        },
        "top_features": feats[:TOP_K],
        "n_drifted_features": int(sum(1 for r in feats if r["drift"])),
        "n_features": len(feats),
        "retrain_triggers": [
            f"step {e['step']}: regime={e['regime']}/action={e['action']}"
            + (", wf-drop" if e["wf_retrain_cum"] else "")
            for e in epochs
            if e["action"] == "retrain" or e["wf_retrain_cum"]
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    for e in epochs:
        print(
            f"step {e['step']}: d_med={e['median_distance']:.2f} "
            f"ks_drift={e['ks_drift']} sil={e['silhouette']:.3f} "
            f"{e['regime']}/{e['action']} pr_auc={e['pr_auc']:.3f}"
        )
    print(f"tau_d={tau:.2f} wf_retrain={report['walkforward']['walk_forward_retrain']} "
          f"mk=({mk_s}, {mk_p:.2e}) gate={report['walkforward']['retrain_gate']}")
    print("top:", [(r["feature"], round(r["cohen_d"], 2)) for r in feats[:3]])
    print("saved ->", out / "report.json")


if __name__ == "__main__":
    main()
