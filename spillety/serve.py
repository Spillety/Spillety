"""FastAPI service for Spillety demo stand."""

import json
import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from spillety.pipeline.pipeline import SpilletyPipeline
from spillety.data.loader import load_elliptic, temporal_split
from spillety.evidence.worm import build_merkle, merkle_proof, verify_proof, evidence_hash

app = FastAPI(title="Spillety API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for last run
_last_result: dict[str, Any] = {}
_data_ready = False
_features = None
_classes = None
_edgelist = None
_merged = None


def _ensure_data():
    global _data_ready, _features, _classes, _edgelist, _merged
    if _data_ready:
        return
    root = Path("data/elliptic_raw")
    if not root.exists():
        root = Path(__file__).resolve().parents[1] / "data" / "elliptic_raw"
    if root.exists():
        _features, _classes, _edgelist, _merged = load_elliptic(root)
        _data_ready = True


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/metrics")
def get_metrics(
    c_fp: float = Query(1.0, ge=0.1, le=1000),
    c_fn: float = Query(50.0, ge=0.1, le=10000),
    tau: float = Query(0.5, ge=0.0, le=1.0),
    window: int = Query(14, ge=1, le=90),
    n_components: int = Query(32, ge=2, le=256),
    k: int = Query(10, ge=1, le=100),
):
    """Run pipeline on a stratified sample and return metrics + evidence."""
    _ensure_data()
    if not _data_ready:
        # deterministic mock when data unavailable
        c_ratio = c_fn / max(c_fp, 1)
        pr_auc = min(0.99, 0.62 + 0.08 * np.log1p(c_ratio) + 0.02 * np.log(window / 7))
        ece = max(0.01, 0.12 - 0.02 * (n_components / 32) + 0.01 * (tau / 0.5))
        latency = 40 + window * 2 + n_components * 1.2
        return {
            "pr_auc": round(float(pr_auc), 3),
            "ece": round(float(ece), 3),
            "latency_ms": int(latency),
            "threshold": round(tau, 3),
            "source": "mock",
        }

    df_all = _merged[_merged["class"].astype(str).isin(["1", "2"])].copy()
    df_all["y"] = (df_all["class"].astype(str) == "1").astype(int)
    # stratified sample up to 5000 for speed
    n_sample = min(5000, len(df_all))
    n_per_class = max(100, n_sample // 2)
    df_0 = df_all[df_all["y"] == 0].sample(min(len(df_all[df_all["y"] == 0]), n_per_class), random_state=72)
    df_1 = df_all[df_all["y"] == 1].sample(min(len(df_all[df_all["y"] == 1]), n_per_class), random_state=72)
    df_sample = pd.concat([df_0, df_1]).sample(frac=1, random_state=72).reset_index(drop=True)
    if len(df_sample) < 200:
        df_sample = df_all.sample(n=min(n_sample, len(df_all)), random_state=72).reset_index(drop=True)

    # temporal-ish split by quantiles
    q1, q2 = df_sample["time_step"].quantile([0.6, 0.8])
    train_df = df_sample[df_sample["time_step"] <= q1].copy()
    valid_df = df_sample[(df_sample["time_step"] > q1) & (df_sample["time_step"] <= q2)].copy()
    test_df = df_sample[df_sample["time_step"] > q2].copy()

    if len(train_df) < 50 or len(valid_df) < 20 or len(test_df) < 20:
        # fallback random split
        shuffled = df_sample.sample(frac=1, random_state=72)
        train_df = shuffled.iloc[: int(len(shuffled) * 0.6)].copy()
        valid_df = shuffled.iloc[int(len(shuffled) * 0.6) : int(len(shuffled) * 0.8)].copy()
        test_df = shuffled.iloc[int(len(shuffled) * 0.8) :].copy()

    pipe = SpilletyPipeline(
        config={
            "random_state": 72,
            "edgelist": _edgelist,
            "C_FP": c_fp,
            "C_FN": c_fn,
            "n_components": n_components,
            "k": k,
        }
    )
    pipe.fit(train_df, valid_df)
    scores, evidence = pipe.predict(test_df)
    y_test = test_df["y"].values.astype(int)
    metrics = pipe.evaluate(y_test, scores)

    # Build merkle root from evidence hashes
    leaves = [evidence_hash({kk: vv for kk, vv in ev.items() if not kk.startswith("_")}) for ev in evidence[:100]]
    merkle_root = None
    if len(leaves) >= 2:
        _, merkle_root = build_merkle(leaves)

    result = {
        "pr_auc": round(float(metrics["pr_auc"]), 4),
        "roc_auc": round(float(metrics.get("roc_auc", 0.0)), 4),
        "brier": round(float(metrics["brier"]), 4),
        "ece": round(float(metrics["ece"]), 4),
        "threshold": round(float(metrics["threshold"]), 4) if metrics.get("threshold") is not None else None,
        "latency_ms": round(float(metrics.get("latency", {}).get("p99_ms", 0.0)), 2) if metrics.get("latency") else None,
        "calibrator": metrics.get("calibrator"),
        "source": "live",
        "n_test": int(len(test_df)),
        "merkle_root": merkle_root,
    }
    global _last_result
    _last_result = {"metrics": result, "scores": scores.tolist(), "evidence": evidence, "test_df": test_df}
    return result


@app.get("/api/plot/{tab}")
def get_plot(tab: str):
    """Return JSON data for frontend visualizations."""
    global _last_result
    if not _last_result:
        return JSONResponse({"error": "run /api/metrics first"}, status_code=400)

    scores = np.array(_last_result["scores"])
    test_df = _last_result["test_df"]
    y_test = test_df["y"].values.astype(int)
    evidence = _last_result["evidence"]

    tab = tab.lower()
    if tab in ("baseline", "graph"):
        # Return graph nodes/edges from edgelist sample
        _ensure_data()
        if _data_ready and _edgelist is not None and len(_edgelist):
            sample = _edgelist.sample(min(200, len(_edgelist)), random_state=72)
            nodes = list(set(sample["txId1"].tolist() + sample["txId2"].tolist()))
            edges = sample[["txId1", "txId2"]].rename(columns={"txId1": "source", "txId2": "target"}).to_dict(orient="records")
            return {"nodes": [{"id": int(n)} for n in nodes], "edges": edges}
        return {"nodes": [], "edges": []}

    if tab in ("embeddings", "tsne", "umap"):
        # Return 2D PCA projection of test features for scatter plot
        feat_cols = [c for c in test_df.columns if c.startswith("feat_")]
        if len(feat_cols) >= 2:
            X = test_df[feat_cols].values.astype(float)
            from sklearn.decomposition import PCA
            pca2 = PCA(n_components=2, random_state=72)
            Z = pca2.fit_transform(X)
            return {
                "points": [
                    {"x": float(Z[i, 0]), "y": float(Z[i, 1]), "label": int(y_test[i]), "txId": int(test_df["txId"].iloc[i])}
                    for i in range(len(Z))
                ]
            }
        return {"points": []}

    if tab in ("pr", "prcurve", "precision-recall"):
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_test, scores)
        return {
            "curve": [
                {"precision": float(p), "recall": float(r)}
                for p, r in zip(precision.tolist(), recall.tolist())
            ]
        }

    if tab in ("reliability", "calibration"):
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_test, scores, n_bins=10, strategy="uniform")
        return {
            "curve": [
                {"prob_true": float(t), "prob_pred": float(p)}
                for t, p in zip(prob_true.tolist(), prob_pred.tolist())
            ]
        }

    if tab == "metrics":
        return _last_result["metrics"]

    return JSONResponse({"error": f"unknown tab {tab}"}, status_code=404)


@app.get("/api/evidence")
def get_evidence(limit: int = Query(50, ge=1, le=500)):
    global _last_result
    if not _last_result:
        return JSONResponse({"error": "run /api/metrics first"}, status_code=400)
    evs = _last_result["evidence"][:limit]
    return {"evidence": [{k: v for k, v in ev.items() if not k.startswith("_")} for ev in evs]}


@app.get("/api/worm")
def get_worm():
    global _last_result
    if not _last_result:
        return JSONResponse({"error": "run /api/metrics first"}, status_code=400)
    evidence = _last_result["evidence"]
    leaves = [evidence_hash({kk: vv for kk, vv in ev.items() if not kk.startswith("_")}) for ev in evidence]
    if len(leaves) < 2:
        return {"root": None, "count": len(leaves), "proofs": []}
    levels, root = build_merkle(leaves)
    proofs = []
    for i in range(min(5, len(leaves))):
        proof = merkle_proof(levels, i)
        proofs.append({"leaf_index": i, "proof": proof, "verified": verify_proof(leaves[i], proof, root)})
    return {"root": root, "count": len(leaves), "proofs": proofs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("spillety.serve:app", host="0.0.0.0", port=8000, reload=False)
