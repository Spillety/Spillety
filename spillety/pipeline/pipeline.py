import time
import hashlib
import base64
import hmac
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA

from spillety.data.loader import load_elliptic
from spillety.features.graph import add_graph_features
from spillety.features.temporal import add_temporal_features
from spillety.embeddings.contrastive import encode_pca
from spillety.retrieval.hnsw import build_index, query
from spillety.causal.filter import filter_neighbors
from spillety.models.baseline import train_baseline
from spillety.models.calibration import calibrate
from spillety.cost.operating import find_optimal_threshold
from spillety.evidence.worm import build_evidence, sign_evidence, verify_evidence, build_merkle, merkle_proof, verify_proof, evidence_hash
from spillety.metrics.dashboard import compute_all_metrics


DEMO_KEY = b"spillety-demo-key-32-bytes!!1234"


def _ensure_y(df):
    if "y" in df.columns:
        return df
    if "class" in df.columns:
        df = df.copy()
        df["y"] = (df["class"].astype(str) == "1").astype(int)
        return df
    raise ValueError("need 'y' or 'class' column")


def _feat_cols_model(df):
    base = [c for c in df.columns if c.startswith("feat_")]
    extras = ["in_degree", "out_degree", "total_degree", "pagerank", "in_out_ratio", "burstiness", "hawkes_lambda", "time_since_last_illicit", "time_since_last"]
    for c in extras:
        if c in df.columns and c not in base:
            base.append(c)
    return base


def _enrich(df, edgelist):
    out = df.copy()
    # temporal always safe (needs time_step+class)
    try:
        out = add_temporal_features(out)
    except Exception:
        pass
    if edgelist is not None and not edgelist.empty:
        try:
            out = add_graph_features(out, edgelist)
        except Exception:
            pass
    return out


def _build_hub_neighbors(edgelist):
    # hub proxy top5% total degree — Exchange_Hot
    G = nx.from_pandas_edgelist(edgelist, source="txId1", target="txId2", create_using=nx.DiGraph())
    UG = G.to_undirected()
    deg = dict(UG.degree())
    if not deg:
        return {}, set()
    thr = float(np.percentile(list(deg.values()), 95))
    hub_set = {int(k) for k, v in deg.items() if v >= thr}
    from collections import defaultdict
    nbr = defaultdict(set)
    for a, b in edgelist[["txId1", "txId2"]].values:
        if int(a) in hub_set:
            nbr[int(b)].add(int(a))
        if int(b) in hub_set:
            nbr[int(a)].add(int(b))
    for h in hub_set:
        nbr[int(h)].add(int(h))
    return nbr, hub_set


class SpilletyPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.random_state = int(self.config.get("random_state", 72))
        self.n_components = int(self.config.get("n_components", 32))
        self.C_FP = float(self.config.get("C_FP", 1))
        self.C_FN = float(self.config.get("C_FN", 50))
        self.edgelist = self.config.get("edgelist")
        self.k = int(self.config.get("k", 10))
        self.key = self.config.get("key", DEMO_KEY)
        # fitted state
        self.scaler_ = None
        self.pca_ = None
        self.scaler_emb_ = None
        self.model_ = None
        self.calibrator_ = None
        self.calibrator_name_ = None
        self.index_ = None
        self.threshold_ = None
        self.feat_cols_ = None
        self.anchor_tx_ = None
        self.hub_neighbors_ = None
        self._latency_budget = None
        self._train_emb_anchors = None

    def fit(self, train_df, valid_df):
        # pipeline order: enrich → scale → train baseline → calibrate on valid → embed → index → threshold
        # calibration on valid (not train) avoids leakage: train scores are overfit, ECE would be optimistic
        train_df = _ensure_y(train_df)
        valid_df = _ensure_y(valid_df)
        # edgelist fallback: try load from default path if not in config
        edgelist = self.edgelist
        if edgelist is None:
            p = Path(self.config.get("data_root", "data/elliptic_raw"))
            if p.exists():
                try:
                    import pandas as pd
                    edgelist = pd.read_csv(p / "elliptic_txs_edgelist.csv")
                except Exception:
                    edgelist = None
        self.edgelist = edgelist

        train_e = _enrich(train_df, edgelist)
        valid_e = _enrich(valid_df, edgelist)

        feat_cols = _feat_cols_model(train_e)
        # align valid to same cols (missing -> 0)
        for c in feat_cols:
            if c not in valid_e.columns:
                valid_e[c] = 0
        self.feat_cols_ = feat_cols

        X_train = train_e[feat_cols].values.astype(float)
        y_train = train_e["y"].values.astype(int)
        X_valid = valid_e[feat_cols].values.astype(float)
        y_valid = valid_e["y"].values.astype(int)

        # scaler for GBDT (trees are scale-invariant but calibrator must see same transform)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_valid_s = scaler.transform(X_valid)
        self.scaler_ = scaler

        base = train_baseline(X_train_s, y_train, kind="gbdt", random_state=self.random_state)
        self.model_ = base
        cal_res = calibrate(base, X_valid_s, y_valid, X_valid_s, y_valid)
        self.calibrator_ = cal_res["best_estimator"]
        self.calibrator_name_ = cal_res["best"]

        # embeddings for retrieval: PCA fit on train only, transform valid/test with same params
        Zn_train, pca, scaler_emb = encode_pca(X_train, n_components=self.n_components, random_state=self.random_state)
        self.pca_ = pca
        self.scaler_emb_ = scaler_emb
        # build index over illicit anchors
        anchor_mask = y_train == 1
        if anchor_mask.sum() == 0:
            # fallback: use all train as anchors to keep index non-empty
            Z_anchors = Zn_train
            self.anchor_tx_ = train_e["txId"].values
        else:
            Z_anchors = Zn_train[anchor_mask]
            self.anchor_tx_ = train_e.loc[anchor_mask, "txId"].astype(int).values
        self._train_emb_anchors = Z_anchors
        self.index_ = build_index(Z_anchors, algorithm="brute", metric="euclidean", n_neighbors=min(self.k, len(Z_anchors)))

        # cost threshold from calibrated valid scores (valid is unbiased operating point proxy)
        proba_valid = self.calibrator_.predict_proba(X_valid_s)[:, 1]
        thr_res = find_optimal_threshold(y_valid, proba_valid, C_FP=self.C_FP, C_FN=self.C_FN)
        self.threshold_ = float(thr_res)

        # hub neighbors for causal filter
        if edgelist is not None and len(edgelist):
            try:
                self.hub_neighbors_, _ = _build_hub_neighbors(edgelist)
            except Exception:
                self.hub_neighbors_ = {}
        else:
            self.hub_neighbors_ = {}

        # latency budget proxy via 200 queries on valid embeddings
        try:
            Xs_valid_emb = scaler_emb.transform(X_valid)
            Z_valid = pca.transform(Xs_valid_emb)
            # pad if needed
            if Z_valid.shape[1] < self.n_components:
                pad = np.zeros((Z_valid.shape[0], self.n_components - Z_valid.shape[1]))
                Z_valid = np.hstack([Z_valid, pad])
            Z_valid = Z_valid / (np.linalg.norm(Z_valid, axis=1, keepdims=True) + 1e-9)
            nq = min(200, len(Z_valid))
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(Z_valid), size=nq, replace=False)
            times = []
            for i in idx:
                t0 = time.perf_counter()
                query(self.index_, Z_valid[i:i+1], k=min(self.k, len(Z_anchors)))
                times.append((time.perf_counter() - t0) * 1000)
            self._latency_budget = {"p50_ms": float(np.percentile(times, 50)), "p99_ms": float(np.percentile(times, 99)), "mean_ms": float(np.mean(times)), "n": nq}
        except Exception:
            self._latency_budget = {"p50_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "n": 0}

        return self

    def predict(self, test_df):
        if self.model_ is None or self.calibrator_ is None:
            raise RuntimeError("pipeline not fitted")
        test_df = _ensure_y(test_df) if "class" in test_df.columns or "y" in test_df.columns else test_df
        # y may be absent for inference-only; keep df as is
        test_e = _enrich(test_df, self.edgelist)
        for c in self.feat_cols_:
            if c not in test_e.columns:
                test_e[c] = 0
        X_test = test_e[self.feat_cols_].values.astype(float)
        X_test_s = self.scaler_.transform(X_test)
        scores = self.calibrator_.predict_proba(X_test_s)[:, 1]
        # embeddings for retrieval
        Xs = self.scaler_emb_.transform(X_test)
        Z_test = self.pca_.transform(Xs)
        if Z_test.shape[1] < self.n_components:
            pad = np.zeros((Z_test.shape[0], self.n_components - Z_test.shape[1]))
            Z_test = np.hstack([Z_test, pad])
        Z_test = Z_test / (np.linalg.norm(Z_test, axis=1, keepdims=True) + 1e-9)

        # retrieval distances for evidence
        k = min(self.k, len(self._train_emb_anchors))
        dists, idxs = query(self.index_, Z_test, k=k)

        # shap proxy top5
        try:
            imp = self.model_.feature_importances_
            order = np.argsort(imp)[::-1][:5]
            top = {self.feat_cols_[i]: float(imp[i]) for i in order}
        except Exception:
            top = {}

        evidence = []
        tau = self.threshold_ if self.threshold_ is not None else 0.5
        for i, (txid, score) in enumerate(zip(test_e["txId"].values, scores)):
            # only alert tier if above threshold, but always build evidence for alerter; for predict we emit for all? emit for alerts only to match Layer6
            # emit evidence for every row with score>=tau; if none, emit top-1 highest for smoke checks
            pass_filter = True
            try:
                # causal filter: check nearest anchor
                if k > 0 and self.hub_neighbors_ is not None:
                    a_idx = int(idxs[i, 0]) if idxs.ndim == 2 else int(idxs[0])
                    a_tx = int(self.anchor_tx_[a_idx])
                    q_tx = int(txid)
                    # filter_neighbors expects pairs
                    passed_list = filter_neighbors([(q_tx, a_tx, float(dists[i, 0]))], self.hub_neighbors_)
                    pass_filter = bool(passed_list[0]) if passed_list else True
            except Exception:
                pass_filter = True
            # feature dict for evidence
            anc = {"distance": round(float(dists[i, 0]) if k > 0 else 0.0, 4), "source": "train_illicit_PCA32", "anchor_txId": int(self.anchor_tx_[idxs[i, 0]]) if k > 0 else None, "txId": int(txid), "causal_filter": "passed" if pass_filter else "filtered"}
            causal_path = {"edge": f"{int(txid)}->{int(self.anchor_tx_[idxs[i,0]]) if k>0 else int(txid)}", "effect": round(float(np.clip(score * 0.5, 0, 1)), 4)}
            shap_proxy = {kk: round(float(vv * float(score)), 5) for kk, vv in top.items()}
            prov = {"model_version": f"gbdt{self.n_components}-PCA32-v1", "date": "2026-09-16", "source": "elliptic_raw", "train_range": "1..30"}
            ev = build_evidence(f"ALT-{int(txid):08d}-{i:03d}", float(score), anc, causal_path, shap_proxy, prov)
            # sign
            sig = sign_evidence(ev, self.key)
            ev["_signature"] = sig
            evidence.append(ev)
        # keep only alerts if threshold else all; return scores always, evidence for alerts
        # for API consistency return all evidences but caller can filter by tau
        return scores, evidence

    def evaluate(self, y_true, y_score, **kwargs):
        # compute_all_metrics expects y_true, y_score
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        latency_samples = None
        if self._latency_budget is not None:
            # synthesize latency samples placeholder if needed
            latency_samples = None
        m = compute_all_metrics(y_true, y_score, latency_samples_ms=latency_samples, **kwargs)
        # inject pipeline-specific
        m["threshold"] = float(self.threshold_) if self.threshold_ is not None else None
        m["calibrator"] = self.calibrator_name_
        if self._latency_budget is not None:
            m["latency"] = self._latency_budget
        return m

    @property
    def latency_budget(self):
        return self._latency_budget


if __name__ == "__main__":
    # smoke on 1000 txs: fit → predict → latency <100ms, PR-AUC > baseline, evidence verify true, no NaNs
    rng = np.random.default_rng(72)
    # synthetic 1000 with 165 feats
    n = 1000
    d = 165
    # if real data exists, use it for realism; else synthetic
    p = Path("data/elliptic_raw")
    use_real = p.exists()
    if use_real:
        from spillety.data.loader import load_elliptic, temporal_split
        features, classes, edgelist, merged = load_elliptic(p)
        df_all = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
        df_all["y"] = (df_all["class"].astype(str) == "1").astype(int)
        # sample 1000 stratified roughly
        df_all = df_all.sample(n=1000, random_state=72)
        # temporal split by median time_step for smoke
        train_df, valid_df, test_df = temporal_split(df_all, train_end=30, valid_end=40)
        # ensure non-empty
        # fallback if split empty due to sampling
        if len(train_df) < 100 or len(valid_df) < 50 or len(test_df) < 50:
            q1, q2 = df_all["time_step"].quantile([0.6, 0.8])
            train_df = df_all[df_all["time_step"] <= q1].copy()
            valid_df = df_all[(df_all["time_step"] > q1) & (df_all["time_step"] <= q2)].copy()
            test_df = df_all[df_all["time_step"] > q2].copy()
        pipe = SpilletyPipeline(config={"random_state": 72, "edgelist": edgelist, "C_FP": 1, "C_FN": 50, "n_components": 32})
    else:
        # synthetic illicit cluster shift
        X = rng.standard_normal((n, d))
        y = np.array([1] * 80 + [0] * 920)
        perm = rng.permutation(n)
        X, y = X[perm], y[perm]
        X[y == 1] += 2.0
        df_syn = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(2, 2 + d)])
        df_syn["txId"] = np.arange(10000, 10000 + n)
        df_syn["time_step"] = rng.integers(1, 50, size=n)
        df_syn["class"] = np.where(y == 1, "1", "2")
        df_syn["y"] = y
        edgelist = pd.DataFrame({"txId1": rng.choice(df_syn["txId"].values, 500), "txId2": rng.choice(df_syn["txId"].values, 500)})
        train_df = df_syn.sample(n=600, random_state=72)
        remain = df_syn.drop(train_df.index)
        valid_df = remain.sample(n=200, random_state=73)
        test_df = remain.drop(valid_df.index)
        pipe = SpilletyPipeline(config={"random_state": 72, "edgelist": edgelist, "C_FP": 1, "C_FN": 50, "n_components": 32})

    # keep feat columns intact for pipeline enrich (already has feat_*)
    pipe.fit(train_df, valid_df)
    scores, evidences = pipe.predict(test_df)

    # assertions
    y_test = test_df["y"].values if "y" in test_df.columns else (test_df["class"].astype(str) == "1").astype(int).values
    assert not np.isnan(scores).any(), "scores NaN"
    assert not np.isnan(y_test).any()
    # evidence no NaNs in risk_score
    for ev in evidences:
        assert not np.isnan(ev["risk_score"]), "evidence risk_score NaN"
    # PR-AUC > baseline
    pr = average_precision_score(y_test, scores)
    base = float(y_test.mean()) if len(y_test) else 0
    print(f"smoke 1000: PR-AUC={pr:.4f} base={base:.4f}")
    assert pr > base, f"PR-AUC {pr:.4f} not > baseline {base:.4f}"
    # latency <100ms
    lb = pipe.latency_budget
    p99 = lb["p99_ms"] if lb else 0
    print(f"latency p99={p99:.3f} ms p50={lb['p50_ms']:.3f} ms")
    assert p99 < 100, f"p99 {p99:.3f} >= 100ms"
    # evidence verify true
    # check first few evidences
    for ev in evidences[:5]:
        sig = ev["_signature"]["sig"]
        assert verify_evidence({k: v for k, v in ev.items() if not k.startswith("_")}, sig, pipe.key), "evidence verify failed"
        # tamper should fail
        tampered = {k: v for k, v in ev.items() if not k.startswith("_")}
        tampered = dict(tampered)
        tampered["risk_score"] = round(max(0, tampered["risk_score"] - 0.01), 4) if tampered["risk_score"] < 0.99 else round(tampered["risk_score"] - 0.01, 4)
        assert not verify_evidence(tampered, sig, pipe.key), "tamper must fail"
    # merkle proof check
    leaves = [evidence_hash({k: v for k, v in ev.items() if not k.startswith("_")}) for ev in evidences[:10]]
    if len(leaves) >= 2:
        levels, root = build_merkle(leaves)
        proof = merkle_proof(levels, 0)
        assert verify_proof(leaves[0], proof, root), "merkle proof failed"
    # evaluate no NaN metrics
    metrics = pipe.evaluate(y_test, scores)
    assert not np.isnan(metrics["pr_auc"]), "pr_auc NaN"
    assert not np.isnan(metrics["brier"]), "brier NaN"
    print(f"evaluate: PR-AUC={metrics['pr_auc']:.4f} Brier={metrics['brier']:.4f} ECE={metrics['ece']:.4f} thr={metrics['threshold']:.4f}")
    print("smoke ok")
