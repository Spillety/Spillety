import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def walk_forward(df, splits, time_col="time_step"):
    """
    Expanding walk-forward splits: train strictly before test (no leakage).
    splits: list of (tr_s, tr_e, te_s, te_e) or (label, tr_s, tr_e, te_s, te_e) or dicts.
    Returns list of (train_df, test_df) tuples.
    """
    out = []
    for s in splits:
        if isinstance(s, dict):
            tr_s = s.get("train_start", s.get("tr_s"))
            tr_e = s.get("train_end", s.get("tr_e"))
            te_s = s.get("test_start", s.get("te_s"))
            te_e = s.get("test_end", s.get("te_e"))
        elif isinstance(s, (list, tuple)):
            if len(s) == 4:
                tr_s, tr_e, te_s, te_e = s
            elif len(s) == 5:
                _, tr_s, tr_e, te_s, te_e = s
            else:
                raise ValueError(f"split tuple must be 4 or 5 elements, got {s}")
        else:
            raise ValueError(f"unsupported split type {type(s)}")
        train = df[(df[time_col] >= tr_s) & (df[time_col] <= tr_e)].copy()
        test = df[(df[time_col] >= te_s) & (df[time_col] <= te_e)].copy()
        out.append((train, test))
    return out


def median_distance(anchors, new_anchors):
    """
    Median Euclidean distance from each new anchor to old anchors.
    anchors: (n_old, dim), new_anchors: (n_new, dim) — already standardized.
    Returns array (n_new,) of median distances per new anchor.
    """
    anchors = np.asarray(anchors)
    new_anchors = np.asarray(new_anchors)
    if len(anchors) == 0 or len(new_anchors) == 0:
        return np.array([])
    # why not normalize inside: caller standardizes via StandardScaler fit on old anchors
    # to keep embedding space consistent (train-only fit, no leakage)
    dists = cdist(new_anchors, anchors, metric="euclidean")
    return np.median(dists, axis=1)


def _cohen_d(a, b):
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * sa**2 + (n2 - 1) * sb**2) / max(1, n1 + n2 - 2))
    return (ma - mb) / pooled if pooled != 0 else 0.0


def _resolve_feat_cols(train, test, feat_cols):
    if feat_cols is not None:
        return feat_cols
    if isinstance(train, pd.DataFrame):
        cols = [c for c in train.columns if c.startswith("feat_")]
        if cols:
            return cols
        # fallback: numeric columns intersecting
        num_train = train.select_dtypes(include=[np.number]).columns.tolist()
        num_test = test.select_dtypes(include=[np.number]).columns.tolist() if isinstance(test, pd.DataFrame) else num_train
        return [c for c in num_train if c in num_test]
    # ndarray case
    n_feat = train.shape[1] if hasattr(train, "shape") else 0
    return list(range(n_feat))


def detect_drift(train, test, feat_cols=None, sample_n=5000, k=5, random_state=72):
    """
    KS + Cohen d per feature + silhouette on old clusters vs new anchors.

    why D>0.10 + |d|>0.50: p<0.05 alone fires on any shift at N>10k (FPR~0.41),
    D captures distributional distance and d captures practical effect size;
    joint threshold reduces FPR to ~0.08 at recall ~0.90 (notebook 08 §8.3.4).
    ponytail: per-feature quantile drift (KS D at 95% quantile) — alternative but single-axis.
    """
    feat_cols = _resolve_feat_cols(train, test, feat_cols)

    # extract arrays for KS / cohen
    if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):
        get = lambda df, c: df[c].values
    else:
        # ndarray with integer feat_cols
        get = lambda arr, c: np.asarray(arr)[:, c]

    ks_rows = []
    for col in feat_cols:
        tr = get(train, col)
        te = get(test, col)
        tr = np.asarray(tr).ravel()
        te = np.asarray(te).ravel()
        # filter NaN
        tr = tr[~np.isnan(tr)] if tr.dtype.kind == "f" else tr
        te = te[~np.isnan(te)] if te.dtype.kind == "f" else te
        if len(tr) == 0 or len(te) == 0:
            D, pval, d = 0.0, 1.0, 0.0
        else:
            D, pval = ks_2samp(tr, te)
            d = _cohen_d(tr, te)
        ks_rows.append({"feature": col, "D": float(D), "p": float(pval), "cohen_d": float(d), "abs_d": abs(float(d))})

    ks_df = pd.DataFrame(ks_rows).sort_values("D", ascending=False) if ks_rows else pd.DataFrame()
    if not ks_df.empty:
        # why joint: p without D/d gives false drift on large N; D>0.10 filters trivial shifts
        # Task spec says |d|>0.50 (medium effect); notebook uses 0.30 — use 0.50 per spec, keep 0.30 as ponytail
        ks_df["drift"] = (ks_df["p"] < 0.05) & (ks_df["D"] > 0.10) & (ks_df["abs_d"] > 0.50)
        n_drift = int(ks_df["drift"].sum())
    else:
        n_drift = 0

    # silhouette: old clusters vs new fallback
    sil = float("nan")
    sil_old = float("nan")
    try:
        # need DataFrame for sample; if ndarray, sample directly
        if isinstance(train, pd.DataFrame):
            # sample per spec: 5k
            n_sample = min(sample_n, len(train))
            train_sample = train.sample(n=n_sample, random_state=random_state) if len(train) > n_sample else train
            # for silhouette, use numeric feat_cols
            X_train = train_sample[feat_cols].values if feat_cols and all(isinstance(c, str) for c in feat_cols) else np.asarray(train_sample)
        else:
            X_train = np.asarray(train)
            if len(X_train) > sample_n:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(len(X_train), size=sample_n, replace=False)
                X_train = X_train[idx]

        if len(X_train) >= k and k >= 2:
            scaler = StandardScaler()
            X_old_s = scaler.fit_transform(X_train)
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            old_labels = kmeans.fit_predict(X_old_s)
            # old silhouette
            if len(np.unique(old_labels)) >= 2 and len(X_old_s) > len(np.unique(old_labels)):
                sil_old = float(silhouette_score(X_old_s, old_labels))

            # combined with test (sample test if large)
            if isinstance(test, pd.DataFrame):
                X_test = test[feat_cols].values if feat_cols and all(isinstance(c, str) for c in feat_cols) else test.values
            else:
                X_test = np.asarray(test)
            # limit test to anchors if many rows — take up to sample_n but keep at least k
            if len(X_test) > 0:
                X_test_s = scaler.transform(X_test)
                from sklearn.metrics import pairwise_distances_argmin_min

                nearest, _ = pairwise_distances_argmin_min(X_test_s, kmeans.cluster_centers_)
                X_comb = np.vstack([X_old_s, X_test_s])
                y_comb = np.concatenate([old_labels, nearest])
                if len(np.unique(y_comb)) >= 2 and len(X_comb) > len(np.unique(y_comb)):
                    sil = float(silhouette_score(X_comb, y_comb))
                else:
                    sil = sil_old
            else:
                sil = sil_old
    except Exception:
        # ponytail: fallback silhouette via subsampled pairwise if n>10k for speed
        pass

    return {
        "ks_df": ks_df,
        "n_drift": n_drift,
        "silhouette": float(sil) if not np.isnan(sil) else float("nan"),
        "silhouette_old": float(sil_old) if not np.isnan(sil_old) else float("nan"),
        "D_median": float(ks_df["D"].median()) if not ks_df.empty else float("nan"),
        "p_median": float(ks_df["p"].median()) if not ks_df.empty else float("nan"),
        "abs_d_median": float(ks_df["abs_d"].median()) if not ks_df.empty else float("nan"),
    }


def classify_drift(drift_result=None, silhouette=None, ks_drift_flag=None, sil_thresh=0.20, ks_df=None, n_drift=None):
    """
    Decision table §8.3.4:
      KS нет + sil высокий -> норма (incremental HNSW)
      KS да  + sil низкий  -> дрейф (retrain)
      KS нет + sil низкий  -> новый паттерн (expert review)
      KS да  + sil высокий -> шум (ignore)

    ks_drift_flag: bool or inferred from drift_result/n_drift/ks_df
    silhouette: float low if < sil_thresh (0.20)
    Returns label: "норма" | "дрейф" | "новый паттерн" | "шум"
    """
    # resolve ks flag
    if ks_drift_flag is None:
        if drift_result is not None and isinstance(drift_result, dict):
            ks_df = drift_result.get("ks_df", ks_df)
            n_drift = drift_result.get("n_drift", n_drift)
            if silhouette is None:
                silhouette = drift_result.get("silhouette", silhouette)
        if n_drift is not None:
            # at least 5 features with joint condition -> drift (notebook §8.3.4 uses >=5 for 165 feats;
            # adaptive threshold for low-dim synthetic: require >= min(5, n_features))
            if ks_df is not None and not ks_df.empty:
                thresh = min(5, max(1, len(ks_df) // 2)) if len(ks_df) < 10 else 5
            else:
                thresh = 5
            ks_drift_flag = int(n_drift) >= thresh
        elif ks_df is not None and not ks_df.empty:
            thresh = min(5, max(1, len(ks_df) // 2)) if len(ks_df) < 10 else 5
            if "drift" in ks_df.columns:
                ks_drift_flag = int(ks_df["drift"].sum()) >= thresh
            else:
                ks_drift_flag = bool(((ks_df["p"] < 0.05) & (ks_df["D"] > 0.10) & (ks_df["abs_d"] > 0.50)).sum() >= thresh)
        else:
            ks_drift_flag = False

    sil_val = silhouette
    if sil_val is None or (isinstance(sil_val, float) and np.isnan(sil_val)):
        sil_low = False
        sil_high = True
    else:
        sil_low = float(sil_val) < sil_thresh
        sil_high = not sil_low

    ks_flag = bool(ks_drift_flag)

    if ks_flag and sil_low:
        return "дрейф"
    if (not ks_flag) and sil_low:
        return "новый паттерн"
    if ks_flag and sil_high:
        return "шум"
    return "норма"


if __name__ == "__main__":
    # synthetic drift demo — no elliptic data required (clustered to give meaningful silhouette)
    np.random.seed(72)
    n_old, n_new, dim, k_demo = 2000, 400, 20, 5
    # clustered old anchors: 5 centers -> silhouette high for same distribution
    centers = np.random.randn(k_demo, dim) * 3
    def _sample_clustered(n):
        labels = np.random.choice(k_demo, size=n)
        return centers[labels] + np.random.randn(n, dim) * 0.6
    anchors = _sample_clustered(n_old)
    new_anchors_same = _sample_clustered(n_new)
    # drifted: far shift + diffuse -> large median distance + low silhouette (дрейф)
    drift_labels = np.random.choice(k_demo, size=n_new)
    new_anchors_drift = centers[drift_labels] + np.random.randn(n_new, dim) * 2.0 + 8.0

    dists_same = median_distance(anchors, new_anchors_same)
    dists_drift = median_distance(anchors, new_anchors_drift)
    tau_d = float(np.quantile(dists_same, 0.95))
    print(f"median_distance same median={np.median(dists_same):.3f} tau_d(95%)={tau_d:.3f}")
    print(f"median_distance drift median={np.median(dists_drift):.3f} flag={'дрейф' if np.median(dists_drift)>=tau_d else 'норма'}")

    # drift detection synthetic DataFrames
    feat_cols = [f"feat_{i}" for i in range(dim)]
    train_df = pd.DataFrame(anchors, columns=feat_cols)
    test_same_df = pd.DataFrame(new_anchors_same, columns=feat_cols)
    test_drift_df = pd.DataFrame(new_anchors_drift, columns=feat_cols)

    for name, test_df in [("same", test_same_df), ("drift", test_drift_df)]:
        res = detect_drift(train_df, test_df, feat_cols=feat_cols, sample_n=1000, k=5)
        label = classify_drift(res)
        print(f"{name}: n_drift={res['n_drift']} sil={res['silhouette']:.3f} D_med={res['D_median']:.3f} -> {label}")

    # walk_forward synthetic temporal DataFrame
    df_temporal = pd.DataFrame({
        "time_step": np.repeat(np.arange(1, 50), 100),
        "feat_0": np.random.randn(4900),
        "feat_1": np.random.randn(4900),
    })
    df_temporal["y"] = (np.random.rand(len(df_temporal)) < 0.1).astype(int)
    splits = [(1, 20, 21, 25), (1, 25, 26, 30), (1, 30, 31, 35), (1, 35, 36, 40)]
    folds = walk_forward(df_temporal, splits)
    print(f"walk_forward folds={len(folds)} sizes: {[(len(tr), len(te)) for tr, te in folds]}")
