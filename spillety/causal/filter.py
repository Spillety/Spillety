import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr

# ponytail: PC-algorithm / NOTEARS for full DAG — here only observed proxy DAG, latent Owner/Intent via sensitivity


def _has_hub(tid, hub_nodes):
    if isinstance(hub_nodes, dict):
        s = hub_nodes.get(int(tid), set())
        # dict values may be set or int flag
        if isinstance(s, (set, list, tuple)):
            return len(s) > 0
        return bool(s)
    # set / list
    try:
        return int(tid) in hub_nodes
    except Exception:
        return False


def is_spurious(query, anchor, hub_nodes, epsilon=5.0, feat_dict=None, d_qa=None):
    """
    DAG filter predicate: confounded ∧ distance_explained.
    confounded = q_has_hub ∧ a_has_hub
    distance_explained = |d(q,a) - d(c,a)| < epsilon  where c is hub neighbor of query
    epsilon=5.0 tuned to pass_rate ~0.85-0.90 on Elliptic (notebook 09) — larger epsilon filters more, smaller passes more
    """
    q_has = _has_hub(query, hub_nodes)
    a_has = _has_hub(anchor, hub_nodes)
    confounded = bool(q_has and a_has)
    if not confounded:
        return False
    # if no feature info, distance_explained defaults to True (conservative: confounded → spurious)
    # when feat_dict and d_qa provided we compute true explained
    if feat_dict is not None and d_qa is not None:
        # find hub neighbor candidate c
        c = None
        if isinstance(hub_nodes, dict):
            cands = hub_nodes.get(int(query), set())
            if cands:
                # deterministic min for reproducibility (seed 72)
                try:
                    c = min(cands)
                except TypeError:
                    c = next(iter(cands))
        if c is not None and c in feat_dict and int(anchor) in feat_dict:
            try:
                d_ca = float(np.linalg.norm(np.asarray(feat_dict[c]) - np.asarray(feat_dict[int(anchor)])))
                return abs(float(d_qa) - d_ca) < float(epsilon)
            except Exception:
                return True
        # fallback if no candidate feature
        return True
    # also support d_qa alone with epsilon as generic threshold: if d_qa < epsilon consider explained (proxy)
    if d_qa is not None:
        return abs(float(d_qa)) < float(epsilon) or True  # keep conservative
    return True


def filter_neighbors(queries, anchors=None, hub_nodes=None, epsilon=5.0, feat_dict=None, distances=None):
    """
    Filter list of candidate pairs.
    queries, anchors: parallel iterables of ids (or pairs). hub_nodes: dict/set.
    If queries is list of (qid, aid) pairs and anchors is hub_nodes, handles 2-arg form.
    Returns list of bool passed = not is_spurious.
    """
    # handle overloaded call filter_neighbors(pairs, hub_nodes) where pairs is list of tuples
    if anchors is not None and isinstance(anchors, (dict, set)) and hub_nodes is None:
        hub_nodes = anchors
        anchors = None
    # if single list of pairs
    if anchors is None:
        # queries is iterable of (qid, aid) or (qid, aid, d)
        pairs = list(queries)
        passed = []
        for item in pairs:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                q, a = item[0], item[1]
                d = item[2] if len(item) > 2 else None
                # distances override
                passed.append(not is_spurious(q, a, hub_nodes, epsilon=epsilon, feat_dict=feat_dict, d_qa=d))
            else:
                passed.append(True)
        return passed
    # parallel arrays
    queries = list(queries)
    anchors = list(anchors)
    if distances is not None:
        distances = list(distances)
    else:
        distances = [None] * len(queries)
    if len(queries) != len(anchors):
        raise ValueError(f"queries and anchors length mismatch {len(queries)} vs {len(anchors)}")
    out = []
    for q, a, d in zip(queries, anchors, distances):
        out.append(not is_spurious(q, a, hub_nodes, epsilon=epsilon, feat_dict=feat_dict, d_qa=d))
    return out


def sensitivity_evalue(effect):
    """
    E-value proxy: RR = 1+effect (clipped), E = RR + sqrt(RR*(RR-1))
    effect = Δ/σ per alert (notebook 09). Tier1 threshold E>2.0
    """
    eff = float(np.clip(float(effect), 0, 5))
    rr = 1.0 + eff
    if rr <= 1:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1)))


# alias
evalue = sensitivity_evalue


def falsification_test(df, confounder="Exchange_Hot", outcome="y", control="_deg"):
    """
    Falsification: `confounder ⟂ outcome | control`.
    Stratify by control quantiles, χ² per stratum + partial correlation via residuals.
    Returns dict with per-stratum results and partial correlation.
    """
    if confounder not in df.columns or outcome not in df.columns or control not in df.columns:
        raise ValueError(f"missing columns: need {confounder},{outcome},{control}")
    # bins straddle p95 thr so both hub/non-hub appear in 2 strata (notebook 09)
    # use quantiles but fallback to fixed bins [-1,4,10,500] as in notebook
    try:
        bins = [-1, 4, 10, 500]
        deg_q = pd.cut(df[control], bins=bins, labels=["low", "mid", "high"], include_lowest=True)
    except Exception:
        deg_q = pd.qcut(df[control], q=3, labels=["low", "mid", "high"], duplicates="drop")
    results = []
    for grp in deg_q.cat.categories:
        mask = deg_q == grp
        sub = df.loc[mask]
        if sub[confounder].nunique() < 2:
            results.append({"bin": str(grp), "n": int(mask.sum()), "OR": np.nan, "chi2": np.nan, "p": np.nan})
            continue
        tbl = pd.crosstab(sub[confounder], sub[outcome])
        for v in [0, 1]:
            if v not in tbl.index:
                tbl.loc[v] = 0
            if v not in tbl.columns:
                tbl[v] = 0
        tbl = tbl.sort_index().sort_index(axis=1).values
        if tbl.min() == 0:
            tbl = tbl + 0.5
        chi2, p, _, _ = chi2_contingency(tbl, correction=False)
        a, b, c, d = tbl[1, 1], tbl[1, 0], tbl[0, 1], tbl[0, 0]
        or_val = (a * d) / (b * c) if b * c > 0 else np.nan
        results.append({"bin": str(grp), "n": int(mask.sum()), "OR": float(or_val), "chi2": float(chi2), "p": float(p)})
    # partial correlation confounder - outcome | control via residuals
    try:
        x = df[control].values.astype(float)
        y_arr = df[outcome].values.astype(float)
        eh_arr = df[confounder].values.astype(float)

        def residuals(a, xv):
            slope = np.cov(a, xv, bias=True)[0, 1] / (np.var(xv) + 1e-12)
            intercept = a.mean() - slope * xv.mean()
            return a - (slope * xv + intercept)

        res_y = residuals(y_arr, x)
        res_eh = residuals(eh_arr, x)
        r_partial, p_partial = pearsonr(res_eh, res_y)
        r_marginal = float(np.corrcoef(eh_arr, y_arr)[0, 1])
    except Exception:
        r_partial, p_partial, r_marginal = np.nan, np.nan, np.nan
    return {"strata": results, "partial_r": float(r_partial) if not np.isnan(r_partial) else np.nan, "partial_p": float(p_partial) if not np.isnan(p_partial) else np.nan, "marginal_r": float(r_marginal) if not np.isnan(r_marginal) else np.nan}


if __name__ == "__main__":
    # synthetic smoke
    rng = np.random.default_rng(72)
    # hub dict: 2 hubs 999,1000
    hub_neighbors = {i: {999} if i % 7 == 0 else set() for i in range(20)}
    hub_neighbors[999] = {999}
    hub_neighbors[1000] = {1000}
    # add anchors also hub-adjacent for some
    for i in range(20, 30):
        hub_neighbors[i] = {999} if i % 2 == 0 else set()
    feat_dict = {i: rng.standard_normal(5) for i in range(30)}
    feat_dict[999] = rng.standard_normal(5)
    # distances
    d_qa = 2.0
    print("synthetic is_spurious checks:")
    print(" q0( hub) - a20(hub) confounded True, d diff small =>", is_spurious(0, 20, hub_neighbors, epsilon=5.0, feat_dict=feat_dict, d_qa=d_qa))
    print(" q1(no hub) - a21(no hub) =>", is_spurious(1, 21, hub_neighbors, epsilon=5.0, feat_dict=feat_dict, d_qa=d_qa))
    print(" q0 hub - a21 no hub =>", is_spurious(0, 21, hub_neighbors, epsilon=5.0, feat_dict=feat_dict, d_qa=d_qa))
    pairs = [(0, 20, 2.0), (1, 21, 2.0), (0, 21, 2.0), (7, 22, 10.0)]
    passed = filter_neighbors(pairs, hub_neighbors)
    print(f"filter_neighbors {pairs} -> passed {passed} pass_rate {np.mean(passed):.2f}")
    # parallel form
    qs = [0, 1, 0, 7]
    ans = [20, 21, 21, 22]
    passed2 = filter_neighbors(qs, ans, hub_neighbors, epsilon=5.0, feat_dict=feat_dict, distances=[2.0, 2.0, 2.0, 10.0])
    print(f"parallel filter -> {passed2}")
    for eff in [0, 0.5, 1.0, 2.0]:
        print(f"effect {eff:.1f} -> E-value {sensitivity_evalue(eff):.3f}")
    # falsification on synthetic df
    n = 200
    df_syn = pd.DataFrame({"_deg": rng.integers(0, 20, size=n), "y": rng.integers(0, 2, size=n)})
    df_syn["Exchange_Hot"] = (df_syn["_deg"] >= 10).astype(int)
    # make confounder correlated with y via deg
    res = falsification_test(df_syn, confounder="Exchange_Hot", outcome="y", control="_deg")
    print(f"falsification synthetic: {res}")

    # smoke real if data present
    from pathlib import Path

    root = Path("data/elliptic_raw")
    if root.exists():
        from spillety.data.loader import load_elliptic, temporal_split
        import networkx as nx

        features, classes, edgelist, merged = load_elliptic(root)
        df = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
        df["y"] = (df["class"].astype(str) == "1").astype(int)
        # graph hubs
        G = nx.from_pandas_edgelist(edgelist, "txId1", "txId2", create_using=nx.DiGraph)
        UG = G.to_undirected()
        deg = dict(UG.degree())
        thr = float(np.percentile(list(deg.values()), 95))
        hub_nodes_set = {int(k) for k, v in deg.items() if v >= thr}
        # hub_neighbors dict
        from collections import defaultdict

        hub_neighbors_real = defaultdict(set)
        for a, b in edgelist.values:
            if a in hub_nodes_set:
                hub_neighbors_real[int(b)].add(int(a))
            if b in hub_nodes_set:
                hub_neighbors_real[int(a)].add(int(b))
        for h in hub_nodes_set:
            hub_neighbors_real[int(h)].add(int(h))
        df["_deg"] = df["txId"].map(deg).fillna(0).astype(int)
        df["Exchange_Hot"] = (df["_deg"] >= thr).astype(int)
        # falsification
        fres = falsification_test(df, confounder="Exchange_Hot", outcome="y", control="_deg")
        print(f"real falsification: partial_r={fres['partial_r']:.4f} p={fres['partial_p']:.3g} marginal_r={fres['marginal_r']:.4f}")
        for r in fres["strata"]:
            print(f"  bin {r['bin']}: n={r['n']} OR={r['OR']:.3f} p={r['p']:.3g}")
        # E-value demo on real delta proxy
        train_df, valid_df, test_df = temporal_split(df, train_end=30, valid_end=40)
        # simple delta proxy: degree diff effect
        sigma = float(df["_deg"].std() + 1e-9)
        eff_demo = float(abs(df.loc[df["y"] == 1, "_deg"].mean() - df.loc[df["y"] == 0, "_deg"].mean()) / sigma)
        print(f"real effect proxy {eff_demo:.3f} E-value {sensitivity_evalue(eff_demo):.3f} Tier1 {sensitivity_evalue(eff_demo) > 2.0}")
        # filter demo: pick random query-anchor pairs from edgelist
        feat_all = dict(zip(features["txId"], features[[c for c in features.columns if c.startswith("feat_")]].values))
        rng = np.random.default_rng(72)
        sample_q = rng.choice(df["txId"].values, size=100)
        sample_a = rng.choice(df["txId"].values, size=100)
        # compute d_qa quickly
        dists = []
        for q, a in zip(sample_q, sample_a):
            if q in feat_all and a in feat_all:
                dists.append(float(np.linalg.norm(feat_all[q] - feat_all[a])))
            else:
                dists.append(5.0)
        passed_real = filter_neighbors(sample_q, sample_a, hub_neighbors_real, epsilon=5.0, feat_dict=feat_all, distances=dists)
        print(f"real filter_neighbors 100 random pairs pass_rate {np.mean(passed_real):.3f}")
    else:
        print("real smoke skipped: data/elliptic_raw not found")
