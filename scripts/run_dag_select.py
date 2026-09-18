#!/usr/bin/env python3
"""## J2 wave J: DAG-select on real Elliptic data (§6.3)."""

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from spillety.causal.filter import causal_filter
from spillety.causal.select import power_gate, select_dag
from spillety.causal.validate import ci_test, falsification_test
from spillety.data.loader import load_elliptic
from spillety.features.temporal import hawkes_lambda_series

SEED = 72
ALPHA = 0.05
ANCHORS = ["feat_2", "feat_3", "feat_4", "feat_5"]
POOL = ["exchange_hot", "mixer", "hawkes"]

CANDIDATES = {
    "DAG-A": [],
    "DAG-B": ["exchange_hot"],
    "DAG-C": ["exchange_hot", "mixer"],
    "DAG-D": ["exchange_hot", "mixer", "hawkes"],
    "DAG-E": ["hawkes"],
}


def build_structural_confounders(
    merged: pd.DataFrame, edgelist: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """
    ## Hub proxies + Hawkes activity as observed confounders (§6.3, nb-09)

    Parameters
    ----------
    merged : pd.DataFrame
        Labeled rows with txId.
    edgelist : pd.DataFrame
        Columns [txId1, txId2].
    features : pd.DataFrame
        Columns [txId, time_step, ...] for all txs.

    Returns
    ----------
    tuple[pd.DataFrame, dict]
        Structural frame indexed by txId, plus hub thresholds.
    """
    g = nx.DiGraph()
    g.add_edges_from(zip(edgelist["txId1"], edgelist["txId2"]))
    ug = g.to_undirected()
    degree = dict(ug.degree())
    # ponytail: power-iteration PageRank; exact-personalized variant
    # if hub ranking ever needs sub-1% stability (§6.3 follow-up).
    pr = nx.pagerank(g, alpha=0.85)
    deg_s = pd.Series(degree, dtype=float)
    pr_s = pd.Series(pr, dtype=float)
    deg_thr = float(deg_s.quantile(0.95))
    pr_thr = float(pr_s.quantile(0.95))
    step_counts = (
        features.groupby("time_step").size().reindex(range(1, 50), fill_value=0)
    )
    hawkes = hawkes_lambda_series(step_counts)
    tx_time = features.set_index("txId")["time_step"]
    out = pd.DataFrame(index=merged["txId"].unique())
    out["degree"] = out.index.map(deg_s).fillna(0.0)
    out["exchange_hot"] = (out.index.map(deg_s).fillna(0.0) >= deg_thr).astype(float)
    out["mixer"] = (out.index.map(pr_s).fillna(0.0) >= pr_thr).astype(float)
    out["hawkes"] = out.index.map(tx_time).map(hawkes).fillna(0.0)
    return out, {"deg_thr": deg_thr, "pr_thr": pr_thr}


def _cols(df: pd.DataFrame, names: list[str]) -> np.ndarray | None:
    if not names:
        return None
    return df[names].to_numpy(dtype=float)


def evaluate_candidate(
    name: str,
    train: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """
    ## Score one DAG: present edges via ci_test, absences via falsification

    Parameters
    ----------
    name : str
        Key in CANDIDATES.
    train : pd.DataFrame
        Train-era labeled rows (steps ≤ 35).
    n_boot : int
        Bootstrap replications for ci_test.
    rng : np.random.Generator
        Seeded generator for the placebo column.

    Returns
    ----------
    dict
        Per-edge ci/falsification outputs, rejections, effect, num_edges.
    """
    conf = CANDIDATES[name]
    y = train["y"].to_numpy(dtype=float)
    c = _cols(train, conf)
    edges = []
    rejections = 0
    effects = []
    for anchor in ANCHORS:
        x = train[anchor].to_numpy(dtype=float)
        ci = ci_test(x, y, c, alpha=ALPHA, n_boot=n_boot, random_state=SEED)
        fals = falsification_test(y, x, c)
        supported = bool(ci["reject"])
        rejections += 0 if supported else 1
        effects.append(abs(float(ci["r"])))
        edges.append(
            {
                "edge": f"{anchor}->y|{conf or []}",
                "kind": "present",
                "ci": _jsonable(ci),
                "fals": _jsonable(fals),
                "rejected": not supported,
                "why": "" if supported else "ci_not_reject",
            }
        )
    absent = [v for v in POOL if v not in conf]
    if name == "DAG-D":
        absent = absent + ["time_step"]
    for var in absent:
        x = train[var].to_numpy(dtype=float)
        ci = ci_test(x, y, c, alpha=ALPHA, n_boot=n_boot, random_state=SEED)
        fals = falsification_test(y, x, c)
        bad = bool(fals["reject"])
        rejections += 1 if bad else 0
        edges.append(
            {
                "edge": f"{var}_||_y|{conf or []}",
                "kind": "absent",
                "ci": _jsonable(ci),
                "fals": _jsonable(fals),
                "rejected": bad,
                "why": "falsified" if bad else "",
            }
        )
    placebo = rng.standard_normal(len(train))
    fals = falsification_test(y, placebo, c)
    bad = bool(fals["reject"])
    rejections += 1 if bad else 0
    edges.append(
        {
            "edge": f"placebo_||_y|{conf or []}",
            "kind": "placebo",
            "ci": None,
            "fals": _jsonable(fals),
            "rejected": bad,
            "why": "falsified" if bad else "",
        }
    )
    return {
        "name": name,
        "confounders": conf,
        "edges": edges,
        "rejections": rejections,
        "num_edges": len(conf) + len(ANCHORS),
        "effect": float(np.median(effects)) if effects else 0.0,
    }


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, tuple):
            out[k] = [float(a) for a in v]
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def filter_pass_rates(train: pd.DataFrame, test: pd.DataFrame, conf: list[str]) -> dict:
    """
    ## causal_filter pass_rate on train vs test eras (§6.4.2 stability)

    Parameters
    ----------
    train, test : pd.DataFrame
        Labeled era frames with anchor + confounder columns.
    conf : list[str]
        Confounder set of the selected DAG.

    Returns
    ----------
    dict
        pass_rate, e_value quantiles and Tier1 share per era.
    """
    mu = train[ANCHORS].mean()
    sd = train[ANCHORS].std().replace(0, 1.0)
    out = {}
    for era, df in (("train", train), ("test", test)):
        a = ((df[ANCHORS] - mu) / sd).to_numpy(dtype=float)
        w = df["y"].to_numpy(dtype=float)
        res = causal_filter(a, w, _cols(df, conf), alpha=ALPHA)
        ev = res["e_value"]
        out[era] = {
            "pass_rate": float(res["pass_rate"]),
            "e_value_p50": float(np.quantile(ev, 0.5)),
            "e_value_p90": float(np.quantile(ev, 0.9)),
            "tier1_share": float(np.mean(ev > 2.0)),
        }
    return out


def main() -> None:
    """## CLI: select DAG on train era, report stability on test era."""
    ap = argparse.ArgumentParser(description="J2 wave J: DAG-select on real data")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=str, default="models/causal_v1")
    args = ap.parse_args()
    t0 = time.time()
    features, _classes, edgelist, merged = load_elliptic("data/elliptic_raw")
    lab = merged[merged["class"].isin(["1", "2"])].copy()
    lab["y"] = (lab["class"] == "1").astype(int)
    struct, attrs = build_structural_confounders(
        lab[["txId"]], edgelist, features[["txId", "time_step"]]
    )
    lab = lab.join(struct, on="txId")
    train = lab[lab["time_step"] <= 35].reset_index(drop=True)
    test = lab[lab["time_step"].between(41, 49)].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    evals = {n: evaluate_candidate(n, train, args.n_boot, rng) for n in CANDIDATES}
    specs = {
        n: {
            "rejections": e["rejections"],
            "num_edges": e["num_edges"],
            "n": len(train),
            "effect": e["effect"],
        }
        for n, e in evals.items()
    }
    sel = select_dag(specs)
    gates = {n: power_gate(len(train), e["effect"]) for n, e in evals.items()}
    best_conf = CANDIDATES[sel["best"]] if sel["best"] else []
    rates = filter_pass_rates(train, test, best_conf)
    report = {
        "seed": args.seed,
        "alpha": ALPHA,
        "n_boot": args.n_boot,
        "ponytail": args.n_boot < 1000,
        "train_n": len(train),
        "test_n": len(test),
        "train_illicit_rate": float(train["y"].mean()),
        "test_illicit_rate": float(test["y"].mean()),
        "hub_thresholds": {
            "degree_top5": attrs["deg_thr"],
            "pagerank_top5": attrs["pr_thr"],
        },
        "candidates": evals,
        "selection": sel,
        "power_gates": gates,
        "selected_confounders": best_conf,
        "filter_stability": rates,
        "elapsed_s": round(time.time() - t0, 1),
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    (outdir / "README.md").write_text(render_readme(report))
    print(
        json.dumps(
            {
                "best": sel["best"],
                "status": sel["status"],
                "rejections": {n: e["rejections"] for n, e in evals.items()},
                "pass_rate": {k: v["pass_rate"] for k, v in rates.items()},
            },
            indent=2,
        )
    )


def render_readme(rep: dict) -> str:
    """## Render models/causal_v1/README.md from the report dict."""
    lines = [
        "# Causal v1 — DAG-select on real data (J2 wave J, §6.3)",
        "",
        (
            "Best: **" + str(rep["selection"]["best"]) + "** "
            f"(status {rep['selection']['status']}), "
            f"train n={rep['train_n']} (steps ≤35 labeled), "
            f"test n={rep['test_n']} (steps 41–49)."
        ),
        "",
        "## Rejected edges",
        "",
    ]
    for name, ev in rep["candidates"].items():
        rej = [e for e in ev["edges"] if e["rejected"]]
        lines.append(
            f"### {name} (C={ev['confounders'] or []}): {ev['rejections']} rejections"
        )
        if not rej:
            lines.append("- none")
        for e in rej:
            if e["kind"] == "present":
                ci = e["ci"]
                lines.append(
                    f"- `{e['edge']}`: unsupported — ci p={ci['p']:.3g}, "
                    f"CI=[{ci['p_ci'][0]:.3g},{ci['p_ci'][1]:.3g}] ≥ α (edge adds nothing beyond C)."
                )
            else:
                if e["kind"] == "placebo":
                    lines.append(
                        f"- `{e['edge']}`: falsified — placebo carries "
                        f"signal beyond C (fals p={e['fals']['p']:.3g}; spec failure)."
                    )
                else:
                    lines.append(
                        f"- `{e['edge']}`: falsified — absent var carries "
                        f"signal beyond C (fals p={e['fals']['p']:.3g})."
                    )
        lines.append("")
    st = rep["filter_stability"]
    lines += [
        "## Stability (causal_filter pass_rate, selected DAG)",
        f"- train: {st['train']['pass_rate']:.3f}, test: {st['test']['pass_rate']:.3f}",
        (
            f"- E-value p50 train/test: {st['train']['e_value_p50']:.2f} / {st['test']['e_value_p50']:.2f}; "
            f"Tier1 (E>2) share train/test: {st['train']['tier1_share']:.3f} / {st['test']['tier1_share']:.3f}"
        ),
        "",
        "## Notes",
        "- Selection stats use train era only (steps ≤35 labeled); no future rows in ci/falsification/select.",
        (
            "- Hub flags are label-free structural proxies (full-graph degree/PageRank, top 5%); "
            "Hawkes λ is causal-in-time (past counts only)."
        ),
        "- Bootstrap n=1000"
        + (
            " (ponytail: reduced n — treat CIs as approximate)."
            if rep["ponytail"]
            else "."
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
