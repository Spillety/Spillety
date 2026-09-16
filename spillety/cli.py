#!/usr/bin/env python3
"""CLI for Spillety pipeline: fit, predict, evaluate, smoke."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spillety.data.loader import load_elliptic, temporal_split
from spillety.pipeline.pipeline import SpilletyPipeline


def cmd_smoke(args):
    rng = np.random.default_rng(args.seed)
    n, d = 1000, 165
    p = Path("data/elliptic_raw")
    if p.exists() and not args.synthetic:
        _, _, _, merged = load_elliptic(p)
        df_all = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
        df_all["y"] = (df_all["class"].astype(str) == "1").astype(int)
        df_all = df_all.sample(n=n, random_state=args.seed)
        train_df, valid_df, test_df = temporal_split(df_all, train_end=30, valid_end=40)
        if len(train_df) < 100 or len(valid_df) < 50 or len(test_df) < 50:
            q1, q2 = df_all["time_step"].quantile([0.6, 0.8])
            train_df = df_all[df_all["time_step"] <= q1].copy()
            valid_df = df_all[(df_all["time_step"] > q1) & (df_all["time_step"] <= q2)].copy()
            test_df = df_all[df_all["time_step"] > q2].copy()
        edgelist = pd.read_csv(p / "elliptic_txs_edgelist.csv")
        pipe = SpilletyPipeline(config={"random_state": args.seed, "edgelist": edgelist})
    else:
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
        train_df = df_syn.sample(n=600, random_state=args.seed)
        remain = df_syn.drop(train_df.index)
        valid_df = remain.sample(n=200, random_state=args.seed + 1)
        test_df = remain.drop(valid_df.index)
        pipe = SpilletyPipeline(config={"random_state": args.seed})

    pipe.fit(train_df, valid_df)
    scores, evidences = pipe.predict(test_df)
    y_test = test_df["y"].values
    from sklearn.metrics import average_precision_score
    pr = average_precision_score(y_test, scores)
    base = float(y_test.mean())
    lb = pipe.latency_budget
    metrics = pipe.evaluate(y_test, scores)
    print(json.dumps({
        "pr_auc": round(pr, 4),
        "baseline": round(base, 4),
        "latency_p99_ms": round(lb["p99_ms"], 3) if lb else None,
        "threshold": round(pipe.threshold_, 4) if pipe.threshold_ is not None else None,
        "calibrator": pipe.calibrator_name_,
        "brier": round(metrics["brier"], 4),
        "ece": round(metrics["ece"], 4),
    }, indent=2, ensure_ascii=False))
    assert pr > base, f"PR-AUC {pr:.4f} not > baseline {base:.4f}"
    assert (lb["p99_ms"] if lb else 0) < 100, "latency too high"
    print("smoke ok")


def cmd_fit(args):
    p = Path(args.data)
    _, _, _, merged = load_elliptic(p)
    df = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
    df["y"] = (df["class"].astype(str) == "1").astype(int)
    train_df, valid_df, test_df = temporal_split(df, train_end=args.train_end, valid_end=args.valid_end)
    edgelist = pd.read_csv(p / "elliptic_txs_edgelist.csv") if (p / "elliptic_txs_edgelist.csv").exists() else None
    pipe = SpilletyPipeline(config={"random_state": args.seed, "edgelist": edgelist, "C_FP": args.c_fp, "C_FN": args.c_fn})
    pipe.fit(train_df, valid_df)
    scores, _ = pipe.predict(test_df)
    y_test = test_df["y"].values
    metrics = pipe.evaluate(y_test, scores)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    if args.out:
        import pickle
        Path(args.out).write_bytes(pickle.dumps(pipe))
        print(f"model saved to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Spillety CLI")
    sub = parser.add_subparsers(dest="command")

    smoke = sub.add_parser("smoke", help="run smoke test")
    smoke.add_argument("--seed", type=int, default=72)
    smoke.add_argument("--synthetic", action="store_true", help="force synthetic data")
    smoke.set_defaults(func=cmd_smoke)

    fit = sub.add_parser("fit", help="fit pipeline on elliptic data")
    fit.add_argument("--data", default="data/elliptic_raw")
    fit.add_argument("--train-end", type=int, default=30)
    fit.add_argument("--valid-end", type=int, default=40)
    fit.add_argument("--c-fp", type=float, default=1.0)
    fit.add_argument("--c-fn", type=float, default=50.0)
    fit.add_argument("--seed", type=int, default=72)
    fit.add_argument("--out", default=None, help="pickle output path")
    fit.set_defaults(func=cmd_fit)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
