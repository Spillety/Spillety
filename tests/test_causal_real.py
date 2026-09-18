import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_dag_select as rds

from spillety.causal.filter import causal_filter
from spillety.causal.select import power_gate, select_dag
from spillety.data.loader import load_elliptic

N_BOOT = 30
SUB_N = 3000


@pytest.fixture(scope="module")
def train_frame():
    """## Train-era labeled frame with structural confounders (seed 72)."""
    features, _classes, edgelist, merged = load_elliptic(ROOT / "data/elliptic_raw")
    lab = merged[merged["class"].isin(["1", "2"])].copy()
    lab["y"] = (lab["class"] == "1").astype(int)
    struct, _thr = rds.build_structural_confounders(
        lab[["txId"]], edgelist, features[["txId", "time_step"]]
    )
    lab = lab.join(struct, on="txId")
    train = lab[lab["time_step"] <= 35].reset_index(drop=True)
    sub = train.sample(n=SUB_N, random_state=72).reset_index(drop=True)
    return sub


def _select_once(train):
    rng = np.random.default_rng(72)
    evals = {n: rds.evaluate_candidate(n, train, N_BOOT, rng) for n in rds.CANDIDATES}
    specs = {
        n: {
            "rejections": e["rejections"],
            "num_edges": e["num_edges"],
            "n": len(train),
            "effect": e["effect"],
        }
        for n, e in evals.items()
    }
    return rds.select_dag(specs), evals


def test_select_deterministic_on_3k_subsample(train_frame):
    sel1, evals1 = _select_once(train_frame)
    sel2, evals2 = _select_once(train_frame)
    assert sel1["best"] == sel2["best"]
    assert sel1["table"] == sel2["table"]
    assert sel1["best"] is None or sel1["best"] in rds.CANDIDATES
    rej1 = {n: e["rejections"] for n, e in evals1.items()}
    rej2 = {n: e["rejections"] for n, e in evals2.items()}
    assert rej1 == rej2


def test_pass_rate_in_unit_interval(train_frame):
    best, _evals = _select_once(train_frame)
    name = best["best"] or best["table"][0]["name"]
    conf = rds.CANDIDATES[name]
    mu = train_frame[rds.ANCHORS].mean()
    sd = train_frame[rds.ANCHORS].std().replace(0, 1.0)
    a = ((train_frame[rds.ANCHORS] - mu) / sd).to_numpy(dtype=float)
    out = causal_filter(
        a, train_frame["y"].to_numpy(dtype=float), rds._cols(train_frame, conf)
    )
    assert 0.0 <= out["pass_rate"] <= 1.0
    assert out["mask"].shape == (len(rds.ANCHORS),)


def test_gate_branches():
    assert power_gate(20, 0.3)["status"] == "insufficient data"
    assert power_gate(31235, 0.05)["status"] == "ok"
    empty = select_dag(
        {"DAG-X": {"rejections": 0, "num_edges": 1, "n": 20, "effect": 0.3}}
    )
    assert empty["best"] is None and empty["status"] == "insufficient data"
