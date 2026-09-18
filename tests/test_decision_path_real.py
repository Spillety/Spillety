"""## Decision-path e2e on real Elliptic++ via the train_decision_path protocol.

Protocol README: single-GBDT select/train/calibrate from
scripts/train_decision_path.py (SEED 72, temporal split 1..30 / 31..40 / 41..49,
171 features) — i.e. the v1 protocol. Reference: models/elliptic_v1/metrics.json
(pr_auc_cal 0.6472, checked within ±0.03). The v3 number (pr_auc 0.6549) is an
ensemble (0.25 * v1 + ...) and is NOT reproduced here — only its distance is reported.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from spillety.cost.operating import assign_tier, select_tau
from spillety.evidence.evidence import build_evidence, evidence_hash, verify_evidence
from spillety.evidence.merkle import merkle_proof, merkle_root, verify_proof
from spillety.models import calibration as cal
from spillety.models.gbdt import train_gbdt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "elliptic_raw"
V1_PR_AUC = 0.6472
V3_PR_AUC = 0.6549
_TOL = 0.03

_spec = importlib.util.spec_from_file_location(
    "train_decision_path", ROOT / "scripts" / "train_decision_path.py"
)
tdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tdp)

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="data/elliptic_raw missing")


@pytest.fixture(scope="module")
def _real():
    """## Build matrix, temporal-split, fit GBDT and calibrate once per module.

    Returns
    ----------
    dict
        Splits, scores, metrics, tiers and evidence artifacts on real data.
    """
    X, y, steps, names = tdp.build_matrix(str(DATA))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = tdp.split_temporal(X, y, steps)
    booster = train_gbdt(Xtr, ytr, Xva, yva)
    p_va = booster.predict(Xva)
    p_te = booster.predict(Xte)
    report = cal.calibrate(p_va, yva, p_te, yte)
    q = report["best_estimator"].predict_proba(p_te)[:, 1]
    q_va = report["best_estimator"].predict_proba(p_va)[:, 1]
    tau1 = select_tau(yva, q_va)
    tiers = [
        assign_tier(float(s), tau1, tau1 / 2.0, tau1 / 4.0, True, 3.0, 2.0)
        for s in q[:2000]
    ]
    shap = {n: float(v) for n, v in zip(names, Xte[0], strict=True)}
    ev = build_evidence(
        float(q[0]),
        tiers[0],
        shap,
        extra={"e_value": 3.0, "gamma": 2.0, "causal_passed": True},
    )
    leaves = [
        json.dumps(
            build_evidence(
                float(s),
                t,
                {n: float(v) for n, v in zip(names, Xte[i], strict=True)},
                extra={"e_value": 3.0, "gamma": 2.0, "causal_passed": True},
            ),
            sort_keys=True,
        ).encode()
        for i, (s, t) in enumerate(zip(q[:8], tiers[:8], strict=True))
    ]
    root = merkle_root(leaves)
    return {
        "X": X,
        "names": names,
        "steps": steps,
        "splits": ((Xtr, ytr), (Xva, yva), (Xte, yte)),
        "p_te": p_te,
        "q": q,
        "report": report,
        "tiers": tiers,
        "tau1": tau1,
        "ev": ev,
        "leaves": leaves,
        "root": root,
    }


def test_matrix_shape_and_temporal_split(_real):
    """## 171-column matrix with train<=30 / valid 31-40 / test 41-49."""
    X, names, steps = _real["X"], _real["names"], _real["steps"]
    assert X.shape[1] == 171
    assert len(names) == 171
    (Xtr, _), (_, _), (_, _) = _real["splits"]
    assert Xtr.shape[0] > 0
    assert steps[X.shape[0] - len(_real["splits"][2][1]) :].min() > 40
    tr = steps <= 30
    va = (steps > 30) & (steps <= 40)
    te = steps > 40
    assert tr.sum() == len(_real["splits"][0][1])
    assert va.sum() == len(_real["splits"][1][1])
    assert te.sum() == len(_real["splits"][2][1])


def test_e2e_metrics_bounds(_real):
    """## PR-AUC above chance, Brier beats trivial, ECE below 0.1."""
    (_, _), (_, yva), (_, yte) = _real["splits"]
    assert len(yva) > 0 and len(yte) > 0
    q = _real["q"]
    pr_auc = float(average_precision_score(yte, q))
    brier = cal.brier_score(yte, q)
    trivial = cal.brier_score(yte, np.full_like(q, yte.mean()))
    ece = cal.ece_score(yte, q)
    assert pr_auc > 0.5
    assert brier < trivial
    assert ece < 0.1


def test_tiers_subset_and_evidence_roundtrip(_real):
    """## Tiers stay in the allowed set; evidence hash and Merkle proof verify."""
    tiers = _real["tiers"]
    assert set(tiers) <= {"tier1", "tier2", "tier3", "clear"}
    assert len(set(tiers)) >= 2
    ev = _real["ev"]
    assert verify_evidence(ev, evidence_hash(ev))
    assert not verify_evidence(ev, "0" * 64)
    leaves, root = _real["leaves"], _real["root"]
    for i, leaf in enumerate(leaves):
        assert verify_proof(leaf, i, merkle_proof(leaves, i), root)
    assert not verify_proof(b"tampered", 0, merkle_proof(leaves, 0), root)


def test_conformance_v1_reference(_real):
    """## Calibrated PR-AUC matches the v1 single-GBDT reference within ±0.03."""
    (_, _), (_, _), (_, yte) = _real["splits"]
    pr_auc = float(average_precision_score(yte, _real["q"]))
    assert abs(pr_auc - V1_PR_AUC) < _TOL
    assert abs(pr_auc - V3_PR_AUC) < 0.10
