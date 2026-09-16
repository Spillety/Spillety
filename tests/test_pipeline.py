import numpy as np
import pandas as pd
import pytest

from spillety.pipeline.pipeline import SpilletyPipeline


def test_pipeline_smoke_synthetic():
    rng = np.random.default_rng(72)
    n, d = 500, 165
    X = rng.standard_normal((n, d))
    y = np.array([1] * 40 + [0] * 460)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    X[y == 1] += 2.0
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(2, 2 + d)])
    df["txId"] = np.arange(10000, 10000 + n)
    df["time_step"] = rng.integers(1, 50, size=n)
    df["class"] = np.where(y == 1, "1", "2")
    df["y"] = y
    train_df = df.sample(n=300, random_state=72)
    remain = df.drop(train_df.index)
    valid_df = remain.sample(n=100, random_state=73)
    test_df = remain.drop(valid_df.index)

    pipe = SpilletyPipeline(config={"random_state": 72, "n_components": 16})
    pipe.fit(train_df, valid_df)
    scores, evidences = pipe.predict(test_df)
    y_test = test_df["y"].values

    assert not np.isnan(scores).any()
    assert len(evidences) == len(test_df)
    from sklearn.metrics import average_precision_score
    pr = average_precision_score(y_test, scores)
    assert pr > y_test.mean()
    metrics = pipe.evaluate(y_test, scores)
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["brier"] <= 1
    lb = pipe.latency_budget
    assert lb["p99_ms"] < 100


def test_pipeline_evidence_verify():
    rng = np.random.default_rng(72)
    n, d = 200, 165
    X = rng.standard_normal((n, d))
    y = np.array([1] * 20 + [0] * 180)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    X[y == 1] += 2.0
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(2, 2 + d)])
    df["txId"] = np.arange(10000, 10000 + n)
    df["time_step"] = rng.integers(1, 50, size=n)
    df["class"] = np.where(y == 1, "1", "2")
    df["y"] = y
    train_df = df.sample(n=120, random_state=72)
    remain = df.drop(train_df.index)
    valid_df = remain.sample(n=40, random_state=73)
    test_df = remain.drop(valid_df.index)

    pipe = SpilletyPipeline(config={"random_state": 72, "n_components": 8})
    pipe.fit(train_df, valid_df)
    scores, evidences = pipe.predict(test_df)
    from spillety.evidence.worm import verify_evidence
    for ev in evidences[:3]:
        sig = ev["_signature"]["sig"]
        plain = {k: v for k, v in ev.items() if not k.startswith("_")}
        assert verify_evidence(plain, sig, pipe.key)
