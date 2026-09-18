import numpy as np

from spillety.evidence.evidence import evidence_hash, verify_evidence
from spillety.pipeline.pipeline import SpilletyPipeline


def _synthetic_frames(rng, n, n_pos, k=5):
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    perm = rng.permutation(n)
    y = y[perm]
    distances = rng.standard_normal((n, k))
    distances[y == 1] -= 2.0
    freqs = rng.standard_normal((n, 2))
    freqs[y == 1, 0] += 1.5
    min_dist = rng.standard_normal((n, 3))
    min_dist[y == 1] -= 1.0
    sensitivity = np.column_stack(
        [
            rng.uniform(0.5, 1.0, n),
            np.where(y == 1, 3.0, 1.0),
            np.where(y == 1, 2.0, 1.0),
            rng.uniform(0.0, 0.3, n),
        ]
    )
    graph = rng.standard_normal((n, 3))
    temporal = rng.standard_normal((n, 2))
    context = rng.standard_normal((n, 2))
    frames = {
        "distances": distances,
        "anchor_type_freqs": freqs,
        "anchor_min_dist": min_dist,
        "sensitivity": sensitivity,
        "graph": graph,
        "temporal": temporal,
        "context": context,
    }
    return frames, y


def _split(frames, y, rng):
    n = len(y)
    idx = rng.permutation(n)
    train_idx, valid_idx, test_idx = idx[:300], idx[300:400], idx[400:]
    sub = lambda d, i: {k: np.asarray(v)[i] for k, v in d.items()}
    return (
        sub(frames, train_idx),
        y[train_idx],
        sub(frames, valid_idx),
        y[valid_idx],
        sub(frames, test_idx),
        y[test_idx],
    )


def test_pipeline_fit_predict_evaluate():
    rng = np.random.default_rng(72)
    frames, y = _synthetic_frames(rng, 500, 40)
    f_train, y_train, f_valid, y_valid, f_test, y_test = _split(frames, y, rng)

    pipe = SpilletyPipeline(config={"random_state": 72, "n_boot": 100})
    pipe.fit(f_train, y_train, f_valid, y_valid)
    scores, tiers, evidences = pipe.predict(f_test)

    assert not np.isnan(scores).any()
    assert len(tiers) == len(y_test) == len(evidences)
    assert set(tiers) <= {"tier1", "tier2", "tier3", "clear"}
    for ev in evidences:
        assert len(ev["shap_values"]) <= 10

    metrics = pipe.evaluate(y_test, scores)
    assert metrics["pr_auc"] > y_test.mean()
    trivial_brier = float(np.mean((y_test - y_test.mean()) ** 2))
    assert metrics["brier"] < trivial_brier
    assert metrics["calibrator"] in ("isotonic", "beta")


def test_pipeline_tier1_gate():
    rng = np.random.default_rng(72)
    frames, y = _synthetic_frames(rng, 500, 40)
    f_train, y_train, f_valid, y_valid, f_test, y_test = _split(frames, y, rng)

    pipe = SpilletyPipeline(config={"random_state": 72, "n_boot": 100})
    pipe.fit(f_train, y_train, f_valid, y_valid)
    scores = pipe.predict_proba(f_test)
    hot = scores > pipe.tau1_
    assert hot.any(), "synthetic shift must place rows above tau1"

    n = len(y_test)
    _, tiers_open, _ = pipe.predict(
        f_test,
        causal_passed=np.ones(n, dtype=bool),
        e_value=np.full(n, 3.0),
        gamma=np.full(n, 2.0),
    )
    assert any(t == "tier1" for t, h in zip(tiers_open, hot) if h)

    _, tiers_closed, _ = pipe.predict(
        f_test,
        causal_passed=np.zeros(n, dtype=bool),
        e_value=np.full(n, 3.0),
        gamma=np.full(n, 2.0),
    )
    assert not any(t == "tier1" for t in tiers_closed)


def test_pipeline_evidence_roundtrip():
    rng = np.random.default_rng(72)
    frames, y = _synthetic_frames(rng, 500, 40)
    f_train, y_train, f_valid, y_valid, f_test, _ = _split(frames, y, rng)

    pipe = SpilletyPipeline(config={"random_state": 72, "n_boot": 100})
    pipe.fit(f_train, y_train, f_valid, y_valid)
    _, _, evidences = pipe.predict(f_test)

    ev = evidences[0]
    digest = evidence_hash(ev)
    assert verify_evidence(ev, digest)
    tampered = dict(ev)
    tampered["score"] = round(max(0.0, ev["score"] - 0.01), 4)
    assert tampered["score"] != ev["score"]
    assert not verify_evidence(tampered, digest)
