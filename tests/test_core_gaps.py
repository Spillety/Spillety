"""H1 unit gaps of the current core (seed 72 everywhere)."""

from itertools import pairwise

import numpy as np
import pandas as pd

from spillety.cost.operating import alerts_at, assign_tier, cost_at, select_tau
from spillety.data.loader import temporal_split
from spillety.evidence.evidence import build_evidence
from spillety.features.news import (
    build_comention_graph,
    entity_news_features,
    news_proximity,
)
from spillety.features.temporal import burstiness_series, hawkes_lambda_series
from spillety.metrics.dashboard import reliability_table
from spillety.models.calibration import (
    BetaCalibrator,
    IsotonicCalibrator,
    ece_score,
    reliability_stats,
)
from spillety.pipeline.pipeline import SpilletyPipeline


def _mentions():
    return pd.DataFrame(
        {
            "entity": ["a", "b", "a", "c", "b", "c", "a", "d"],
            "article_id": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )


def test_news_graph_shape_and_determinism():
    m = _mentions()
    g1, edges1 = build_comention_graph(m)
    g2, edges2 = build_comention_graph(m)
    assert set(g1.nodes()) == {"a", "b", "c", "d"}
    assert g1.number_of_edges() == 4
    assert sorted(map(tuple, map(sorted, g1.edges()))) == sorted(
        map(tuple, map(sorted, g2.edges()))
    )
    assert edges1.equals(edges2)


def test_news_entity_features_counts():
    m = _mentions()
    feat = entity_news_features(m, "a")
    assert feat["mention_count"] == 3
    assert feat["unique_articles"] == 3
    missing = entity_news_features(m, "zzz")
    assert missing["mention_count"] == 0
    assert missing["unique_articles"] == 0


def test_news_proximity_order():
    g, _ = build_comention_graph(_mentions())
    assert news_proximity("a", "a", g) == 1.0
    assert news_proximity("a", "b", g) == 0.5
    assert news_proximity("a", "b", g) > news_proximity("b", "d", g) > 0.0
    assert news_proximity("a", "zzz", g) == 0.0


def test_cost_at_ties_count_as_alert():
    y = np.array([0, 1])
    s = np.array([0.5, 0.5])
    assert cost_at(y, s, 0.5) == 1.0
    assert cost_at(y, s, 0.51) == 10.0


def test_alerts_at_monotone_in_tau():
    rng = np.random.default_rng(72)
    s = rng.uniform(0, 1, 500)
    taus = np.linspace(0, 1, 11)
    volumes = [alerts_at(s, t) for t in taus]
    assert all(b <= a for a, b in pairwise(volumes))


def test_select_tau_respects_budget():
    rng = np.random.default_rng(72)
    y = (rng.uniform(0, 1, 500) < 0.1).astype(int)
    s = rng.uniform(0, 1, 500)
    tau = select_tau(y, s, budget=25)
    assert alerts_at(s, tau) <= 25


def test_assign_tier_gates():
    assert assign_tier(0.99, 0.8, 0.5, 0.2, True, 3.0, 2.0) == "tier1"
    assert assign_tier(0.99, 0.8, 0.5, 0.2, False, 3.0, 2.0) != "tier1"
    assert assign_tier(0.99, 0.8, 0.5, 0.2, True, 2.0, 2.0) != "tier1"
    assert assign_tier(0.99, 0.8, 0.5, 0.2, True, 3.0, 1.5) != "tier1"
    assert assign_tier(0.99, 0.8, 0.5, 0.2, True, 1.0, 2.0) == "tier2"
    assert assign_tier(0.6, 0.8, 0.5, 0.2, True, 3.0, 2.0) == "tier2"
    assert assign_tier(0.3, 0.8, 0.5, 0.2, True, 3.0, 2.0) == "tier3"
    assert assign_tier(0.1, 0.8, 0.5, 0.2, True, 3.0, 2.0) == "clear"


def test_ece_perfect_vs_constant():
    rng = np.random.default_rng(72)
    y = rng.integers(0, 2, 500)
    assert ece_score(y, y.astype(float)) == 0.0
    assert ece_score(y, np.zeros(500)) > 0.0


def test_reliability_stats_bin_sums():
    rng = np.random.default_rng(72)
    y = rng.integers(0, 2, 500)
    stats = reliability_stats(y, rng.uniform(0, 1, 500))
    assert sum(stats["count"]) == 500


def test_calibrators_clip_out_of_bounds():
    rng = np.random.default_rng(72)
    x = rng.uniform(0, 1, 200)
    y = rng.integers(0, 2, 200)
    probe = np.array([-1e6, 0.5, 1e6])
    for cls in (IsotonicCalibrator, BetaCalibrator):
        p = cls().fit(x, y).predict_proba(probe)[:, 1]
        assert not np.isnan(p).any()
        assert ((p >= 0.0) & (p <= 1.0)).all()


def test_temporal_split_boundaries():
    df = pd.DataFrame({"time_step": list(range(50)) * 2, "x": np.arange(100)})
    train, valid, test = temporal_split(df)
    assert train["time_step"].max() <= 30
    assert valid["time_step"].min() > 30
    assert valid["time_step"].max() <= 40
    assert test["time_step"].min() > 40
    assert len(train) + len(valid) + len(test) == len(df)


def test_reliability_table_diagonal_on_calibrated():
    rng = np.random.default_rng(72)
    p = np.where(rng.uniform(0, 1, 2000) < 0.5, 0.2, 0.8)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    for row in reliability_table(y, p, n_bins=5):
        if row["count"] > 0:
            assert abs(row["acc"] - row["conf"]) < 0.05


def test_burstiness_fixed_series():
    counts = pd.Series([1, 1, 4, 1, 1], index=[1, 2, 3, 4, 5])
    got = burstiness_series(counts).tolist()
    assert got == [
        -1.0,
        -1.0,
        -0.17157287525380988,
        -0.17157287525380988,
        -0.17157287525380988,
    ]


def test_hawkes_fixed_series():
    counts = pd.Series([1, 1, 4, 1, 1], index=[1, 2, 3, 4, 5])
    got = hawkes_lambda_series(counts).tolist()
    assert got == [
        1.6,
        1.7839397205857213,
        1.8516073622040277,
        2.428320058145123,
        2.0886616406872456,
    ]


def test_build_evidence_tier_top10_extra():
    shap = {f"f{i}": float(i - 10) for i in range(20)}
    ev = build_evidence(
        0.5,
        "tier2",
        shap,
        extra={"e_value": 1.0, "gamma": 1.0, "causal_passed": False},
    )
    assert ev["tier"] == "tier2"
    assert len(ev["shap_values"]) == 10
    assert set(ev["shap_values"]) == {
        "f0",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f16",
        "f17",
        "f18",
        "f19",
    }
    assert ev["e_value"] == 1.0
    assert ev["gamma"] == 1.0
    assert ev["causal_passed"] is False


def _synthetic_frames(rng, n, n_pos, k=5):
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    y = y[rng.permutation(n)]
    distances = rng.standard_normal((n, k))
    distances[y == 1] -= 2.0
    return (
        {
            "distances": distances,
            "anchor_type_freqs": rng.standard_normal((n, 2)),
            "anchor_min_dist": rng.standard_normal((n, 3)),
            "sensitivity": rng.standard_normal((n, 4)),
            "graph": rng.standard_normal((n, 3)),
            "temporal": rng.standard_normal((n, 2)),
            "context": rng.standard_normal((n, 2)),
        },
        y,
    )


def _fit(seed=72):
    rng = np.random.default_rng(seed)
    frames, y = _synthetic_frames(rng, 500, 40)
    idx = rng.permutation(500)
    sub = lambda d, i: {kk: np.asarray(v)[i] for kk, v in d.items()}
    pipe = SpilletyPipeline(config={"random_state": 72, "n_boot": 100})
    pipe.fit(sub(frames, idx[:300]), y[idx[:300]], sub(frames, idx[300:400]), y[idx[300:400]])
    scores = pipe.predict_proba(sub(frames, idx[400:]))
    return pipe, scores


def test_pipeline_threshold_order_and_determinism():
    pipe, scores = _fit()
    assert pipe.tau1_ >= pipe.tau2_ >= pipe.tau3_
    _, scores2 = _fit()
    assert np.array_equal(scores, scores2)
