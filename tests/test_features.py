import numpy as np
import pandas as pd
import pytest

from spillety.features.build import build_feature_matrix
from spillety.features.graph import add_graph_features, compute_graph_features
from spillety.features.temporal import add_temporal_features, compute_temporal_features


def test_build_feature_matrix_shape_names_and_validation():
    rng = np.random.default_rng(72)
    frames = {
        "distances": rng.standard_normal((50, 4)),
        "anchor_type_freqs": rng.standard_normal((50, 2)),
        "anchor_min_dist": rng.standard_normal((50, 3)),
        "sensitivity": rng.standard_normal((50, 4)),
        "graph": rng.standard_normal((50, 3)),
        "temporal": rng.standard_normal((50, 2)),
        "context": rng.standard_normal((50, 2)),
    }
    x, names = build_feature_matrix(frames)
    assert x.shape == (50, len(names))
    assert len(names) == len(set(names))
    assert names[:4] == ["d_1", "d_2", "d_3", "d_4"]
    assert "dist_min" in names and "anchor_source_diversity" in names
    assert "distance_to_nearest_OFAC" in names

    x2, names2 = build_feature_matrix(frames)
    assert names2 == names
    assert np.array_equal(x, x2)

    with pytest.raises(ValueError, match="distances"):
        build_feature_matrix({"graph": rng.standard_normal((50, 2))})
    bad = dict(frames)
    bad["graph"] = rng.standard_normal((49, 3))
    with pytest.raises(ValueError, match="n="):
        build_feature_matrix(bad)


def test_graph_features_shape_and_no_nan():
    edgelist = pd.DataFrame({"txId1": [1, 1, 2, 5], "txId2": [2, 3, 3, 6]})
    df = pd.DataFrame({"txId": [1, 2, 3, 4, 5, 6], "time_step": [1, 1, 2, 2, 3, 3]})
    feat = compute_graph_features(edgelist, tx_ids=df["txId"])
    assert feat.shape == (6, 6)  # txId + 5 features
    assert set(feat.columns) == {"txId", "in_degree", "out_degree", "total_degree", "pagerank", "in_out_ratio"}
    assert not feat.isna().any().any(), "graph features must not contain NaN"
    # isolated node 4 has 0 degree and 0 pagerank contribution -> still not NaN
    row4 = feat.loc[feat["txId"] == 4].iloc[0]
    assert row4["in_degree"] == 0 and row4["out_degree"] == 0
    assert row4["in_out_ratio"] == 1.0  # (0+1)/(0+1)

    enriched = add_graph_features(df, edgelist)
    assert enriched.shape[0] == 6
    assert not enriched[["in_degree", "out_degree", "pagerank", "in_out_ratio"]].isna().any().any()


def test_temporal_features_shape_and_ranges():
    df = pd.DataFrame(
        {
            "txId": range(1, 10),
            "time_step": [1, 1, 2, 2, 3, 3, 4, 4, 5],
            "class": ["2", "1", "2", "2", "1", "2", "2", "2", "2"],
        }
    )
    stats = compute_temporal_features(df)
    assert not stats[["burstiness", "hawkes_lambda", "time_since_last_illicit"]].isna().any().any()
    assert stats["burstiness"].between(-1, 1).all()
    assert (stats["hawkes_lambda"] >= 0).all()

    enriched = add_temporal_features(df)
    assert enriched.shape[0] == df.shape[0]
    assert not enriched[["burstiness", "hawkes_lambda", "time_since_last_illicit"]].isna().any().any()
    # burstiness and hawkes map correctly for each time_step
    for t in df["time_step"].unique():
        assert (enriched.loc[enriched["time_step"] == t, "burstiness"] == stats.loc[t, "burstiness"]).all()


def test_temporal_degenerate_no_illicit():
    df = pd.DataFrame({"txId": [1, 2], "time_step": [1, 2], "class": ["2", "2"]})
    stats = compute_temporal_features(df)
    # when no illicit ever, fill 49
    assert (stats["time_since_last_illicit"] == 49).all()
    enriched = add_temporal_features(df)
    assert (enriched["time_since_last_illicit"] == 49).all()


def test_graph_and_temporal_on_real_data_smoke():
    from pathlib import Path

    root = Path("data/elliptic_raw")
    if not root.exists():
        return
    from spillety.data.loader import load_elliptic

    _, _, edgelist, merged = load_elliptic(root)
    g = compute_graph_features(edgelist, tx_ids=merged["txId"].head(2000))
    assert not g.isna().any().any()
    t = compute_temporal_features(merged)
    assert t.shape[0] == 49
    assert not t.isna().any().any()
