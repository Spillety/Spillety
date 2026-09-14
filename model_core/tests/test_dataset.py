import numpy as np
import pandas as pd
import torch
import pytest

from model_core.dataset.loader import load_ellipticpp_v2, EllipticDataset
from model_core.dataset.split import temporal_split
from model_core.dataset.negatives import sample_hard_negatives


def test_load_ellipticpp_v2():
    import tempfile, os
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2014-01-01", periods=10, freq="D"),
        "feature_0": np.random.rand(10),
        "label": [0, 1, 2, 3, 4, 5, 0, 1, 2, -1],
    })
    path = Path(tmp) / "ellipticpp_v2.parquet"
    df.to_parquet(path)
    loaded = load_ellipticpp_v2(tmp)
    assert len(loaded) == 9
    assert loaded["label"].min() >= 0
    os.remove(path)
    os.rmdir(tmp)


def test_load_ellipticpp_v2_string_labels():
    import tempfile, os
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2014-01-01", periods=7, freq="D"),
        "feature_0": np.random.rand(7),
        "label": ["scam", "ransomware", "terrorist_financing", "sanctions", "mixer", "legitimate", "unknown"],
    })
    path = Path(tmp) / "ellipticpp_v2.parquet"
    df.to_parquet(path)
    loaded = load_ellipticpp_v2(tmp)
    assert len(loaded) == 6
    assert sorted(loaded["label"].tolist()) == [0, 1, 2, 3, 4, 5]
    os.remove(path)
    os.rmdir(tmp)


def test_elliptic_dataset():
    df = pd.DataFrame({
        "feature_0": [1.0, 2.0, 3.0],
        "label": [0, 1, 5],
    })
    ds = EllipticDataset(df, ["feature_0"])
    assert len(ds) == 3
    f, l = ds[0]
    assert f.shape == (1,) and l.item() in range(6)


def test_temporal_split():
    import tempfile, os, json
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    cfg_path = Path(tmp) / "split_config.json"
    cfg = {"temporal_split": {"train_end": "2015-12-31", "val_end": "2016-06-30", "test_end": "2016-12-31"}}
    cfg_path.write_text(json.dumps(cfg))
    dates = pd.date_range("2014-01-01", periods=1096, freq="D")
    df = pd.DataFrame({"timestamp": dates, "label": [0] * 548 + [5] * 548})
    train, val, test = temporal_split(df, str(cfg_path))
    assert train["timestamp"].max() <= pd.Timestamp("2015-12-31")
    assert val["timestamp"].min() > pd.Timestamp("2015-12-31")
    assert test["timestamp"].min() > pd.Timestamp("2016-06-30")
    os.remove(cfg_path)
    os.rmdir(tmp)


def test_temporal_split_no_shuffle_within_periods():
    import tempfile, os, json
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    cfg_path = Path(tmp) / "split_config.json"
    cfg = {"temporal_split": {"train_end": "2015-12-31", "val_end": "2016-06-30", "test_end": "2016-12-31"}}
    cfg_path.write_text(json.dumps(cfg))
    dates = pd.date_range("2014-01-01", periods=1096, freq="D")
    df = pd.DataFrame({"timestamp": dates, "label": list(range(1096))})
    train, val, test = temporal_split(df, str(cfg_path))
    for split in (train, val, test):
        assert split["timestamp"].is_monotonic_increasing
        assert split["label"].is_monotonic_increasing
    assert train["timestamp"].max() <= pd.Timestamp("2015-12-31")
    assert val["timestamp"].min() > pd.Timestamp("2015-12-31")
    assert test["timestamp"].min() > pd.Timestamp("2016-06-30")
    os.remove(cfg_path)
    os.rmdir(tmp)


def test_temporal_split_sorts_shuffled_input():
    import tempfile, os, json
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    cfg_path = Path(tmp) / "split_config.json"
    cfg = {"temporal_split": {"train_end": "2015-12-31", "val_end": "2016-06-30", "test_end": "2016-12-31"}}
    cfg_path.write_text(json.dumps(cfg))
    dates = pd.date_range("2014-01-01", periods=100, freq="D")
    df = pd.DataFrame({"timestamp": dates}).sample(frac=1.0, random_state=7).reset_index(drop=True)
    train, val, test = temporal_split(df, str(cfg_path))
    for split in (train, val, test):
        if len(split):
            assert split["timestamp"].is_monotonic_increasing
    os.remove(cfg_path)
    os.rmdir(tmp)


def test_sample_hard_negatives():
    np.random.seed(42)
    features = np.random.rand(100, 128)
    labels = np.array([0] * 10 + [5] * 90)
    idx = sample_hard_negatives(features, labels, k=5, threshold=0.5)
    assert len(idx) > 0
    idx2 = sample_hard_negatives(features, labels, k=5, threshold=0.5)
    assert len(idx2) > 0
