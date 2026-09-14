import json
import pandas as pd
import numpy as np
from pathlib import Path


def temporal_split(df: pd.DataFrame, config_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = json.loads(Path(config_path).read_text())
    train_end = pd.Timestamp(cfg["temporal_split"]["train_end"])
    val_end = pd.Timestamp(cfg["temporal_split"]["val_end"])
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    train = df[df["timestamp"] <= train_end]
    val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test = df[df["timestamp"] > val_end]
    rng = np.random.default_rng(42)
    for split in [train, val, test]:
        if len(split) > 0:
            split_idx = rng.permutation(len(split))
            split.iloc[:] = split.iloc[split_idx].reset_index(drop=True)
    return train, val, test


def demo() -> None:
    import tempfile, os, json
    tmp = tempfile.mkdtemp()
    cfg_path = Path(tmp) / "split_config.json"
    cfg = {"temporal_split": {"train_end": "2015-12-31", "val_end": "2016-06-30", "test_end": "2016-12-31"}}
    cfg_path.write_text(json.dumps(cfg))

    dates = pd.date_range("2014-01-01", periods=1000, freq="D")
    df = pd.DataFrame({"timestamp": dates, "label": np.random.randint(0, 6, 1000), "address": np.random.randint(0, 50, 1000)})
    train, val, test = temporal_split(df, str(cfg_path))

    assert (len(train) > 0 and len(val) > 0 and len(test) > 0)
    assert train["timestamp"].max() <= pd.Timestamp("2015-12-31")
    assert val["timestamp"].min() > pd.Timestamp("2015-12-31")
    assert test["timestamp"].min() > pd.Timestamp("2016-06-30")

    os.remove(cfg_path)
    os.rmdir(tmp)
    print("split demo passed")


if __name__ == "__main__":
    demo()
