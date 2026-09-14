import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

LABEL_MAP = {
    "scam": 0, "ransomware": 1, "terrorist_financing": 2,
    "sanctions": 3, "mixer": 4, "legitimate": 5,
}
NUM_CLASSES = 6


def load_ellipticpp_v2(data_path: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(data_path) / "ellipticpp_v2.parquet")
    df["label"] = df["label"].map(LABEL_MAP).fillna(-1).astype(int)
    df = df[df["label"] >= 0].reset_index(drop=True)
    return df


class EllipticDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        self._features = df[feature_cols].values.astype(np.float32)
        self._labels = df["label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self._features[idx]), torch.tensor(self._labels[idx])


def demo() -> None:
    import tempfile, os
    tmp = tempfile.mkdtemp()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2014-01-01", periods=100, freq="D"),
        "feature_0": np.random.rand(100),
        "label": [0] * 50 + [5] * 50,
    })
    path = Path(tmp) / "ellipticpp_v2.parquet"
    df.to_parquet(path)
    loaded = load_ellipticpp_v2(tmp)
    assert len(loaded) == 100, "All labels valid"
    assert loaded["label"].min() >= 0, "No label=-1"

    ds = EllipticDataset(loaded, ["feature_0"])
    assert len(ds) == 100
    f, l = ds[0]
    assert f.shape == (1,) and l.item() in range(6)
    os.remove(path)
    os.rmdir(tmp)
    print("loader demo passed")


if __name__ == "__main__":
    demo()
