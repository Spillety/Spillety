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

# Numeric ids pass through unchanged; stringified ints accepted for parquet variance.
LABEL_MAP_FULL = {**LABEL_MAP, **{i: i for i in range(NUM_CLASSES)}, **{str(i): i for i in range(NUM_CLASSES)}}


def load_ellipticpp_v2(data_path: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(data_path) / "ellipticpp_v2.parquet")
    df["label"] = df["label"].map(LABEL_MAP_FULL).fillna(-1).astype(int)
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
