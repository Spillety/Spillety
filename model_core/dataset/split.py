import json
import pandas as pd
from pathlib import Path


def temporal_split(df: pd.DataFrame, config_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = json.loads(Path(config_path).read_text())
    train_end = pd.Timestamp(cfg["temporal_split"]["train_end"])
    val_end = pd.Timestamp(cfg["temporal_split"]["val_end"])
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    train = df[df["timestamp"] <= train_end].reset_index(drop=True)
    val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].reset_index(drop=True)
    test = df[df["timestamp"] > val_end].reset_index(drop=True)
    return train, val, test
