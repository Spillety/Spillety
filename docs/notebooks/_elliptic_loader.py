from pathlib import Path

import pandas as pd


def load_elliptic(root: Path | str = "data/elliptic_raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ## Load Elliptic++ raw CSVs as immutable source

    Parameters
    ----------
    root : Path | str
        Directory containing the three CSVs (read-only).

    Returns
    ----------
    features : DataFrame
        Columns [txId, time_step, feat_2..feat_166] (167 cols, no header in source).
    classes : DataFrame
        Columns [txId, class] where class in {1, 2, unknown}.
    edgelist : DataFrame
        Columns [txId1, txId2].
    merged : DataFrame
        Inner join features+classes on txId, with time_step as int.
    """
    root = Path(root)
    if not root.exists():
        # fallback: archive.zip location
        alt = Path("archive.zip")
        raise FileNotFoundError(f"Elliptic root not found: {root}. Expected 3 CSVs or {alt}")

    # features has no header: col0=txId, col1=time_step, col2..166=features
    features = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    n_cols = features.shape[1]
    features.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(2, n_cols)]

    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    edgelist = pd.read_csv(root / "elliptic_txs_edgelist.csv")

    # ensure types without mutating source files
    features["time_step"] = features["time_step"].astype(int)
    merged = features.merge(classes, on="txId", how="inner")

    return features, classes, edgelist, merged


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "time_step",
    train_end: int = 30,
    valid_end: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ## Temporal split without shuffling

    Parameters
    ----------
    df : DataFrame
        Must contain `time_col`.
    time_col : str
        Column with integer time step.
    train_end : int
        Inclusive upper bound for train.
    valid_end : int
        Inclusive upper bound for validation; rest goes to test.
    """
    train = df[df[time_col] <= train_end].copy()
    valid = df[(df[time_col] > train_end) & (df[time_col] <= valid_end)].copy()
    test = df[df[time_col] > valid_end].copy()
    return train, valid, test
