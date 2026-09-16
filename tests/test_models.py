from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from spillety.data.loader import load_elliptic, temporal_split
from spillety.models.baseline import train_baseline
from spillety.models.calibration import calibrate


def test_baseline_pr_auc_above_random_and_calibration_improves_ece():
    root = Path("data/elliptic_raw")
    if not root.exists():
        root = Path(__file__).resolve().parents[1] / "data" / "elliptic_raw"
    _, _, _, merged = load_elliptic(root)
    df = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
    df["y"] = (df["class"].astype(str) == "1").astype(int)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    train_df, valid_df, test_df = temporal_split(df, train_end=30, valid_end=40)

    X_train, y_train = train_df[feat_cols].values, train_df["y"].values
    X_valid, y_valid = valid_df[feat_cols].values, valid_df["y"].values
    X_test, y_test = test_df[feat_cols].values, test_df["y"].values

    clf = train_baseline(X_train, y_train, kind="gbdt", random_state=72)
    proba = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    assert pr_auc > y_test.mean(), f"PR-AUC {pr_auc:.4f} not above baseline {y_test.mean():.4f}"

    result = calibrate(clf, X_valid, y_valid, X_test, y_test)
    assert result["best"] in ("isotonic", "sigmoid")
    assert result["metrics"][result["best"]]["ece"] <= result["metrics"]["raw"]["ece"]
