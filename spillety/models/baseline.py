import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def train_baseline(X_train, y_train, kind="gbdt", random_state=72):
    # ponytail: GradientBoosting как легковесный аналог LightGBM — без зависимости lightgbm,
    # hist/leaf-wise быстрее и экономичнее, апгрейд — заменить на LGBMClassifier без смены интерфейса.
    if kind == "gbdt":
        # GradientBoosting не поддерживает class_weight, делаем balanced через sample_weight
        classes, counts = np.unique(y_train, return_counts=True)
        n = len(y_train)
        weight_map = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([weight_map[y] for y in y_train])
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=random_state
        )
        clf.fit(X_train, y_train)
        return clf
    raise ValueError(f"unknown kind={kind!r}, expected 'gbdt' or 'rf'")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from spillety.data.loader import load_elliptic, temporal_split  # type: ignore

    from sklearn.metrics import average_precision_score

    DATA_ROOT = Path("data/elliptic_raw")
    if not DATA_ROOT.exists():
        DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "elliptic_raw"

    _, _, _, merged = load_elliptic(DATA_ROOT)
    df = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
    df["y"] = (df["class"].astype(str) == "1").astype(int)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    train_df, valid_df, test_df = temporal_split(df, train_end=30, valid_end=40)

    X_train = train_df[feat_cols].values
    y_train = train_df["y"].values
    X_test = test_df[feat_cols].values
    y_test = test_df["y"].values

    clf = train_baseline(X_train, y_train, kind="gbdt", random_state=72)
    proba = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    baseline = y_test.mean()
    print(f"PR-AUC={pr_auc:.4f} baseline={baseline:.4f} lift={pr_auc / baseline:.1f}x")
    assert pr_auc > baseline, f"PR-AUC {pr_auc:.4f} not above baseline {baseline:.4f}"
    # sanity: predict_proba contract
    assert proba.shape == y_test.shape
    assert (proba >= 0).all() and (proba <= 1).all()
