import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins=10):
    # 10 бинов — стандартный компромисс: достаточно гранулярно для reliability,
    # но в каждом бине остаётся поддержка при N~5-10k (как в valid 31..40).
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == 0 else (y_prob > lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return float(ece)


def calibrate(base_estimator, X_valid, y_valid, X_eval=None, y_eval=None):
    # ponytail: beta calibration (Kull et al., 2017) — 3 параметра, лучше для сильного
    # дисбаланса; здесь sigmoid (Platt) как прокси — 2 параметра, устойчивее на малом valid.
    # Апгрейд — заменить method="sigmoid" на beta-калибровку при наличии реализации.
    # FrozenEstimator — современный эквивалент cv="prefit" (один калибровочный fit без CV).
    cal_iso = CalibratedClassifierCV(FrozenEstimator(base_estimator), method="isotonic")
    cal_sig = CalibratedClassifierCV(FrozenEstimator(base_estimator), method="sigmoid")
    cal_iso.fit(X_valid, y_valid)
    cal_sig.fit(X_valid, y_valid)

    def _metrics(y_true, y_prob):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        return {
            "ece": expected_calibration_error(y_true, y_prob, n_bins=10),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }

    calibrators = {"isotonic": cal_iso, "sigmoid": cal_sig}
    metrics = {}
    if X_eval is not None and y_eval is not None:
        for name, cal in calibrators.items():
            proba = cal.predict_proba(X_eval)[:, 1]
            metrics[name] = _metrics(y_eval, proba)
        proba_raw = base_estimator.predict_proba(X_eval)[:, 1]
        metrics["raw"] = _metrics(y_eval, proba_raw)
        # выбор лучшего: минимальный ECE, при равенстве — Brier
        best = min(calibrators, key=lambda k: (metrics[k]["ece"], metrics[k]["brier"]))
    else:
        best = "sigmoid"
        metrics = {}

    return {
        "isotonic": cal_iso,
        "sigmoid": cal_sig,
        "best": best,
        "best_estimator": calibrators[best],
        "metrics": metrics,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from spillety.data.loader import load_elliptic, temporal_split  # type: ignore

    from sklearn.metrics import average_precision_score
    from spillety.models.baseline import train_baseline

    DATA_ROOT = Path("data/elliptic_raw")
    if not DATA_ROOT.exists():
        DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "elliptic_raw"

    _, _, _, merged = load_elliptic(DATA_ROOT)
    df = merged[merged["class"].astype(str).isin(["1", "2"])].copy()
    df["y"] = (df["class"].astype(str) == "1").astype(int)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    train_df, valid_df, test_df = temporal_split(df, train_end=30, valid_end=40)

    X_train, y_train = train_df[feat_cols].values, train_df["y"].values
    X_valid, y_valid = valid_df[feat_cols].values, valid_df["y"].values
    X_test, y_test = test_df[feat_cols].values, test_df["y"].values

    base = train_baseline(X_train, y_train, kind="gbdt", random_state=72)
    result = calibrate(base, X_valid, y_valid, X_test, y_test)

    for name in ("raw", "isotonic", "sigmoid"):
        m = result["metrics"][name]
        print(f"{name:12s} ECE={m['ece']:.4f} Brier={m['brier']:.4f}")
    print(f"best={result['best']}")

    pr_raw = average_precision_score(y_test, base.predict_proba(X_test)[:, 1])
    pr_best = average_precision_score(y_test, result["best_estimator"].predict_proba(X_test)[:, 1])
    baseline = y_test.mean()
    print(f"raw PR-AUC={pr_raw:.4f} best PR-AUC={pr_best:.4f} baseline={baseline:.4f}")
    assert pr_best > baseline
    # калибровка не должна ухудшать ECE относительно raw
    assert result["metrics"][result["best"]]["ece"] <= result["metrics"]["raw"]["ece"] + 1e-9
