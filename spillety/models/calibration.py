import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-15


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def _as_scores(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D scores, got shape {arr.shape}")
    return arr


def ece_score(y_true, y_prob, n_bins=10):
    """
    ## Expected calibration error over uniform bins (§7.6.2)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Bin count M.

    Returns
    ----------
    float
        Volume-weighted |acc − conf| over non-empty bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for m in range(n_bins):
        lo, hi = edges[m], edges[m + 1]
        mask = (
            (y_prob >= lo) & (y_prob <= hi)
            if m == 0
            else (y_prob > lo) & (y_prob <= hi)
        )
        if not np.any(mask):
            continue
        ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return float(ece)


def brier_score(y_true, y_prob):
    """
    ## Mean squared deviation of probabilities from outcomes (§7.6.1)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    ----------
    float
        Brier score, lower is better.
    """
    return float(
        np.mean(
            (np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2
        )
    )


def reliability_stats(y_true, y_prob, n_bins=10):
    """
    ## Per-bin accuracy, confidence and counts for the reliability diagram (§7.6.3)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Bin count M.

    Returns
    ----------
    dict
        Lists `lo`, `hi`, `acc`, `conf`, `count` of length M; `acc`/`conf`
        are NaN for empty bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    stats = {"lo": [], "hi": [], "acc": [], "conf": [], "count": []}
    for m in range(n_bins):
        lo, hi = float(edges[m]), float(edges[m + 1])
        mask = (
            (y_prob >= lo) & (y_prob <= hi)
            if m == 0
            else (y_prob > lo) & (y_prob <= hi)
        )
        stats["lo"].append(lo)
        stats["hi"].append(hi)
        stats["count"].append(int(mask.sum()))
        stats["acc"].append(
            float(y_true[mask].mean()) if np.any(mask) else float("nan")
        )
        stats["conf"].append(
            float(y_prob[mask].mean()) if np.any(mask) else float("nan")
        )
    return stats


class IsotonicCalibrator(BaseEstimator):
    """
    ## Monotone stepwise recalibration of raw scores (§7.5.2)

    Parameters
    ----------
    None

    Returns
    ----------
    IsotonicCalibrator
        Fitted estimator with `predict_proba`.
    """

    def fit(self, X, y):
        self.ir_ = IsotonicRegression(out_of_bounds="clip")
        self.ir_.fit(_as_scores(X), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X):
        # Clip keeps test scores outside the validation range on the edge step, not NaN.
        q = _clip(self.ir_.predict(_as_scores(X)))
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class BetaCalibrator(BaseEstimator):
    """
    ## Parametric beta recalibration with MLE on validation (§7.5.2)

    Parameters
    ----------
    None

    Returns
    ----------
    BetaCalibrator
        Fitted estimator with `a_`, `b_`, `c_` and `predict_proba`.
    """

    def fit(self, X, y):
        p = _clip(_as_scores(X))
        target = np.asarray(y, dtype=float)
        logit = np.log(p / (1 - p))
        logp = np.log(p)

        def _nll(theta):
            a, b, c = theta
            q = _clip(1.0 / (1.0 + np.exp(-a * logit - b * logp - c)))
            return float(-np.mean(target * np.log(q) + (1 - target) * np.log(1 - q)))

        self.a_, self.b_, self.c_ = minimize(_nll, (1.0, 0.0, 0.0), method="L-BFGS-B").x
        return self

    def predict_proba(self, X):
        p = _clip(_as_scores(X))
        q = _clip(
            1.0
            / (
                1.0
                + np.exp(-self.a_ * np.log(p / (1 - p)) - self.b_ * np.log(p) - self.c_)
            )
        )
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _bootstrap_deltas(y_true, p_iso, p_beta, n_boot, random_state):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    d_ece = np.empty(n_boot)
    d_brier = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d_ece[b] = ece_score(y_true[idx], p_iso[idx]) - ece_score(
            y_true[idx], p_beta[idx]
        )
        d_brier[b] = brier_score(y_true[idx], p_iso[idx]) - brier_score(
            y_true[idx], p_beta[idx]
        )
    return {
        "ece_lower": float(np.percentile(d_ece, 2.5)),
        "ece_upper": float(np.percentile(d_ece, 97.5)),
        "brier_lower": float(np.percentile(d_brier, 2.5)),
        "brier_upper": float(np.percentile(d_brier, 97.5)),
    }


def calibrate(p_valid, y_valid, p_test, y_test, *, n_boot=1000, random_state=72):
    """
    ## Fit isotonic vs beta on validation, pick by test ECE/Brier bootstrap (§7.5.3)

    Parameters
    ----------
    p_valid, y_valid : np.ndarray
        Validation scores and labels for fitting both calibrators.
    p_test, y_test : np.ndarray
        Test scores and labels for scoring and the bootstrap CI.
    n_boot : int
        Bootstrap replications for ΔECE/ΔBrier.
    random_state : int
        Seed for resampling.

    Returns
    ----------
    dict
        `isotonic`/`beta` estimators, `best` name, `best_estimator`,
        per-method `metrics` and the `bootstrap` CI with `n_boot`.
    """
    iso = IsotonicCalibrator().fit(p_valid, y_valid)
    beta = BetaCalibrator().fit(p_valid, y_valid)
    y_test = np.asarray(y_test).astype(int)
    q_iso = iso.predict_proba(p_test)[:, 1]
    q_beta = beta.predict_proba(p_test)[:, 1]
    metrics = {
        "isotonic": {
            "ece": ece_score(y_test, q_iso),
            "brier": brier_score(y_test, q_iso),
        },
        "beta": {
            "ece": ece_score(y_test, q_beta),
            "brier": brier_score(y_test, q_beta),
        },
    }
    ci = _bootstrap_deltas(y_test, q_iso, q_beta, n_boot, random_state)
    # Δ = iso − beta: upper < 0 → isotonic wins, else beta (tie defaults to beta, §7.5.3).
    best = "isotonic" if ci["ece_upper"] < 0.0 else "beta"
    # Brier guard: fall back when the ECE winner is significantly worse on Brier.
    if best == "isotonic" and ci["brier_lower"] > 0.0:
        best = "beta"
    elif best == "beta" and ci["brier_upper"] < 0.0:
        best = "isotonic"
    return {
        "isotonic": iso,
        "beta": beta,
        "best": best,
        "best_estimator": iso if best == "isotonic" else beta,
        "metrics": metrics,
        "bootstrap": {**ci, "n_boot": n_boot},
    }
