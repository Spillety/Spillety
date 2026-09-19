import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import fbeta_score


def focal_loss_objective(gamma=2.0, alpha=0.25):
    """
    ## Focal loss objective for LightGBM: focuses on hard examples (§7.4)

    Parameters
    ----------
    gamma : float
        Focusing parameter; higher gamma down-weights well-classified examples.
    alpha : float
        Weighting factor for the rare class (illicit).

    Returns
    ----------
    callable
        LightGBM custom objective function (grad, hess).
    """
    def objective(y_pred, dataset):
        y_true = np.asarray(dataset.get_label()).astype(float)
        y_pred = np.asarray(y_pred).astype(float)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        pt = np.where(y_true == 1, p, 1 - p)
        grad = -alpha * (1 - pt) ** gamma * (y_true - p)
        hess = alpha * (1 - pt) ** gamma * p * (1 - p) * (gamma * (y_true - p) + 1)
        return grad, hess
    return objective


def train_gbdt_focal(
    X_train,
    y_train,
    X_holdout,
    y_holdout,
    *,
    random_state=72,
    n_boot=1000,
    gamma=2.0,
    alpha=0.25,
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=500,
):
    """
    ## Train LightGBM with focal loss objective

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training matrix and binary labels.
    X_holdout, y_holdout : np.ndarray
        Hold-out temporal split for validation.
    random_state : int
        Seed for training.
    n_boot : int
        Bootstrap replications (unused here, kept for API compatibility).
    gamma : float
        Focal loss gamma.
    alpha : float
        Focal loss alpha.
    num_leaves : int
        Tree complexity.
    learning_rate : float
        Learning rate.
    n_estimators : int
        Number of boosting rounds.

    Returns
    ----------
    tuple[lightgbm.Booster, IsotonicRegression, float]
        Fitted booster, calibrated isotonic regressor, optimal F-beta threshold.
    """
    y_train = np.asarray(y_train).astype(int)
    y_holdout = np.asarray(y_holdout).astype(int)

    params = {
        "objective": focal_loss_objective(gamma, alpha),
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": random_state,
        "deterministic": True,
    }

    booster = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=n_estimators,
    )

    holdout_probs = booster.predict(X_holdout)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(holdout_probs, y_holdout)
    calibrated = calibrator.predict(holdout_probs)

    best_thresh = 0.5
    best_fbeta = -1.0
    for thresh in np.linspace(0.01, 0.99, 99):
        preds = (calibrated >= thresh).astype(int)
        fb = fbeta_score(y_holdout, preds, beta=2.0, zero_division=0)
        if fb > best_fbeta:
            best_fbeta = fb
            best_thresh = thresh

    return booster, calibrator, float(best_thresh)


def threshold_moving_fbeta(
    y_true,
    y_prob,
    beta=2.0,
    min_recall=None,
):
    """
    ## Find threshold maximizing F-beta (recall-weighted) or satisfying min_recall

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Calibrated probabilities.
    beta : float
        Beta for F-beta; beta > 1 favors recall.
    min_recall : float | None
        If set, return smallest threshold achieving at least this recall.

    Returns
    ----------
    float
        Optimal threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if min_recall is not None:
        for thresh in np.linspace(0.0, 1.0, 1001):
            preds = (y_prob >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fn = np.sum((preds == 0) & (y_true == 1))
            if tp + fn == 0:
                continue
            recall = tp / (tp + fn)
            if recall >= min_recall:
                return float(thresh)
        return 1.0

    best_thresh = 0.5
    best_fbeta = -1.0
    for thresh in np.linspace(0.01, 0.99, 99):
        preds = (y_prob >= thresh).astype(int)
        fb = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if fb > best_fbeta:
            best_fbeta = fb
            best_thresh = thresh
    return float(best_thresh)