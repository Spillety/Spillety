import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score

CANDIDATE_NUM_LEAVES = (31, 63, 127)


def _scale_pos_weight(y_train):
    n_pos = int(np.sum(np.asarray(y_train) == 1))
    n_neg = int(np.sum(np.asarray(y_train) == 0))
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def _params(num_leaves, scale_pos_weight, random_state):
    return {
        "objective": "binary",
        "num_leaves": num_leaves,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": random_state,
        "deterministic": True,
    }


def select_num_leaves(
    X_train,
    y_train,
    X_holdout,
    y_holdout,
    *,
    candidates=CANDIDATE_NUM_LEAVES,
    n_boot=1000,
    random_state=72,
):
    """
    ## Pick num_leaves by PR-AUC with paired bootstrap CI and parsimony

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training matrix and binary labels.
    X_holdout, y_holdout : np.ndarray
        Hold-out temporal split for scoring.
    candidates : tuple
        Candidate leaf counts, simplest first.
    n_boot : int
        Bootstrap replications for PR-AUC gaps.
    random_state : int
        Seed for training and resampling.

    Returns
    ----------
    tuple[int, dict]
        Selected leaf count and per-candidate PR-AUC scores.
    """
    y_holdout = np.asarray(y_holdout).astype(int)
    weight = _scale_pos_weight(y_train)
    scores = {}
    for nl in candidates:
        booster = lgb.train(
            _params(nl, weight, random_state),
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=500,
        )
        scores[nl] = booster.predict(X_holdout)
    pr_auc = {
        nl: float(average_precision_score(y_holdout, scores[nl])) for nl in candidates
    }
    # Paired bootstrap: same resampled rows score every candidate, CI on the gaps.
    rng = np.random.default_rng(random_state)
    n = len(y_holdout)
    boot = np.full((n_boot, len(candidates)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_holdout[idx])) < 2:
            continue
        for j, nl in enumerate(candidates):
            boot[b, j] = average_precision_score(y_holdout[idx], scores[nl][idx])
    # Parsimony walk: adopt harder config only on significant PR-AUC gain (§7.4.2).
    best = 0
    for j in range(1, len(candidates)):
        if float(np.nanpercentile(boot[:, j] - boot[:, best], 2.5)) > 0.0:
            best = j
    return candidates[best], pr_auc


def train_gbdt(X_train, y_train, X_holdout, y_holdout, *, random_state=72, n_boot=1000):
    """
    ## Fit binary LightGBM with bootstrap-selected complexity

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training matrix and binary labels.
    X_holdout, y_holdout : np.ndarray
        Hold-out temporal split for `num_leaves` selection.
    random_state : int
        Seed for training and resampling.
    n_boot : int
        Bootstrap replications for the selection CI.

    Returns
    ----------
    lightgbm.Booster
        Fitted booster predicting P(illicit | x).
    """
    # ponytail: final fit on train only, refit on train+holdout when drift monitor (§7.10.2) lands.
    selected, _ = select_num_leaves(
        X_train, y_train, X_holdout, y_holdout, n_boot=n_boot, random_state=random_state
    )
    return lgb.train(
        _params(selected, _scale_pos_weight(y_train), random_state),
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=500,
    )
