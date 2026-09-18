import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

__all__ = [
    "brier_score",
    "copula_proba",
    "ece_score",
    "estimate_prior",
    "fit_copula_theta",
    "fit_logistic",
    "pr_auc",
    "select_fusion",
    "to_pseudo_obs",
]

_EPS = 1e-6
_TOL = 1e-12
_PARSIMONY = ("logistic", "frank", "clayton", "gumbel")

_THETA_GRIDS = {
    "clayton": (0.2, 0.5, 1.0, 2.0, 5.0),
    "gumbel": (1.1, 1.5, 2.0, 3.0, 5.0),
    "frank": (0.5, 1.0, 2.0, 5.0, 9.0),
}


def estimate_prior(y_train: np.ndarray) -> float:
    """
    ## Empirical P(c=1) from the train time-slice (§5.4.2)

    Parameters
    ----------
    y_train : np.ndarray
        Binary pair labels of the train slice (steps 1–35, caller-sliced).

    Returns
    ----------
    float
        Clipped mean rate in [eps, 1 - eps].
    """
    y = np.asarray(y_train).astype(int).ravel()
    if y.size == 0:
        raise ValueError("y_train is empty")
    return float(np.clip(y.mean(), _EPS, 1.0 - _EPS))


def to_pseudo_obs(x: np.ndarray) -> np.ndarray:
    """
    ## Rank-transform columns to pseudo-observations in (0, 1)

    Parameters
    ----------
    x : np.ndarray
        Signal matrix (n, k).

    Returns
    ----------
    np.ndarray
        Ranks / (n + 1), clipped away from exact 0/1.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected 2D signal matrix, got {a.ndim}D")
    n = a.shape[0]
    ranks = np.empty_like(a)
    for j in range(a.shape[1]):
        order = np.argsort(a[:, j], kind="stable")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(1, n + 1) / (n + 1)
        ranks[:, j] = r
    return np.clip(ranks, _EPS, 1.0 - _EPS)


def _copula_pdf(u: np.ndarray, family: str, theta: float) -> np.ndarray:
    # ponytail: bivariate on the first two signals only, upgrade to k-dim vine with §5.3.
    uu = np.clip(np.asarray(u, dtype=float)[:, 0], _EPS, 1.0 - _EPS)
    vv = np.clip(np.asarray(u, dtype=float)[:, 1], _EPS, 1.0 - _EPS)
    if family == "clayton":
        if theta <= 0:
            raise ValueError("clayton theta must be > 0")
        term = uu**-theta + vv**-theta - 1.0
        return (1.0 + theta) * (uu * vv) ** (-theta - 1.0) * term ** (-1.0 / theta - 2.0)
    if family == "gumbel":
        if theta < 1:
            raise ValueError("gumbel theta must be >= 1")
        # Closed-form bivariate Gumbel density via s = tx^θ + ty^θ.
        tx, ty = -np.log(uu), -np.log(vv)
        s = tx**theta + ty**theta
        root = s ** (1.0 / theta)
        c = np.exp(-root) * (tx * ty) ** (theta - 1.0) / (uu * vv)
        return c * s ** (2.0 / theta - 2.0) * (root + theta - 1.0)
    if family == "frank":
        if theta == 0:
            raise ValueError("frank theta must be != 0")
        e = np.exp(-theta)
        num = theta * (1.0 - e) * np.exp(-theta * (uu + vv))
        den = (1.0 - e - (1.0 - np.exp(-theta * uu)) * (1.0 - np.exp(-theta * vv))) ** 2
        return num / np.maximum(den, 1e-300)
    raise ValueError(f"unknown family {family}")


def fit_copula_theta(u: np.ndarray, family: str) -> float:
    """
    ## Grid MLE of the copula dependence parameter (§5.3.2)

    Parameters
    ----------
    u : np.ndarray
        Pseudo-observations (n, 2) for one class.
    family : str
        One of clayton, gumbel, frank.

    Returns
    ----------
    float
        Grid value with max log-likelihood; first wins ties.
    """
    if family not in _THETA_GRIDS:
        raise ValueError(f"unknown family {family}")
    a = np.asarray(u, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("copula fit needs an (n, 2) matrix")
    best_t, best_ll = _THETA_GRIDS[family][0], -np.inf
    for t in _THETA_GRIDS[family]:
        ll = float(np.sum(np.log(np.maximum(_copula_pdf(a, family, t), 1e-300))))
        if ll > best_ll + _TOL:
            best_t, best_ll = t, ll
    return best_t


def copula_proba(
    u: np.ndarray,
    family: str,
    theta1: float,
    theta0: float,
    prior: float,
) -> np.ndarray:
    """
    ## Posterior P(c=1 | s) under a bivariate copula (§5.3.2)

    Parameters
    ----------
    u : np.ndarray
        Pseudo-observations (n, 2).
    family : str
        One of clayton, gumbel, frank.
    theta1 : float
        Dependence parameter fitted on c=1 pairs.
    theta0 : float
        Dependence parameter fitted on c=0 pairs.
    prior : float
        P(c=1) estimated on the train slice (steps 1–35, caller-sliced).

    Returns
    ----------
    np.ndarray
        Posterior probabilities, clipped to [eps, 1 - eps].
    """
    a = np.asarray(u, dtype=float)
    c1 = _copula_pdf(a, family, theta1)
    c0 = _copula_pdf(a, family, theta0)
    p = float(np.clip(prior, _EPS, 1.0 - _EPS))
    post = p * c1 / (p * c1 + (1.0 - p) * c0 + 1e-300)
    return np.clip(post, _EPS, 1.0 - _EPS)


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    ## Logistic fusion baseline σ(β0 + Σβi·si) (§5.3.3)

    Parameters
    ----------
    x_train : np.ndarray
        Signal matrix (n, k) of the train slice.
    y_train : np.ndarray
        Binary pair labels of the train slice.

    Returns
    ----------
    LogisticRegression
        Fitted lbfgs model, deterministic for fixed input.
    """
    # ponytail: L1/L2/lambda grid lives in §5.7.1, add when tuned.
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(np.asarray(x_train, dtype=float), np.asarray(y_train).astype(int))
    return model


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ## PR-AUC with single-class guard (§5.7.2)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score : np.ndarray
        Predicted probabilities.

    Returns
    ----------
    float
        Average precision, 0.0 when labels are single-class.
    """
    y = np.asarray(y_true).astype(int).ravel()
    if len(np.unique(y)) < 2:
        return 0.0
    return float(average_precision_score(y, np.asarray(y_score, dtype=float).ravel()))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    ## Mean squared probability error (§5.7.2)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    ----------
    float
        Mean((y - p) ** 2).
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((y - p) ** 2))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    ## Expected calibration error over M=15 bins (§5.7.2)

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
        Weighted mean |acc - conf|.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / len(p)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def select_fusion(y_valid: np.ndarray, candidates: dict[str, np.ndarray]) -> dict:
    """
    ## Pick fusion by max PR-AUC, then min ECE/Brier (§5.7.3)

    Parameters
    ----------
    y_valid : np.ndarray
        Binary labels of the validation slice (steps 36–42, caller-sliced).
    candidates : dict[str, np.ndarray]
        Name -> predicted probabilities on validation.

    Returns
    ----------
    dict
        `best` name plus per-candidate `pr_auc`, `ece`, `brier` table.
    """
    # ponytail: strict max-PR-AUC ranking; add the §5.7.3 0.01-filter when tuned.
    if not candidates:
        raise ValueError("candidates is empty")
    rows: dict[str, dict[str, float]] = {}
    for name, proba in candidates.items():
        p = np.asarray(proba, dtype=float).ravel()
        rows[name] = {
            "pr_auc": pr_auc(y_valid, p),
            "ece": ece_score(y_valid, p),
            "brier": brier_score(y_valid, p),
        }

    def key(name: str) -> tuple:
        r = rows[name]
        rank = _PARSIMONY.index(name) if name in _PARSIMONY else len(_PARSIMONY)
        return (-r["pr_auc"], r["ece"], r["brier"], rank, name)

    names = sorted(rows, key=key)
    return {"best": names[0], "metrics": rows}
