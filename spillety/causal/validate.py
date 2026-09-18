import numpy as np
from scipy import stats


def _residuals(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(z, a, rcond=None)
    return a - z @ coef


def partial_corr_pvalue(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None
) -> tuple[float, float]:
    """
    ## Partial correlation and Fisher-z p-value (§6.3.1)

    Parameters
    ----------
    x, y : np.ndarray
        Paired samples, shape (n,).
    z : np.ndarray | None
        Conditioning set, shape (n, p); None tests marginal correlation.

    Returns
    ----------
    tuple[float, float]
        Partial correlation r and two-sided p-value for H0: r = 0.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if z is None:
        r = float(np.corrcoef(x, y)[0, 1])
        dof = n - 2
    else:
        zm = np.asarray(z, dtype=float)
        if zm.ndim == 1:
            zm = zm.reshape(-1, 1)
        # ponytail: linear OLS residualization as CMI proxy; nonlinear
        # upgrade via kNN-CMI when non-monotone effects matter (§6.3.1).
        z1 = np.column_stack([np.ones(n), zm - zm.mean(axis=0)])
        rx, ry = _residuals(x, z1), _residuals(y, z1)
        denom = float(np.sqrt((rx @ rx) * (ry @ ry)))
        r = float((rx @ ry) / denom) if denom > 0 else 0.0
        dof = n - 2 - zm.shape[1]
    if not np.isfinite(r) or dof < 1:
        return 0.0, 1.0
    r = float(np.clip(r, -0.999999, 0.999999))
    zstat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(dof - 1)
    return r, float(2 * stats.norm.sf(abs(zstat)))


def ci_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    random_state: int = 72,
) -> dict:
    """
    ## Conditional-independence test with bootstrap p-value CI (§6.3.1)

    Parameters
    ----------
    x, y : np.ndarray
        Paired samples, shape (n,).
    z : np.ndarray | None
        Conditioning set, shape (n, p).
    alpha : float
        Significance level; H0 (independence) rejected below it.
    n_boot : int
        Bootstrap replications for the p-value interval.
    random_state : int
        Seed for resampling.

    Returns
    ----------
    dict
        `r`, `p`, `p_boot` (median), `p_ci` (95% interval), `reject`
        (True only if p and the whole CI sit below alpha).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    r, p = partial_corr_pvalue(x, y, z)
    rng = np.random.default_rng(random_state)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        zb = None if z is None else np.asarray(z)[idx]
        boot[b] = partial_corr_pvalue(x[idx], y[idx], zb)[1]
    lo, med, hi = (float(v) for v in np.quantile(boot, [0.025, 0.5, 0.975]))
    return {
        "r": r,
        "p": p,
        "p_boot": med,
        "p_ci": (lo, hi),
        "reject": bool(p < alpha and hi < alpha),
    }


def falsification_test(
    y: np.ndarray, x_test: np.ndarray, controls: np.ndarray | None = None
) -> dict:
    """
    ## Falsification of a DAG-implied absent edge (§6.3.2)

    Parameters
    ----------
    y : np.ndarray
        Outcome, shape (n,).
    x_test : np.ndarray
        Variable the DAG claims is conditionally independent of y.
    controls : np.ndarray | None
        Conditioning set, shape (n, p).

    Returns
    ----------
    dict
        OLS `coef`, `se`, `t`, `p` for H0: β(x_test) = 0 and `reject`
        (True means the absent edge is falsified, p < 0.05).
    """
    y = np.asarray(y, dtype=float).ravel()
    xt = np.asarray(x_test, dtype=float).ravel()
    n = y.shape[0]
    base = np.ones((n, 1))
    if controls is not None:
        cm = np.asarray(controls, dtype=float)
        if cm.ndim == 1:
            cm = cm.reshape(-1, 1)
        base = np.column_stack([base, cm])
    design = np.column_stack([base, xt])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    dof = n - rank
    resid = y - design @ coef
    # ponytail: homoskedastic SE; heteroskedasticity-robust (HC3) upgrade
    # if residual-variance structure ever matters (§6.3.2 follow-up).
    s2 = float(resid @ resid / dof) if dof > 0 else float("nan")
    try:
        cov = s2 * np.linalg.inv(design.T @ design)
        se = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    beta = float(coef[-1])
    if not np.isfinite(se) or se == 0 or dof < 1:
        return {"coef": beta, "se": se, "t": 0.0, "p": 1.0, "reject": False}
    t = beta / se
    p = float(2 * stats.t.sf(abs(t), dof))
    return {"coef": beta, "se": se, "t": float(t), "p": p, "reject": bool(p < 0.05)}
