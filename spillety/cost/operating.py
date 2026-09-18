import numpy as np

_TIERS = ("tier1", "tier2", "tier3", "clear")


def cost_at(
    y_true: np.ndarray,
    scores: np.ndarray,
    tau: float,
    c_fp: float = 1.0,
    c_fn: float = 10.0,
) -> float:
    """
    ## Expected operating cost at threshold (§7.7.2)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    scores : np.ndarray
        Calibrated probabilities.
    tau : float
        Alert threshold.
    c_fp, c_fn : float
        False-positive / false-negative unit costs.

    Returns
    ----------
    float
        Cost(τ) = C_FP·FP(τ) + C_FN·FN(τ).
    """
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(scores) >= tau
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return float(c_fp * fp + c_fn * fn)


def alerts_at(scores: np.ndarray, tau: float) -> int:
    """
    ## Alert volume at threshold (§7.7.3)

    Parameters
    ----------
    scores : np.ndarray
        Calibrated probabilities.
    tau : float
        Alert threshold.

    Returns
    ----------
    int
        Alerts(τ) = |{i: score_i >= τ}|.
    """
    return int((np.asarray(scores) >= tau).sum())


def select_tau(
    y_true: np.ndarray,
    scores: np.ndarray,
    c_fp: float = 1.0,
    c_fn: float = 10.0,
    budget: int | None = None,
    taus: np.ndarray | None = None,
) -> float:
    """
    ## Cost-optimal threshold under analyst budget (§7.7.2–7.7.3)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    scores : np.ndarray
        Calibrated probabilities.
    c_fp, c_fn : float
        Unit costs.
    budget : int | None
        Max alerts; None disables the constraint.
    taus : np.ndarray | None
        Candidate thresholds; defaults to sorted unique scores.

    Returns
    ----------
    float
        Feasible τ with minimum Cost; ties break toward fewer alerts.
    """
    scores = np.asarray(scores, dtype=float)
    cand = np.unique(scores) if taus is None else np.asarray(taus, dtype=float)
    # Cost ties favor higher τ (fewer alerts); infeasible τ sorts after feasible.
    best, best_cost, best_alerts = float(cand.max()), float("inf"), None
    for tau in sorted(cand):
        alerts = alerts_at(scores, float(tau))
        if budget is not None and alerts > budget:
            continue
        cost = cost_at(y_true, scores, float(tau), c_fp, c_fn)
        if cost < best_cost or (cost == best_cost and (best_alerts is None or alerts < best_alerts)):
            best, best_cost, best_alerts = float(tau), cost, alerts
    return best


def assign_tier(
    score: float,
    tau1: float,
    tau2: float,
    tau3: float,
    causal_passed: bool,
    e_value: float,
    gamma: float,
) -> str:
    """
    ## Route score to Tier 1/2/3 or auto-clear (§7.7.4)

    Parameters
    ----------
    score : float
        Calibrated P(illicit).
    tau1, tau2, tau3 : float
        Blocking / manual-review / async-review thresholds.
    causal_passed : bool
        Upstream causal filter outcome.
    e_value, gamma : float
        Sensitivity metrics (E-value, Rosenbaum Γ*).

    Returns
    ----------
    str
        One of tier1 / tier2 / tier3 / clear.
    """
    # Tier 1 gate is conjunctive; scores above τ1 that fail it demote to tier2.
    if score > tau1 and bool(causal_passed) and e_value > 2.0 and gamma > 1.5:
        return "tier1"
    if score > tau2:
        return "tier2"
    if score > tau3:
        return "tier3"
    return "clear"
