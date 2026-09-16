import numpy as np
from sklearn.metrics import precision_recall_curve


class OptimalThresholdResult(float):
    """float tau* that also exposes cost diagnostics via mapping interface."""

    def __new__(cls, tau, mapping):
        obj = float.__new__(cls, tau)
        obj._mapping = mapping
        return obj

    def __getitem__(self, key):
        return self._mapping[key]

    def __contains__(self, key):
        return key in self._mapping

    def keys(self):
        return self._mapping.keys()

    def get(self, key, default=None):
        return self._mapping.get(key, default)

    def __iter__(self):
        return iter(self._mapping)

    @property
    def tau_star(self):
        return float(self)

    def to_dict(self):
        return dict(self._mapping)


def find_optimal_threshold(y_true, scores, C_FP=1, C_FN=10, budget=None):
    """
    Cost-optimal threshold via PR curve.

    Cost(tau) = C_FP*FP(tau) + C_FN*FN(tau), tau from precision_recall_curve.
    Budget constrains Alerts(tau) <= B — without it optimum drifts to tau->0
    when C_FN>>C_FP and queue explodes; budget forces feasible tradeoff.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    _, _, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns thresholds len = len(prec)-1; finite thresholds only
    if len(thresholds) == 0:
        # degenerate: single threshold fallback
        thresholds = np.array([0.5])

    costs, fps, fns, alerts = [], [], [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        costs.append(C_FP * fp + C_FN * fn)
        fps.append(fp)
        fns.append(fn)
        alerts.append(int((pred == 1).sum()))

    costs = np.array(costs)
    alerts = np.array(alerts)
    thresholds = np.array(thresholds)

    # feasible set under budget
    if budget is not None:
        # why budget: C_FN>>C_FP pushes tau* ->0 (alert everything) but analyst queue is finite
        # Alerts(tau) <= B enforces operational feasibility even at higher Cost
        feasible = alerts <= budget
        if np.any(feasible):
            idx = np.argmin(np.where(feasible, costs, np.inf))
        else:
            # no feasible tau -> minimal alerts (highest threshold) is least infeasible
            idx = int(np.argmin(alerts))
            # ponytail: alternative is quantile fallback (top-B alerts) but cost-based is preferred
    else:
        idx = int(np.argmin(costs))

    tau_star = float(thresholds[idx])
    mapping = {
        "tau_star": tau_star,
        "tau": tau_star,
        "cost": float(costs[idx]),
        "FP": int(fps[idx]),
        "FN": int(fns[idx]),
        "alerts": int(alerts[idx]),
        "thresholds": thresholds,
        "costs": costs,
        "fps": np.array(fps),
        "fns": np.array(fns),
        "alerts_all": alerts,
    }
    return OptimalThresholdResult(tau_star, mapping)


def tier_metrics(y_true, scores, tau, t_per_alert_h=0.25):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= tau).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    alerts = tp + fp
    fte_h = alerts * t_per_alert_h
    return {
        "tau": float(tau),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(prec),
        "recall": float(rec),
        "alerts": int(alerts),
        "FTE_h": float(fte_h),
        "FTE_days": float(fte_h / 8),
    }


def evaluate_tiers(y_true, scores, tiers, t_per_alert_h=0.25):
    """
    tiers: dict name -> tau
    returns DataFrame-like list of tier_metrics rows
    """
    rows = []
    for name, tau in tiers.items():
        m = tier_metrics(y_true, scores, tau, t_per_alert_h=t_per_alert_h)
        m["tier"] = name
        rows.append(m)
    return rows


def power_analysis(p, e, z=1.96):
    """
    Sample size for proportion estimate: n = z^2 * p(1-p) / e^2
    Random sampling from auto-clear pool is required for unbiased precision;
    selective sampling inflates risk and cannot be extrapolated.
    """
    p = float(p)
    e = float(e)
    z = float(z)
    if e == 0:
        return float("inf")
    # ponytail: exact Clopper-Pearson interval narrower for small p, here normal approximation
    n = (z**2 * p * (1 - p)) / (e**2)
    return float(n)


if __name__ == "__main__":
    # synthetic cost curve — mirrors notebook 06 without elliptic data
    np.random.seed(72)
    n = 5000
    y_true = (np.random.rand(n) < 0.1).astype(int)
    # scores: illicit higher mean
    scores = np.where(y_true == 1, np.random.beta(5, 2, n), np.random.beta(2, 5, n))

    for C_FN in [10, 100]:
        res = find_optimal_threshold(y_true, scores, C_FP=1, C_FN=C_FN, budget=None)
        print(f"C_FN={C_FN}: tau*={float(res):.4f} cost={res['cost']:.0f} alerts={res['alerts']} FP={res['FP']} FN={res['FN']}")

    # budget constraint demo: B=500
    res_b = find_optimal_threshold(y_true, scores, C_FP=1, C_FN=100, budget=500)
    print(f"budget B=500: tau*={float(res_b):.4f} alerts={res_b['alerts']} (feasible={res_b['alerts']<=500})")

    # tier metrics
    tiers = {
        "Tier 1 (C_FN=100)": float(find_optimal_threshold(y_true, scores, C_FP=1, C_FN=100)),
        "Tier 2 (C_FN=10)": float(find_optimal_threshold(y_true, scores, C_FP=1, C_FN=10)),
        "Tier 3 (0.10)": 0.10,
    }
    for name, tau in tiers.items():
        m = tier_metrics(y_true, scores, tau)
        print(f"{name} tau={tau:.4f} prec={m['precision']:.3f} rec={m['recall']:.3f} alerts={m['alerts']} FTE_h={m['FTE_h']:.1f}")

    # quantile comparison (ponytail: quantile is circular — threshold drifts with score distribution)
    q90 = float(np.quantile(scores, 0.90))
    qm = tier_metrics(y_true, scores, q90)
    print(f"Quantile 90% tau={q90:.4f} alerts={qm['alerts']} prec={qm['precision']:.3f}")

    # power analysis
    print(f"power n(p=0.05,e=0.01)={power_analysis(0.05, 0.01):.0f} (expected ~1825)")
    print(f"power n(p=0.05,e=0.005)={power_analysis(0.05, 0.005):.0f}")
