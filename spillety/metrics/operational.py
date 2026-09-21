import numpy as np

from spillety.cost.operating import select_tau

__all__ = [
    "alert_to_sar_rate",
    "bootstrap_ci",
    "cost_per_alert",
    "fp_rate",
    "latency_p99",
    "precision_at_k",
    "savings_vs_baseline",
    "ttd",
]


def ttd(t_signal, t_alert):
    """
    ## Detection delay per case (§12.5.3)

    Parameters
    ----------
    t_signal : array-like
        First on-chain signal timestamps.
    t_alert : array-like
        Alert timestamps.

    Returns
    ----------
    np.ndarray
        `t_alert − t_signal` per case, same unit as inputs.
    """
    return np.asarray(t_alert, dtype=float) - np.asarray(t_signal, dtype=float)


def alert_to_sar_rate(n_sar, n_alerts):
    """
    ## Share of alerts converted to SAR (§12.5.2)

    Parameters
    ----------
    n_sar : int
        Filed SAR count.
    n_alerts : int
        Alert count.

    Returns
    ----------
    float
        `n_sar / n_alerts`, 0.0 when there are no alerts.
    """
    if n_alerts == 0:
        return 0.0
    return float(n_sar) / float(n_alerts)


def fp_rate(y_true, y_pred):
    """
    ## False-positive share among alerts (§12.5.1)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_pred : np.ndarray
        Binary alert decisions.

    Returns
    ----------
    float
        `FP / (TP + FP)`, 0.0 when nothing was alerted.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    if tp + fp == 0:
        return 0.0
    return float(fp) / float(tp + fp)


def cost_per_alert(fte_cost, infra_cost, n_alerts):
    """
    ## Full handling cost of one alert (§12.6.3)

    Parameters
    ----------
    fte_cost : float
        Analyst FTE cost over the period.
    infra_cost : float
        Infrastructure cost over the period.
    n_alerts : int
        Alert count.

    Returns
    ----------
    float
        `(fte + infra) / n_alerts`, 0.0 when there are no alerts.
    """
    if n_alerts == 0:
        return 0.0
    return float(float(fte_cost) + float(infra_cost)) / float(n_alerts)


def latency_p99(latencies_ms):
    """
    ## 99th percentile of response latency (§12.5.4)

    Parameters
    ----------
    latencies_ms : np.ndarray
        Per-request latencies in milliseconds.

    Returns
    ----------
    float
        99th percentile.
    """
    return float(np.percentile(np.asarray(latencies_ms, dtype=float), 99))


def bootstrap_ci(values, stat, n_boot=1000, random_state=72):
    """
    ## Percentile bootstrap CI for a scalar statistic (§12.9)

    Parameters
    ----------
    values : np.ndarray
        Sample to resample.
    stat : callable
        Scalar statistic over a resample.
    n_boot : int
        Bootstrap replications.
    random_state : int
        Seed for resampling.

    Returns
    ----------
    dict
        `lower`, `upper` (2.5/97.5 percentiles), `mean` and `n_boot`.
    """
    # ponytail: i.i.d. percentile CI, upgrade path block bootstrap on temporal dependence.
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    n = len(values)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = stat(values[rng.integers(0, n, n)])
    return {
        "lower": float(np.percentile(reps, 2.5)),
        "upper": float(np.percentile(reps, 97.5)),
        "mean": float(np.mean(reps)),
        "n_boot": n_boot,
    }


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 100, n_bootstrap: int = 600, seed: int = 72) -> tuple[float, tuple[float, float]]:
    """
    ## Precision@K with bootstrap 95% CI (percentile method)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (1 = illicit).
    scores : np.ndarray
        Risk scores, higher = riskier.
    k : int
        Top-K to evaluate (reporting triplet: 100, 500, 1000).
    n_bootstrap : int
        Bootstrap replications.
    seed : int
        RNG seed for reproducibility.

    Returns
    ----------
    tuple[float, tuple[float, float]]
        (precision_at_k, (ci_lower, ci_upper)).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    k = min(k, n)
    if k == 0:
        return 0.0, (0.0, 0.0)

    order = np.argsort(scores)[::-1]
    topk_idx = order[:k]
    tp = int(y_true[topk_idx].sum())
    prec = float(tp) / float(k)

    # Bootstrap CI (percentile method)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        s_b = scores[idx]
        ob = np.argsort(s_b)[::-1]
        topk_b = ob[:k]
        tp_b = int(y_b[topk_b].sum())
        reps[b] = float(tp_b) / float(k)

    ci_lower = float(np.percentile(reps, 2.5))
    ci_upper = float(np.percentile(reps, 97.5))
    return prec, (ci_lower, ci_upper)


def savings_vs_baseline(
    y_true: np.ndarray,
    scores_model: np.ndarray,
    scores_baseline: np.ndarray,
    c_fp: float = 1.0,
    c_fn: float = 10.0,
) -> dict:
    """
    ## Cost savings of model vs rule-based baseline

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (1 = illicit).
    scores_model : np.ndarray
        Model risk scores (higher = riskier).
    scores_baseline : np.ndarray
        Baseline rule-based scores (higher = riskier).
    c_fp, c_fn : float
        Unit costs for FP and FN.

    Returns
    ----------
    dict
        Keys: gain, cost_fp_model, cost_fn_model, cost_baseline,
        fp_prevented, n_model, n_baseline.
        gain = cost_baseline - cost_model (positive = savings).
    """
    y_true = np.asarray(y_true).astype(int)
    scores_model = np.asarray(scores_model, dtype=float)
    scores_baseline = np.asarray(scores_baseline, dtype=float)

    # Model: cost-optimal threshold (unconstrained budget -> optimal by cost)
    tau_model = select_tau(y_true, scores_model, c_fp=c_fp, c_fn=c_fn, budget=None)
    pred_model = scores_model >= tau_model
    fp_model = int(((pred_model == 1) & (y_true == 0)).sum())
    fn_model = int(((pred_model == 0) & (y_true == 1)).sum())
    cost_model = float(c_fp * fp_model + c_fn * fn_model)
    n_model = int(pred_model.sum())

    # Baseline: fixed threshold at 0.5 (rule-based)
    tau_baseline = 0.5
    pred_baseline = scores_baseline >= tau_baseline
    fp_baseline = int(((pred_baseline == 1) & (y_true == 0)).sum())
    fn_baseline = int(((pred_baseline == 0) & (y_true == 1)).sum())
    cost_baseline = float(c_fp * fp_baseline + c_fn * fn_baseline)
    n_baseline = int(pred_baseline.sum())

    gain = cost_baseline - cost_model
    fp_prevented = fp_baseline - fp_model

    return {
        "gain": float(gain),
        "cost_fp_model": float(c_fp * fp_model),
        "cost_fn_model": float(c_fn * fn_model),
        "cost_baseline": float(cost_baseline),
        "fp_prevented": int(fp_prevented),
        "n_model": n_model,
        "n_baseline": n_baseline,
    }
