import numpy as np

__all__ = [
    "alert_to_sar_rate",
    "bootstrap_ci",
    "cost_per_alert",
    "fp_rate",
    "latency_p99",
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
