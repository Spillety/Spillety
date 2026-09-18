import numpy as np
import pandas as pd


def _counts_per_step(df: pd.DataFrame, time_col: str = "time_step") -> pd.Series:
    counts = df.groupby(time_col).size()
    # Elliptic is 1..49 — reindex for continuity if within that range
    # ponytail: MLE for Hawkes (scipy.optimize on train 1..30) → fit alpha/beta/mu instead of fixed 0.5/1.0
    if counts.index.min() >= 1 and counts.index.max() <= 49:
        counts = counts.reindex(range(1, 50), fill_value=0)
    return counts


def burstiness_series(counts: pd.Series, window: int = 3) -> pd.Series:
    burst: dict[int, float] = {}
    idx = counts.index.tolist()
    lo = min(idx) if idx else 1
    for t in idx:
        vals = counts.loc[max(lo, t - window + 1) : t].values
        mu_w = float(vals.mean()) if len(vals) else 0.0
        sigma = float(vals.std(ddof=0)) if len(vals) else 0.0
        denom = sigma + mu_w
        burst[t] = (sigma - mu_w) / denom if denom != 0 else 0.0
    return pd.Series(burst, name="burstiness")


def hawkes_lambda_series(counts: pd.Series, alpha: float = 0.5, beta: float = 1.0) -> pd.Series:
    mu = float(counts.mean()) if len(counts) else 0.0
    hawkes: dict[int, float] = {}
    idx = counts.index.tolist()
    for t in idx:
        s = 0.0
        for k in idx:
            if k >= t:
                break
            s += float(np.exp(-beta * (t - k)) * counts.loc[k])
        hawkes[t] = mu + alpha * s
    return pd.Series(hawkes, name="hawkes_lambda")


def time_since_last_illicit_series(
    df: pd.DataFrame,
    counts_index: pd.Index,
    time_col: str = "time_step",
    class_col: str = "class",
) -> pd.Series:
    illicit_steps = set(df.loc[df[class_col].astype(str) == "1", time_col].unique())
    tsil: dict[int, float] = {}
    last = None
    for t in counts_index:
        if t in illicit_steps:
            last = t
        tsil[t] = float("nan") if last is None else float(t - last)
    s = pd.Series(tsil, name="time_since_last_illicit")
    s = s.fillna(49)
    return s


def compute_temporal_features(
    df: pd.DataFrame,
    time_col: str = "time_step",
    class_col: str = "class",
    window: int = 3,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    ## Per-step temporal stats: burstiness, Hawkes λ, time_since_last_illicit (§7.3.4)

    Parameters
    ----------
    df : DataFrame
        Must contain `time_col` and `class_col`.
    window : int
        Rolling window for burstiness.
    alpha, beta : float
        Fixed Hawkes params.

    Returns
    ----------
    DataFrame
        Indexed by time_step, no NaNs.
    """
    counts = _counts_per_step(df, time_col=time_col)
    burst = burstiness_series(counts, window=window)
    hawkes = hawkes_lambda_series(counts, alpha=alpha, beta=beta)
    tsil = time_since_last_illicit_series(df, counts.index, time_col=time_col, class_col=class_col)
    out = pd.DataFrame({"count": counts, "burstiness": burst, "hawkes_lambda": hawkes, "time_since_last_illicit": tsil})
    return out


def add_temporal_features(
    df: pd.DataFrame,
    time_col: str = "time_step",
    class_col: str = "class",
    window: int = 3,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    ## Broadcast per-step temporal stats to each transaction row

    Returns
    ----------
    DataFrame
        Copy of `df` with 3 new columns, NaNs filled.
    """
    stats = compute_temporal_features(df, time_col=time_col, class_col=class_col, window=window, alpha=alpha, beta=beta)
    out = df.copy()
    out["burstiness"] = out[time_col].map(stats["burstiness"]).fillna(0)
    out["hawkes_lambda"] = out[time_col].map(stats["hawkes_lambda"]).fillna(float(stats["hawkes_lambda"].mean() if len(stats) else 0))
    out["time_since_last_illicit"] = out[time_col].map(stats["time_since_last_illicit"]).fillna(49)
    return out
