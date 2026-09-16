import numpy as np
import pandas as pd


def _counts_per_step(df: pd.DataFrame, time_col: str = "time_step") -> pd.Series:
    counts = df.groupby(time_col).size()
    # Elliptic is 1..49 — reindex to 1..49 for continuity if data looks elliptic, else use observed range
    # ponytail: MLE for Hawkes (scipy.optimize on train 1..30) → fit alpha/beta/mu instead of fixed 0.5/1.0
    if counts.index.min() >= 1 and counts.index.max() <= 49:
        counts = counts.reindex(range(1, 50), fill_value=0)
    return counts


def burstiness_series(counts: pd.Series, window: int = 3) -> pd.Series:
    """
    Goh & Barabási burstiness B=(sigma-mu)/(sigma+mu) in [-1,1] over rolling window
    """
    burst: dict[int, float] = {}
    idx = counts.index.tolist()
    lo = min(idx) if idx else 1
    for t in idx:
        # window [t-window+1 .. t] clipped to observed range
        vals = counts.loc[max(lo, t - window + 1) : t].values
        mu_w = float(vals.mean()) if len(vals) else 0.0
        # ddof=0 population std — otherwise w=1 gives NaN (single sample)
        sigma = float(vals.std(ddof=0)) if len(vals) else 0.0
        denom = sigma + mu_w
        burst[t] = (sigma - mu_w) / denom if denom != 0 else 0.0
    return pd.Series(burst, name="burstiness")


def hawkes_lambda_series(counts: pd.Series, alpha: float = 0.5, beta: float = 1.0) -> pd.Series:
    """
    Discrete Hawkes proxy lambda(t)=mu+alpha*sum_{k<t} exp(-beta*(t-k))*count[k]; mu=mean(count). No MLE.
    """
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
    """
    Steps since last t' with class==1; fill 49 if none yet (Elliptic illicit is every step → degenerate 0)
    """
    # class may be str "1" or int 1
    illicit_steps = set(df.loc[df[class_col].astype(str) == "1", time_col].unique())
    tsil: dict[int, float] = {}
    last = None
    for t in counts_index:
        if t in illicit_steps:
            last = t
        tsil[t] = float("nan") if last is None else float(t - last)
    s = pd.Series(tsil, name="time_since_last_illicit")
    # fill large value (49 = max horizon) when no illicit before t — keeps feature finite for scaler
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
    ## Per-step temporal stats (notebook 03) — burstiness, Hawkes λ, time_since_last_illicit

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
        Indexed by time_step, columns [count, burstiness, hawkes_lambda, time_since_last_illicit].
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
    ## Broadcast per-step temporal features to each transaction row

    Returns
    ----------
    DataFrame
        Copy of `df` with 3 new columns: burstiness, hawkes_lambda, time_since_last_illicit.
    """
    stats = compute_temporal_features(df, time_col=time_col, class_col=class_col, window=window, alpha=alpha, beta=beta)
    out = df.copy()
    out["burstiness"] = out[time_col].map(stats["burstiness"]).fillna(0)
    out["hawkes_lambda"] = out[time_col].map(stats["hawkes_lambda"]).fillna(float(stats["hawkes_lambda"].mean() if len(stats) else 0))
    # time_since_last_illicit is degenerate (always 0 on Elliptic) but kept for other chains where illicit is sparse
    out["time_since_last_illicit"] = out[time_col].map(stats["time_since_last_illicit"]).fillna(49)
    return out


if __name__ == "__main__":
    from pathlib import Path

    from spillety.data.loader import load_elliptic

    root = Path("data/elliptic_raw")
    if not root.exists():
        df = pd.DataFrame(
            {
                "txId": [1, 2, 3, 4, 5, 6],
                "time_step": [1, 1, 2, 2, 3, 3],
                "class": ["2", "1", "2", "2", "1", "2"],
            }
        )
        stats = compute_temporal_features(df)
        assert stats.shape[1] == 4, stats.shape
        assert not stats[["burstiness", "hawkes_lambda", "time_since_last_illicit"]].isna().any().any()
        # burstiness must be in [-1,1]
        assert stats["burstiness"].between(-1, 1).all(), stats["burstiness"].tolist()
        enriched = add_temporal_features(df)
        assert enriched.shape[0] == df.shape[0]
        assert not enriched[["burstiness", "hawkes_lambda", "time_since_last_illicit"]].isna().any().any()
        # time_since_last_illicit for t=1 (has illicit) should be 0
        assert float(enriched.loc[enriched["time_step"] == 1, "time_since_last_illicit"].iloc[0]) == 0
        print("temporal smoke test (synthetic) passed:", stats.shape)
    else:
        _, _, _, merged = load_elliptic(root)
        stats = compute_temporal_features(merged)
        assert stats.shape[0] == 49, stats.shape
        assert stats.shape[1] == 4
        assert not stats[["burstiness", "hawkes_lambda", "time_since_last_illicit"]].isna().any().any()
        assert stats["burstiness"].between(-1, 1).all()
        enriched = add_temporal_features(merged.head(1000))
        assert enriched.shape[0] == 1000
        assert not enriched[["burstiness", "hawkes_lambda", "time_since_last_illicit"]].isna().any().any()
        print("temporal demo passed:", stats.shape, "hawkes range", stats["hawkes_lambda"].min(), "..", stats["hawkes_lambda"].max())
