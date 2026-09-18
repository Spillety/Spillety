import numpy as np
from scipy import stats


def power_n_min(effect: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    ## Minimum n to detect a correlation with given power (§6.3.4)

    Parameters
    ----------
    effect : float
        Target correlation magnitude.
    alpha, power : float
        Significance level and target power (1−β).

    Returns
    ----------
    int
        Fisher-z sample-size estimate, at least 4.
    """
    # ponytail: normal approximation; exact t-power upgrade if rare
    # subclasses ever need sub-5% sizing error (§6.3.4 follow-up).
    r = float(np.clip(abs(effect), 1e-6, 0.999999))
    zr = 0.5 * np.log((1 + r) / (1 - r))
    n = ((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / zr) ** 2 + 3
    return max(4, int(np.ceil(n)))


def power_gate(
    n: int, effect: float, *, alpha: float = 0.05, power: float = 0.8
) -> dict:
    """
    ## Power gate: testable stratum or «недостаточно данных» (§6.3.4)

    Parameters
    ----------
    n : int
        Available sample size.
    effect, alpha, power : float
        Same as in `power_n_min`.

    Returns
    ----------
    dict
        `n_min`, `enough` and `status` (`ok` / `insufficient data`).
    """
    n_min = power_n_min(effect, alpha=alpha, power=power)
    enough = int(n) >= n_min
    return {
        "n_min": n_min,
        "enough": bool(enough),
        "status": "ok" if enough else "insufficient data",
    }


def select_dag(candidates: dict[str, dict]) -> dict:
    """
    ## Pick a DAG by fewest rejections, then parsimony (§6.3.3)

    Parameters
    ----------
    candidates : dict[str, dict]
        Name → {`rejections`, `num_edges`, optional `n`, `effect`
        for the power gate}.

    Returns
    ----------
    dict
        `best` (None when nothing is testable), ranked `table`,
        `status`.
    """
    rows = []
    for name, spec in candidates.items():
        gate = None
        if "n" in spec and "effect" in spec:
            gate = power_gate(int(spec["n"]), float(spec["effect"]))
        rows.append(
            {
                "name": name,
                "rejections": int(spec.get("rejections", 0)),
                "num_edges": int(spec.get("num_edges", 0)),
                "eligible": gate["enough"] if gate else True,
                "status": gate["status"] if gate else "ok",
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    # Fewest rejections first, fewest edges on ties (parsimony).
    table = sorted(rows, key=lambda r: (not r["eligible"], r["rejections"], r["num_edges"]))
    if not eligible:
        return {"best": None, "table": table, "status": "insufficient data"}
    best = min(eligible, key=lambda r: (r["rejections"], r["num_edges"]))
    return {"best": best["name"], "table": table, "status": "ok"}
