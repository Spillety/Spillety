import math

__all__ = ["AUTO_BLOCK_RATES", "AUTO_CLEAR_RATES", "Z_95", "audit_range", "power_n"]

Z_95 = 1.96
AUTO_CLEAR_RATES = (0.001, 0.01)
AUTO_BLOCK_RATES = (0.01, 0.05)


def power_n(p: float, e: float, z: float = Z_95) -> int:
    """
    ## Random-sample size for an unbiased precision estimate (§8.5.2)

    Parameters
    ----------
    p : float
        Expected precision (illicit share).
    e : float
        Half-width of the confidence interval.
    z : float
        Normal quantile, 1.96 for 95% CI.

    Returns
    ----------
    int
        Ceiling of z²·p(1−p)/e².
    """
    return math.ceil(z**2 * p * (1.0 - p) / e**2)


def audit_range(pool_size: int, kind: str = "clear") -> tuple:
    """
    ## Random-audit count bounds for auto-clear / auto-block pools (§12.10)

    Parameters
    ----------
    pool_size : int
        Pool size (auto-clear or auto-block).
    kind : str
        `clear` → 0.1–1%, `block` → 1–5% random audit.

    Returns
    ----------
    tuple
        (lo, hi) audit counts.
    """
    rates = {"clear": AUTO_CLEAR_RATES, "block": AUTO_BLOCK_RATES}[kind]
    return math.ceil(pool_size * rates[0]), math.ceil(pool_size * rates[1])
