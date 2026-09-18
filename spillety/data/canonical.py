__all__ = ["BTC_CONFIRMATIONS", "ETH_CONFIRMATIONS", "is_finalized"]

BTC_CONFIRMATIONS = (0, 1, 3, 6)
ETH_CONFIRMATIONS = (12, 32, 64)

_DEFAULT_REQUIRED = {"btc": 6, "bitcoin": 6, "eth": 32, "ethereum": 32}
_ALLOWED = {"btc": BTC_CONFIRMATIONS, "bitcoin": BTC_CONFIRMATIONS, "eth": ETH_CONFIRMATIONS, "ethereum": ETH_CONFIRMATIONS}


def is_finalized(chain: str, confirmations: int, required: int | None = None) -> bool:
    """
    ## Finality check against the confirmation buffer (§2.2.3)

    Parameters
    ----------
    chain : str
        `btc`/`bitcoin` or `eth`/`ethereum`.
    confirmations : int
        Observed confirmation count.
    required : int | None
        Required buffer; must be one of 0/1/3/6 (BTC) or 12/32/64 (ETH).
        Defaults to 6 (BTC) or 32 (ETH).

    Returns
    ----------
    bool
        True iff confirmations reach the required buffer.
    """
    key = chain.lower()
    if key not in _ALLOWED:
        raise ValueError(f"unknown chain: {chain!r}")
    if confirmations < 0:
        raise ValueError(f"confirmations must be non-negative: {confirmations}")
    need = _DEFAULT_REQUIRED[key] if required is None else required
    if need not in _ALLOWED[key]:
        raise ValueError(f"required must be one of {_ALLOWED[key]}: {required}")
    return bool(confirmations >= need)
