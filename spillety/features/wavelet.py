import numpy as np


def haar_level2(x: np.ndarray) -> np.ndarray:
    """
    ## Level-2 Haar wavelet decomposition along axis=1

    Decomposes each row into: [approx_L2, detail_L2, detail_L1_high, detail_L1_low]
    Preserves temporal structure of feature sequences.

    Parameters
    ----------
    x : np.ndarray
        Input array (N, D).

    Returns
    -------
    np.ndarray
        Wavelet coefficients (N, D), same shape as input.
    """
    _, d = x.shape
    out = np.zeros_like(x, dtype=np.float64)

    # Level 1: pair up adjacent elements
    d_even = d - (d % 2)
    if d_even == 0:
        return out
    low1 = (x[:, 0:d_even:2] + x[:, 1:d_even:2]) / 2.0
    high1 = (x[:, 0:d_even:2] - x[:, 1:d_even:2]) / 2.0

    # Level 2: split lowpass again
    d1_even = low1.shape[1] - (low1.shape[1] % 2)
    if d1_even == 0:
        out[:, :d_even] = np.concatenate([low1, high1], axis=1)
        return out

    low2 = (low1[:, 0:d1_even:2] + low1[:, 1:d1_even:2]) / 2.0
    high2 = (low1[:, 0:d1_even:2] - low1[:, 1:d1_even:2]) / 2.0

    # Pack coefficients: [approx_L2, detail_L2, detail_L1_high, detail_L1_low]
    pos = 0
    n_l2 = low2.shape[1]
    out[:, pos : pos + n_l2] = low2
    pos += n_l2
    out[:, pos : pos + n_l2] = high2
    pos += n_l2
    n_h1 = high1.shape[1]
    remaining = d - pos
    n_h1_fit = min(n_h1, remaining)
    out[:, pos : pos + n_h1_fit] = high1[:, :n_h1_fit]
    return out


def wavelet_features(x: np.ndarray) -> np.ndarray:
    """
    ## Append level-2 Haar wavelet coefficients to original features

    Parameters
    ----------
    x : np.ndarray
        Feature matrix (N, D).

    Returns
    -------
    np.ndarray
        Extended features (N, 2*D): [original, wavelet_coeffs].
    """
    wc = haar_level2(x)
    return np.hstack([x, wc])
