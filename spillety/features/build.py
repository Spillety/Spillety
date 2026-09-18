import numpy as np

_SENSITIVITY_DEFAULT = (
    "causal_filter_pass_rate",
    "e_value_max",
    "rosenbaum_gamma_min",
    "sensemakr_r2_max",
)
_MIN_DIST_NAMES = (
    "distance_to_nearest_OFAC",
    "distance_to_nearest_mixer",
    "distance_to_nearest_exchange",
)


def _as_2d(arr: np.ndarray, key: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"frames[{key!r}] must be 1D or 2D, got ndim={a.ndim}")
    return a


def build_feature_matrix(
    frames: dict[str, np.ndarray],
    names: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    ## Assemble GBDT input matrix from upstream arrays (§7.3)

    Parameters
    ----------
    frames : dict[str, np.ndarray]
        Upstream blocks, each with n rows: `distances` (n, K),
        `anchor_type_freqs` (n, T), `anchor_min_dist` (n, 3),
        `sensitivity` (n, 4), `graph` (n, G), `temporal` (n, Tm),
        `context` (n, C). Only `distances` is required.
    names : dict | None
        Optional column labels per block: `anchor_types`, `graph`,
        `temporal`, `context`, `sensitivity`.

    Returns
    ----------
    tuple[np.ndarray, list[str]]
        Feature matrix (n, d) and deterministic column names.
    """
    names = names or {}
    if "distances" not in frames:
        raise ValueError("frames must contain 'distances' (n, K)")
    d = _as_2d(frames["distances"], "distances")
    n, k = d.shape
    for key, arr in frames.items():
        if np.asarray(arr).shape[0] != n:
            raise ValueError(f"frames[{key!r}] has n={np.asarray(arr).shape[0]}, want {n}")

    cols: list[np.ndarray] = []
    feature_names: list[str] = []

    cols.append(d)
    feature_names += [f"d_{i + 1}" for i in range(k)]
    agg = np.column_stack(
        [
            d.mean(axis=1),
            d.min(axis=1),
            d.std(axis=1),
            np.quantile(d, 0.25, axis=1),
            np.quantile(d, 0.50, axis=1),
            np.quantile(d, 0.75, axis=1),
        ]
    )
    cols.append(agg)
    feature_names += [
        "dist_mean",
        "dist_min",
        "dist_std",
        "dist_q25",
        "dist_q50",
        "dist_q75",
    ]

    if "anchor_type_freqs" in frames:
        f = _as_2d(frames["anchor_type_freqs"], "anchor_type_freqs")
        cols.append(f)
        labels = names.get("anchor_types", [f"type_{i}" for i in range(f.shape[1])])
        feature_names += [f"anchor_type_{t}" for t in labels]
        # Diversity not observable from freqs alone when passed separately.
        div = frames.get("anchor_source_diversity", (f > 0).sum(axis=1))
        cols.append(np.asarray(div, dtype=float).reshape(n, 1))
        feature_names.append("anchor_source_diversity")

    if "anchor_min_dist" in frames:
        m = _as_2d(frames["anchor_min_dist"], "anchor_min_dist")
        if m.shape[1] != 3:
            raise ValueError(f"anchor_min_dist must have 3 columns, got {m.shape[1]}")
        cols.append(m)
        feature_names += list(_MIN_DIST_NAMES)

    if "sensitivity" in frames:
        s = _as_2d(frames["sensitivity"], "sensitivity")
        if s.shape[1] != 4:
            raise ValueError(f"sensitivity must have 4 columns, got {s.shape[1]}")
        cols.append(s)
        feature_names += list(names.get("sensitivity", _SENSITIVITY_DEFAULT))

    # Passthrough blocks keep upstream order; no retrieval/GraphSAGE here.
    # ponytail: no imputation ceiling — NaNs propagate, add median fill when GBDT needs it.
    for key in ("graph", "temporal", "context"):
        if key in frames:
            b = _as_2d(frames[key], key)
            cols.append(b)
            labels = names.get(key, [f"{key}_{i}" for i in range(b.shape[1])])
            feature_names += list(labels)

    return np.column_stack(cols), feature_names
