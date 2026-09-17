import time

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return float(ece)


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if k <= 0 or len(y_true) == 0:
        return 0.0, 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp_k = int(y_sorted[:k].sum())
    prec = tp_k / k
    rec = tp_k / max(1, int(y_true.sum()))
    return float(prec), float(rec)


def _latency_stats(samples_ms: np.ndarray | list[float] | None) -> dict | None:
    if samples_ms is None:
        return None
    arr = np.asarray(samples_ms, dtype=float)
    if arr.size == 0:
        return None
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "n": int(arr.size),
    }


def _drift_ks(train_vals: np.ndarray | None, test_vals: np.ndarray | None) -> dict | None:
    if train_vals is None or test_vals is None:
        return None
    train_vals = np.asarray(train_vals).ravel()
    test_vals = np.asarray(test_vals).ravel()
    if train_vals.size == 0 or test_vals.size == 0:
        return None
    D, p = ks_2samp(train_vals, test_vals)
    return {"D": float(D), "p": float(p)}


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    K_list: list[int] | tuple[int, ...] = (10, 50, 100),
    latency_samples_ms: list[float] | np.ndarray | None = None,
    cost_per_alert_h: float = 0.25,
    fte_rate_per_h: float = 50.0,
    drift_train: np.ndarray | None = None,
    drift_test: np.ndarray | None = None,
    n_alerts_for_cost: int | None = None,
    audit_verified: int | None = None,
    audit_total: int | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    base_rate = float(y_true.mean()) if len(y_true) else 0.0

    # ranking
    pr_auc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        roc_auc = 0.0

    # calibration
    brier = float(brier_score_loss(y_true, y_score))
    ece = expected_calibration_error(y_true, y_score, n_bins=10)

    # retrieval @K
    pr_at_k = {}
    for k in K_list:
        prec, rec = precision_recall_at_k(y_true, y_score, int(k))
        pr_at_k[int(k)] = {"precision@K": prec, "recall@K": rec}

    # latency
    latency = _latency_stats(latency_samples_ms)

    # drift
    drift = _drift_ks(drift_train, drift_test)

    # cost
    if n_alerts_for_cost is None:
        n_alerts_for_cost = int((y_pred == 1).sum())
    cost_per_alert = float(cost_per_alert_h * fte_rate_per_h)
    total_cost = float(n_alerts_for_cost * cost_per_alert)

    # audit (placeholder if not provided: assume all alerts verified)
    if audit_total is None:
        audit_total = n_alerts_for_cost
    if audit_verified is None:
        audit_verified = audit_total
    audit = {
        "verified": int(audit_verified),
        "total": int(audit_total),
        "share": float(audit_verified / max(1, audit_total)),
    }

    # power analysis canonical p=0.05 e=0.01
    z = 1.96
    p_example, e_example = 0.05, 0.01
    n_power = (z**2 * p_example * (1 - p_example)) / (e_example**2)

    return {
        "base_rate": base_rate,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece,
        "precision_recall_at_k": pr_at_k,
        "latency": latency,
        "drift_ks": drift,
        "cost": {
            "cost_per_alert": cost_per_alert,
            "cost_per_alert_h": float(cost_per_alert_h),
            "fte_rate_per_h": float(fte_rate_per_h),
            "n_alerts": int(n_alerts_for_cost),
            "total_cost": total_cost,
        },
        "audit": audit,
        "power_n_example": float(n_power),
    }


def benchmark_latency(predict_fn, X_sample: np.ndarray, n_queries: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_sample), size=min(n_queries, len(X_sample)), replace=len(X_sample) < n_queries)
    X_q = X_sample[idx]
    times_ms: list[float] = []
    for i in range(len(X_q)):
        t0 = time.perf_counter()
        predict_fn(X_q[i : i + 1])
        times_ms.append((time.perf_counter() - t0) * 1000)
    return _latency_stats(np.array(times_ms))  # type: ignore[return-value]


if __name__ == "__main__":
    # synthetic demo: imbalanced 5% positives, well-separated scores
    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.random(n) < 0.05).astype(int)
    y_score = rng.random(n) * 0.4 + y_true * 0.5  # positives ~0.5-0.9
    y_score = np.clip(y_score, 0, 1)

    m = compute_all_metrics(y_true, y_score, K_list=[10, 50, 100], latency_samples_ms=rng.random(200) * 2 + 1, drift_train=rng.normal(0, 1, 500), drift_test=rng.normal(0.2, 1, 500))
    assert 0 <= m["pr_auc"] <= 1
    assert 0 <= m["brier"] <= 1
    assert 0 <= m["ece"] <= 1
    assert m["audit"]["share"] == 1.0
    for k in [10, 50, 100]:
        assert 0 <= m["precision_recall_at_k"][k]["precision@K"] <= 1
        assert 0 <= m["precision_recall_at_k"][k]["recall@K"] <= 1
    assert m["latency"]["p50_ms"] > 0
    assert m["drift_ks"]["D"] >= 0

    # tamper-like: flip 1 label changes metrics slightly but not crash
    y_true2 = y_true.copy()
    y_true2[0] = 1 - y_true2[0]
    m2 = compute_all_metrics(y_true2, y_score)
    assert m["pr_auc"] != m2["pr_auc"] or m["brier"] != m2["brier"]

    # proof 7 for N=100 via evidence layer (cross-check)
    from spillety.evidence.worm import build_merkle, merkle_proof, verify_proof
    import hashlib
    import math

    leaves = [hashlib.sha256(f"leaf-{i}".encode()).hexdigest() for i in range(100)]
    levels, root = build_merkle(leaves)
    proof = merkle_proof(levels, 0)
    assert len(proof) == 7 == math.ceil(math.log2(100))
    assert verify_proof(leaves[0], proof, root)
    # tamper 1 char
    bad_leaf = leaves[0][:-1] + ("0" if leaves[0][-1] != "0" else "1")
    assert not verify_proof(bad_leaf, proof, root)

    print(f"ok: PR-AUC={m['pr_auc']:.3f} Brier={m['brier']:.3f} ECE={m['ece']:.3f} proof N=100={len(proof)} p50={m['latency']['p50_ms']:.2f}ms")
