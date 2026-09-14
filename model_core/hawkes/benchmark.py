import time

from .updater import HawkesUpdater


def measure_update_latency(n_events: int = 1000, n_repeats: int = 5) -> dict:
    """Harness stub for wave E: measures mean λ(t) update latency.

    No <1ms assertion here by design; the DoD benchmark runs in wave E.
    """
    history = [float(i * 10) for i in range(n_events)]
    updater = HawkesUpdater()
    t = float(n_events * 10 + 5)
    latencies = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        updater.intensity(t, history)
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    return {"mean_ms": sum(latencies) / len(latencies), "p50_ms": latencies[len(latencies) // 2], "n_events": n_events}
