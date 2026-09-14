import numpy as np

TTL_DAYS = 14


class PowerLawKernel:
    def __init__(self):
        self.ttl = TTL_DAYS

    def compute_lambda(self, event_times: list[float], alpha: float, t: float) -> float:
        cutoff = t - self.ttl * 86400
        contrib = sum((t - et) ** (-alpha) for et in event_times if et > cutoff and et < t)
        return contrib


def demo() -> None:
    kernel = PowerLawKernel()
    times = [0.0, 1000.0, 2000.0]
    lam = kernel.compute_lambda(times, alpha=0.5, t=3000.0)
    assert lam > 0
    lam2 = kernel.compute_lambda(times, alpha=1.0, t=3000.0)
    assert lam2 < lam
    print("power_law demo passed")


if __name__ == "__main__":
    demo()
