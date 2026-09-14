import numpy as np

TTL_DAYS = 14


class PowerLawKernel:
    def __init__(self):
        self.ttl = TTL_DAYS

    def compute_lambda(self, event_times: list[float], alpha: float, t: float) -> float:
        cutoff = t - self.ttl * 86400
        contrib = sum((t - et) ** (-alpha) for et in event_times if et > cutoff and et < t)
        return contrib
