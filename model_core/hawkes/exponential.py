import numpy as np

TTL_DAYS = 14


class ExponentialKernel:
    def __init__(self, ttl_days: int = TTL_DAYS):
        self.ttl = ttl_days

    def compute_lambda(self, event_times: list[float], beta: float, t: float) -> float:
        cutoff = t - self.ttl * 86400
        return float(sum(np.exp(-beta * (t - et)) for et in event_times if cutoff < et < t))
