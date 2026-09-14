import numpy as np
from datetime import datetime, timezone

from .exponential import ExponentialKernel
from .power_law import PowerLawKernel
from .seasonal_mu import SeasonalMu

BURST_TRIGGER = "burst_detected"
NO_BURST_TRIGGER = "no_burst"


class HawkesUpdater:
    def __init__(
        self,
        alpha_init: float = 0.5,
        ttl_days: int = 14,
        kernel: str = "power_law",
        beta: float = 0.01,
        base_mu: float = 0.1,
        mu_model: SeasonalMu | None = None,
        threshold: float = 2.0,
    ):
        self.alpha = alpha_init
        self.beta = beta
        self.base_mu = base_mu
        self.threshold = threshold
        self.kernel_name = kernel
        self.power_law = PowerLawKernel()
        self.exponential = ExponentialKernel(ttl_days=ttl_days)
        self.mu_model = mu_model or SeasonalMu()
        self.ttl_days = ttl_days
        self.t = 0

    def _kernel_sum(self, history: list[float], t: float) -> float:
        if self.kernel_name == "exponential":
            return self.exponential.compute_lambda(history, self.beta, t)
        return self.power_law.compute_lambda(history, self.alpha, t)

    def intensity(self, t: float, history: list[float], timestamp: datetime | None = None) -> float:
        # ponytail: SeasonalMu may be negative (raw Fourier sum); base_mu keeps background rate non-negative.
        if timestamp is None:
            mu = self.base_mu + max(self.mu_model.at_epoch(t), 0.0)
        else:
            mu = self.base_mu + max(self.mu_model.compute(timestamp), 0.0)
        return mu + self._kernel_sum(history, t)

    def update(self, event_time: float, history: list[float], timestamp: datetime | None = None) -> float:
        self.t += 1
        eta_t = 1.0 / np.sqrt(self.t)
        lam = self.intensity(event_time, history, timestamp)
        valid = [et for et in history if event_time - self.ttl_days * 86400 < et < event_time]
        if valid and self.kernel_name == "power_law":
            # Closed-form gradient of sum (dt^-alpha) w.r.t. alpha; keeps update O(n) over TTL window.
            grad = -sum((event_time - et) ** (-self.alpha) * np.log(event_time - et) for et in valid if event_time - et > 1.0)
            self.alpha = max(self.alpha - eta_t * grad, 0.01)
        return lam

    def intensity_payload(self, lambda_t: float, threshold: float | None = None) -> dict:
        thr = self.threshold if threshold is None else threshold
        trigger = BURST_TRIGGER if lambda_t > thr else NO_BURST_TRIGGER
        return {"lambda_t": float(lambda_t), "threshold": float(thr), "trigger": trigger}

    def update_and_trigger(
        self, event_time: float, history: list[float], timestamp: datetime | None = None
    ) -> tuple[float, dict, bool]:
        lam = self.update(event_time, history, timestamp)
        payload = self.intensity_payload(lam)
        return lam, payload, payload["trigger"] == BURST_TRIGGER
