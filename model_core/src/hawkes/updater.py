import numpy as np

from .power_law import PowerLawKernel


class HawkesUpdater:
    def __init__(self, alpha_init: float = 0.5, ttl_days: int = 14):
        self.alpha = alpha_init
        self.kernel = PowerLawKernel()
        self.ttl_days = ttl_days
        self.t = 0

    def update(self, event_time: float, history: list[float]) -> float:
        self.t += 1
        eta_t = 1.0 / np.sqrt(self.t)
        lam = self.kernel.compute_lambda(history, self.alpha, event_time)
        valid = [et for et in history if event_time - self.ttl_days * 86400 < et < event_time]
        if valid:
            grad = -sum((event_time - et) ** (-self.alpha) * np.log(event_time - et) for et in valid)
            self.alpha = max(self.alpha - eta_t * grad, 0.01)
        return lam


def demo() -> None:
    updater = HawkesUpdater()
    history = [0.0, 100.0, 200.0]
    lam = updater.update(300.0, history)
    assert lam > 0
    assert updater.alpha > 0
    print("updater demo passed")



