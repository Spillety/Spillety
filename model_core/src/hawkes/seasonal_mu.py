import numpy as np
from datetime import datetime


class SeasonalMu:
    def __init__(self, hourly_components: int = 4, daily_components: int = 2):
        self.hourly_components = hourly_components
        self.daily_components = daily_components
        self._cache: dict[int, float] = {}

    def _compute_fourier(self, t: float) -> float:
        mu = 0.0
        for k in range(1, self.hourly_components + 1):
            mu += np.cos(2 * np.pi * k * t / 3600)
        for k in range(1, self.daily_components + 1):
            mu += np.cos(2 * np.pi * k * t / 86400)
        return mu

    def compute(self, timestamp: datetime) -> float:
        hour_key = timestamp.hour
        if hour_key not in self._cache:
            self._cache[hour_key] = self._compute_fourier(float(hour_key * 3600))
        return self._cache[hour_key]


def demo() -> None:
    mu = SeasonalMu()
    v1 = mu.compute(datetime(2024, 1, 1, 12, 0))
    v2 = mu.compute(datetime(2024, 1, 2, 12, 0))
    assert abs(v1 - v2) < 1e-6
    v3 = mu.compute(datetime(2024, 1, 1, 0, 0))
    assert v3 != v1
    print("seasonal_mu demo passed")


if __name__ == "__main__":
    demo()
