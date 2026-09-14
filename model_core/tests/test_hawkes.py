import pytest
import numpy as np
from datetime import datetime

from model_core.hawkes.exponential import ExponentialKernel
from model_core.hawkes.power_law import PowerLawKernel
from model_core.hawkes.seasonal_mu import SeasonalMu
from model_core.hawkes.updater import HawkesUpdater


def test_power_law_positive():
    kernel = PowerLawKernel()
    times = [0.0, 1000.0, 2000.0]
    lam = kernel.compute_lambda(times, alpha=0.5, t=3000.0)
    assert lam > 0


def test_power_law_alpha_effect():
    kernel = PowerLawKernel()
    times = [0.0, 1000.0, 2000.0]
    lam_05 = kernel.compute_lambda(times, alpha=0.5, t=3000.0)
    lam_10 = kernel.compute_lambda(times, alpha=1.0, t=3000.0)
    assert lam_10 < lam_05


def test_power_law_ttl_truncation():
    kernel = PowerLawKernel()
    lam = kernel.compute_lambda([1e9 - 100.0, -1e9], alpha=0.5, t=1e9)
    assert lam > 0
    assert kernel.compute_lambda([-1e9], alpha=0.5, t=1e9) == 0.0


def test_power_law_empty_history():
    kernel = PowerLawKernel()
    lam = kernel.compute_lambda([], alpha=0.5, t=100.0)
    assert lam == 0.0


def test_seasonal_mu_deterministic():
    mu = SeasonalMu()
    v1 = mu.compute(datetime(2024, 1, 1, 12, 0))
    v2 = mu.compute(datetime(2024, 1, 2, 12, 0))
    assert abs(v1 - v2) < 1e-6


def test_seasonal_mu_hourly_variation():
    mu = SeasonalMu()
    v1 = mu.compute(datetime(2024, 1, 1, 0, 0))
    v2 = mu.compute(datetime(2024, 1, 1, 12, 0))
    assert v1 != v2


def test_seasonal_mu_cache():
    mu = SeasonalMu()
    t1 = datetime(2024, 1, 1, 5, 0)
    t2 = datetime(2024, 1, 2, 5, 0)
    v1 = mu.compute(t1)
    v2 = mu.compute(t2)
    assert abs(v1 - v2) < 1e-6


def test_seasonal_mu_fourier_components():
    mu = SeasonalMu(hourly_components=4, daily_components=2)
    assert mu.hourly_components == 4
    assert mu.daily_components == 2


def test_updater_positive_lambda():
    updater = HawkesUpdater()
    history = [0.0, 100.0, 200.0]
    lam = updater.update(300.0, history)
    assert lam > 0


def test_updater_alpha_bounded():
    updater = HawkesUpdater(alpha_init=0.5)
    history = [0.0, 100.0, 200.0]
    updater.update(300.0, history)
    assert updater.alpha > 0


def test_updater_alpha_update():
    updater = HawkesUpdater(alpha_init=0.5)
    history = [0.0, 100.0, 200.0]
    old_alpha = updater.alpha
    updater.update(300.0, history)
    assert updater.alpha != old_alpha


def test_updater_adaptive_lr():
    updater = HawkesUpdater()
    history = [0.0]
    updater.update(10.0, history)
    eta1 = updater.t
    updater.update(20.0, history)
    assert updater.t == 2


class _ZeroMu:
    def at_epoch(self, t):
        return 0.0

    def compute(self, ts):
        return 0.0


def test_updater_kernel_selection():
    up_exp = HawkesUpdater(kernel="exponential", beta=0.0, base_mu=0.0, mu_model=_ZeroMu())
    assert up_exp.intensity(10.0, [0.0, 5.0]) == pytest.approx(2.0)
    up_pl = HawkesUpdater(kernel="power_law", alpha_init=1.0, base_mu=0.0, mu_model=_ZeroMu())
    assert up_pl.intensity(8.0, [0.0]) == pytest.approx(0.125)


def test_trigger_fires():
    updater = HawkesUpdater(threshold=2.0, base_mu=0.0, mu_model=_ZeroMu())
    _, payload, recompute = updater.update_and_trigger(8.0, [7.0, 7.0, 7.0])
    assert payload["trigger"] == "burst_detected"
    assert recompute is True
    assert set(payload) == {"lambda_t", "threshold", "trigger"}


def test_trigger_silent():
    updater = HawkesUpdater(threshold=100.0, base_mu=0.0, mu_model=_ZeroMu())
    _, payload, recompute = updater.update_and_trigger(8.0, [0.0])
    assert payload["trigger"] == "no_burst"
    assert recompute is False
