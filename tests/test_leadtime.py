import numpy as np
import pytest

from spillety.temporal.leadtime import LeadTimeResult, evaluate_lead_time


def test_lead_time_median_p90_manual_fixture():
    alert_steps = {1: 10, 2: 15, 3: 20, 4: 25}
    sanction_steps = {1: 13, 2: 20, 3: 28, 4: 35}
    res = evaluate_lead_time(alert_steps, sanction_steps)
    assert isinstance(res, LeadTimeResult)
    assert res.lead_steps == [3, 5, 8, 10]
    assert res.median == pytest.approx(6.5)
    assert res.p90 == pytest.approx(9.4)


def test_lead_time_stratification_pre_post_43():
    alert_steps = {1: 10, 2: 30, 3: 40, 4: 45}
    sanction_steps = {1: 15, 2: 38, 3: 50, 4: 55}
    res = evaluate_lead_time(alert_steps, sanction_steps, era_split=43)
    assert "pre_43" in res.by_era
    assert "post_43" in res.by_era
    assert res.by_era["pre_43"] == pytest.approx(6.5)
    assert res.by_era["post_43"] == pytest.approx(10.0)


def test_lead_time_stratification_dynamic_era_split():
    alert_steps = {1: 10, 2: 30}
    sanction_steps = {1: 15, 2: 38}
    res = evaluate_lead_time(alert_steps, sanction_steps, era_split=50)
    assert "pre_50" in res.by_era
    assert "post_50" in res.by_era


def test_lead_time_censored_excluded_from_median():
    alert_steps = {1: 10, 2: 20}
    sanction_steps = {1: 15, 2: 25, 3: 30, 4: 35}
    res = evaluate_lead_time(alert_steps, sanction_steps)
    assert res.censored_count == 2
    assert res.median == pytest.approx(5.0)
    assert len(res.lead_steps) == 2


def test_lead_time_recall_at_k_new():
    alert_steps = {1: 10, 2: 30, 3: 50}
    sanction_steps = {1: 15, 2: 38, 3: 55, 4: 60}
    res = evaluate_lead_time(alert_steps, sanction_steps, k=100, era_split=43)
    assert res.recall_at_k_new == pytest.approx(0.5)


def test_lead_time_determinism():
    alert_steps = {1: 10, 2: 20, 3: 30}
    sanction_steps = {1: 15, 2: 25, 3: 35}
    r1 = evaluate_lead_time(alert_steps, sanction_steps)
    r2 = evaluate_lead_time(alert_steps, sanction_steps)
    assert r1.median == r2.median
    assert r1.p90 == r2.p90
    for era in ("pre_43", "post_43"):
        v1, v2 = r1.by_era[era], r2.by_era[era]
        if np.isnan(v1):
            assert np.isnan(v2)
        else:
            assert v1 == v2
    assert r1.recall_at_k_new == r2.recall_at_k_new
    assert r1.censored_count == r2.censored_count


def test_lead_time_alert_before_sanction_sanity():
    alert_steps = {1: 10}
    sanction_steps = {1: 5}
    with pytest.raises(ValueError):
        evaluate_lead_time(alert_steps, sanction_steps)


def test_lead_time_positive_steps_sanity():
    alert_steps = {1: -1}
    sanction_steps = {1: 10}
    with pytest.raises(ValueError):
        evaluate_lead_time(alert_steps, sanction_steps)


def test_lead_time_empty_no_common_anchors():
    alert_steps = {1: 10}
    sanction_steps = {2: 20}
    res = evaluate_lead_time(alert_steps, sanction_steps)
    assert np.isnan(res.median)
    assert np.isnan(res.p90)
    assert np.isnan(res.by_era["pre_43"])
    assert np.isnan(res.by_era["post_43"])
    assert res.recall_at_k_new == 0.0
    assert res.censored_count == 1


def test_lead_time_empty_with_custom_era_split():
    alert_steps = {1: 10}
    sanction_steps = {2: 20}
    res = evaluate_lead_time(alert_steps, sanction_steps, era_split=50)
    assert np.isnan(res.by_era["pre_50"])
    assert np.isnan(res.by_era["post_50"])