from dataclasses import dataclass

import numpy as np


@dataclass
class LeadTimeResult:
    lead_steps: list[int]
    median: float
    p90: float
    by_era: dict[str, float]
    recall_at_k_new: float
    censored_count: int


def evaluate_lead_time(
    alert_steps: dict[int, int],
    sanction_steps: dict[int, int],
    k: int = 100,
    era_split: int = 43,
) -> LeadTimeResult:
    """
    ## Parameters
    ----------
    alert_steps : dict[int, int]
        Mapping anchor_id -> first step where alert appeared in top-K.
        Must be built ONLY from alerts passing DAG filter AND cost-based τ*.
    sanction_steps : dict[int, int]
        Mapping anchor_id -> sanction step.
    k : int
        Top-K cutoff for recall_at_k_new.
    era_split : int
        Step threshold for pre/post stratification (default 43 per R2).

    ## Returns
    -------
    LeadTimeResult
        Lead time statistics with era stratification and censored count.
    """
    common_anchors = set(alert_steps.keys()) & set(sanction_steps.keys())
    lead_steps = []
    by_era = {f"pre_{era_split}": [], f"post_{era_split}": []}
    censored_count = 0
    post_train_with_alert = 0
    post_train_total = 0

    for aid in common_anchors:
        a_step = alert_steps[aid]
        s_step = sanction_steps[aid]
        if a_step > s_step:
            raise ValueError(f"anchor {aid}: alert_step {a_step} > sanction_step {s_step}")
        if a_step <= 0 or s_step <= 0:
            raise ValueError(f"anchor {aid}: steps must be positive")
        lead = s_step - a_step
        lead_steps.append(lead)
        era_key = f"pre_{era_split}" if s_step < era_split else f"post_{era_split}"
        by_era[era_key].append(lead)
        if s_step >= era_split:
            post_train_total += 1
            if a_step <= k:
                post_train_with_alert += 1

    all_sanction_anchors = set(sanction_steps.keys())
    for aid in all_sanction_anchors:
        if aid not in alert_steps:
            censored_count += 1
            s_step = sanction_steps[aid]
            if s_step >= era_split:
                post_train_total += 1

    if not lead_steps:
        median = float("nan")
        p90 = float("nan")
        era_medians = {f"pre_{era_split}": float("nan"), f"post_{era_split}": float("nan")}
        recall = 0.0
    else:
        lead_arr = np.array(lead_steps, dtype=float)
        median = float(np.median(lead_arr))
        p90 = float(np.percentile(lead_arr, 90))
        era_medians = {}
        for era, vals in by_era.items():
            era_medians[era] = float(np.median(vals)) if vals else float("nan")
        recall = post_train_with_alert / post_train_total if post_train_total > 0 else 0.0

    return LeadTimeResult(
        lead_steps=lead_steps,
        median=median,
        p90=p90,
        by_era=era_medians,
        recall_at_k_new=recall,
        censored_count=censored_count,
    )