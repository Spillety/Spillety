"""Priority scorer: orchestrator JSON risk_score -> priority/action (TODO G7.2)."""
from __future__ import annotations

from typing import Any

DEFAULT_THRESHOLDS = {"p0": 0.95, "p1": 0.7, "p2": 0.5}


def score_alert(alert: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Attach priority/action to an orchestrator alert dict without mutating it."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    try:
        score = float(alert["risk_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"alert lacks numeric risk_score: {exc}") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"risk_score out of range [0, 1]: {score}")
    if score > th["p0"]:
        priority, action = "P0", "BLOCK"
    elif score >= th["p1"]:
        priority, action = "P1", "REVIEW"
    elif score >= th["p2"]:
        priority, action = "P2", "MONITOR"
    else:
        priority, action = "IGNORE", "ALLOW"
    return {**alert, "priority": priority, "action": action}
