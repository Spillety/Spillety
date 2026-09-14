"""Dedup/merge: same entity+pattern collapse into one alert, with rate limiting."""
from __future__ import annotations

import time
from typing import Any


class DedupMerger:
    """Groups scored alerts by (entity, pattern) inside a time window.

    Merge keeps the highest score and counts collapsed alerts; per-entity
    hourly rate limit drops excess alerts instead of queuing them.
    """

    def __init__(
        self,
        entity_side: str = "to",
        pattern_field: str = "nearest_scam_cluster",
        window_hours: float = 24.0,
        rate_limit_per_entity_per_hour: int = 10,
    ) -> None:
        self._entity_side = entity_side
        self._pattern_field = pattern_field
        self._window_sec = window_hours * 3600.0
        self._rate_limit = rate_limit_per_entity_per_hour
        self._groups: dict[tuple[str, str], dict[str, Any]] = {}
        self._hits: dict[str, list[float]] = {}

    def _key(self, alert: dict[str, Any]) -> tuple[str, str]:
        entity = str(alert["transaction"][self._entity_side])
        pattern = str(alert["explanation"]["hyperbolic_distance"][self._pattern_field])
        return (entity, pattern)

    def _expired(self, group: dict[str, Any], now: float) -> bool:
        return now - group["first_seen"] > self._window_sec

    def _rate_limited(self, entity: str, now: float) -> bool:
        recent = [t for t in self._hits.get(entity, []) if now - t < 3600.0]
        self._hits[entity] = recent
        return len(recent) >= self._rate_limit

    def add(self, alert: dict[str, Any], now: float | None = None) -> dict[str, Any]:
        """Add a scored alert; returns dict with outcome: merged/dropped/emitted."""
        now = time.time() if now is None else now
        key = self._key(alert)
        entity = key[0]
        if self._rate_limited(entity, now):
            return {"outcome": "dropped", "reason": "rate_limited", "alert": alert}
        self._hits.setdefault(entity, []).append(now)
        group = self._groups.get(key)
        if group is None or self._expired(group, now):
            self._groups[key] = {
                "first_seen": now,
                "alert_ids": [alert["alert_id"]],
                "best": alert,
                "merged_count": 1,
            }
            return {"outcome": "emitted", "alert": alert, "merged_count": 1}
        group["alert_ids"].append(alert["alert_id"])
        group["merged_count"] += 1
        # Keep the highest-score representative for the merged alert.
        if float(alert["risk_score"]) > float(group["best"]["risk_score"]):
            group["best"] = alert
        merged = {**group["best"], "merged_count": group["merged_count"], "merged_ids": list(group["alert_ids"])}
        return {"outcome": "merged", "alert": merged, "merged_count": group["merged_count"]}
