# 7.1 Alert Generation Engine — Design

## Context
Adaptive thresholding per entity using baseline + >2σ detection, alert fatigue mitigation via correlation/merge, P0/P1/P2 priority assignment, auto-escalation for P0.

### [human] Design Decisions
- **Adaptive Threshold**: Per-entity baseline = rolling mean (30d); trigger when current value > baseline + 2σ; σ computed from same 30d window
- **Alert Fatigue Mitigation**: Correlation engine merges alerts from same entity within 30min window; deduplicate on `(entity_id, alert_type, causal_path)`
- **P0/P1/P2 Priorities**: P0 = >5σ or known scam pattern; P1 = >3σ; P2 = >2σ; auto-escalate P0 → P0_SRE within 5min
- **Auto-Escalate P0**: P0 alerts routed to P0_SRE queue with SLA 5min; if unacknowledged after 15min, escalate to on-call manager

### [agent] resolved
- Baseline: `avg(amount) OVER (PARTITION BY entity_id ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW)`; σ = `stddev_samp` over same window
- Threshold: `current > baseline + 2 * sigma` → alert
- Correlation merge: Alerts grouped by `entity_id` + `alert_type` within 30min; merged into single alert with count and max severity
- Priority mapping: `>5σ OR scam_pattern_match → P0`; `>3σ → P1`; `>2σ → P2`
- Auto-escalate: P0 → `p0_sre_queue` Kafka topic; 15min timer via Temporal workflow

## Architecture
```
Anomaly Scores → Adaptive Threshold → Correlation/Merge → Priority Assignment → Alert Routing
  ├── Baseline + 2σ: rolling window per entity
  ├── Correlation: 30min merge window on (entity_id, alert_type)
  ├── Priority: P0 (>5σ/scam), P1 (>3σ), P2 (>2σ)
  └── Auto-escalate: P0 → p0_sre_queue → 15min timer → manager
```

**Topics**: `alerts_raw`, `alerts_merged`, `p0_sre_queue`

## Pushback Hypotheses
1. **H1**: Adaptive 2σ threshold produces false positives during legitimate volume spikes (e.g., payday). *Mitigation*: Add business-hour adjustment factor; exclude known high-volume days from baseline. `# ponytail:` static vs. adaptive threshold
2. **H2**: Correlation merge may hide distinct scam patterns within same entity. *Mitigation*: Only merge if causal paths overlap ≥ 50%; keep separate alerts for different patterns. `# ponytail:` aggressive vs. conservative merge

## Open Questions
- What is the optimal merge window for correlation (30min vs. 1h)?
- How to handle first-time entities with no baseline history?

## [human] Acceptance
- False positive rate < 5% on labeled test set
- P0 escalation SLA < 5min from detection
- Alert merge reduces volume by ≥ 40% without losing critical alerts
- Per-entity baseline adapts within 5 days of new pattern
