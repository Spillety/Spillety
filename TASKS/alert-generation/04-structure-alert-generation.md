# 7.1 Alert Generation Engine — Structure

```
alert-generation/
├── threshold/
│   ├── adaptive_threshold.py     # Per-entity baseline + 2σ computation
│   ├── baseline_service.py       # Rolling mean/stddev maintenance
│   └── priority_classifier.py    # P0/P1/P2 assignment logic
├── correlation/
│   ├── merge_engine.py           # 30min window correlation/merge
│   ├── causal_path_matcher.py    # Overlap ≥ 50% for merge eligibility
│   └── dedup_filter.py           # (entity_id, alert_type, causal_path) dedup
├── escalation/
│   ├── p0_router.py              # P0 → p0_sre_queue
│   ├── timer_service.py          # 15min escalation timer
│   └── oncall_manager.py         # Manager escalation logic
└── tests/
    ├── __init__.py
    ├── test_adaptive_threshold.py
    ├── test_merge_engine.py
    └── test_escalation.py
```

## Key Components
- `adaptive_threshold.py`: Rolling baseline + σ per entity; triggers on >2σ. English comments only.
- `merge_engine.py`: Groups alerts by `(entity_id, alert_type)` within 30min window
- `priority_classifier.py`: Maps σ levels and scam patterns to P0/P1/P2
- `p0_router.py`: Sends P0 to Kafka `p0_sre_queue`; starts escalation timer

## Pushback Hypotheses
1. **H1**: 30min merge window may delay P0 alerts. *Mitigation*: P0 alerts bypass merge; only P1/P2 are correlated. `# ponytail:` merge window vs. P0 latency
2. **H2**: Per-entity baseline computation is expensive at 100k+ entities. *Mitigation*: Pre-compute baselines nightly; delta update on new data. `# ponytail:` real-time vs. nightly baseline

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
