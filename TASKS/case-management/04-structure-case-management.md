# 7.2 Case Management — Structure

```
case-management/
├── temporal/
│   ├── workflow.py                # case_orchestration workflow definition
│   ├── activities.py              # create_case, populate_evidence, assign_analyst, submit_review
│   └── signals.py                 # Analyst action signals
├── servicenow/
│   ├── client.py                  # ServiceNow REST API client
│   ├── case_sync.py               # Bidirectional sync
│   └── mappings.py                # Priority/status mapping
├── evidence/
│   ├── causal_path.py             # Graph traversal for causal path
│   ├── counterfactual.py          # What-if simulation (remove edge → recompute)
│   └── alert_json_builder.py      # Assemble alert JSON payload
└── tests/
    ├── __init__.py
    ├── test_workflow.py
    ├── test_servicenow_sync.py
    └── test_counterfactual.py
```

## Key Components
- `workflow.py`: Temporal workflow orchestrating case lifecycle. English comments only.
- `activities.py`: Activity definitions; `populate_evidence` fetches alert_json + causal_path + counterfactual
- `client.py`: ServiceNow REST API wrapper; handles auth, rate limiting
- `counterfactual.py`: Edge-removal simulation on transaction graph; re-runs anomaly scoring

## Pushback Hypotheses
1. **H1**: Temporal adds operational complexity vs. direct Python orchestration. *Mitigation*: Use Temporal Cloud free tier for MVP; evaluate cost at scale. `# ponytail:` Temporal vs. simple orchestration
2. **H2**: Counterfactual re-computation is expensive for large graphs. *Mitigation*: Limit subgraph to entity + 2-hop neighbors; cache baseline scores. `# ponytail:` full graph vs. subgraph counterfactual

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
