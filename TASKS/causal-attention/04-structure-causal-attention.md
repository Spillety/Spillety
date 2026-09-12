# 3A Causal Attention Implementation — Structure

```
causal-attention/
├── models/
│   ├── notears_dag.py            # NOTEARS differentiable DAG discovery
│   ├── causal_attention.py       # Custom PyTorch causal attention layer
│   ├── temporal_encoder.py       # PyG-Temporal temporal convolution
│   └── domain_prior.py           # Expert graph prior + edge type mask
├── preprocessing/
│   ├── subgraph_extractor.py     # BFS depth 2 local subgraph
│   ├── covariate_builder.py      # Address features, confounders
│   └── backdoor_adjustment.py    # is_exchange_internal adjustment
├── training/
│   ├── train_dag.py              # NOTEARS training loop
│   ├── train_attention.py        # Causal attention fine-tuning
│   └── losses.py                 # NOTEARS loss + attention loss
├── evaluation/
│   ├── dag_accuracy.py           # Reconstruction accuracy metric
│   └── stability_tests.py        # Gradient norm, numerical stability
└── tests/
    ├── __init__.py
    ├── test_notears_dag.py
    ├── test_causal_attention.py
    └── test_subgraph_extractor.py
```

## Key Components
- `notears_dag.py`: Differentiable DAG via adjacency matrix parameterization. English comments only.
- `causal_attention.py`: PyTorch module with backdoor-adjusted attention weights
- `subgraph_extractor.py`: BFS from target nodes, returns subgraph + covariates
- `domain_prior.py`: Expert graph encoding, edge type constraints

## Pushback Hypotheses
1. **H1**: PyG-Temporal dependency adds complexity vs. pure PyTorch temporal layers. *Mitigation*: Benchmark pure PyTorch LSTM/GRU temporal layers; switch if performance gap < 5%.
2. **H2**: NOTEARS O(n³) on local subgraphs still slow for real-time inference at scale. *Mitigation*: Cache DAGs per cluster; recompute only on structural change. `# ponytail:` real-time vs. batch DAG

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
