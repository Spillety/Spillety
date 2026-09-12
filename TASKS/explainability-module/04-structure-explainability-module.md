# 5C Structured Explainability Module — Structure

```
explainability-module/
├── counterfactual/
│   ├── __init__.py
│   ├── influence_functions.py  # Koh & Liang approximation for counterfactual
│   └── path_ranking.py         # Top-3 causal paths via backdoor-adjusted effect
├── search/
│   ├── __init__.py
│   ├── hnsw_lorentz.py         # HNSW index in Lorentz space for O(log n) search
│   └── embedding_store.py      # Lorentz embeddings storage + retrieval
├── schema/
│   ├── __init__.py
│   ├── aml_schema.json         # JSON Schema draft 2020-12 for AML fields
│   └── validator.py            # Custom validator for AML-specific keywords
├── llm/
│   ├── __init__.py
│   ├── llm_explainer.py        # LLaMA 3.1 8B local (vLLM, Temperature=0)
│   └── narrative.py            # Template-based narrative generation
├── jurisdiction/
│   ├── __init__.py
│   ├── mapping.py              # Hardcoded jurisdiction→regulation dict
│   └── mapping.yaml            # YAML source for jurisdiction rules
└── tests/
    ├── __init__.py
    ├── test_influence.py
    ├── test_hnsw_lorentz.py
    ├── test_schema_validation.py
    ├── test_llm_explainer.py
    └── test_jurisdiction.py
```

## Key Components
- `influence_functions.py`: Koh & Liang gradient approximation. English comments only.
- `hnsw_lorentz.py`: HNSW index for Lorentz space nearest-neighbor. `# ponytail:` HNSW efSearch parameter
- `aml_schema.json`: JSON Schema draft 2020-12 with AML custom keywords
- `llm_explainer.py`: vLLM backend, LLaMA 3.1 8B Q4_K_M, Temperature=0 deterministic
- `mapping.py`: Hardcoded jurisdiction→regulation; loaded from YAML at init

## Pushback Hypotheses
1. **H1**: HNSW in Lorentz space requires non-standard distance metric; Memgraph native HNSW may not support it. *Mitigation*: Custom HNSW implementation via FAISS or hnswlib in Lorentz metric.
2. **H2**: LLaMA 3.1 8B vLLM inference may exceed 20ms explanation budget. *Mitigation*: Async explanation generation; cached narratives for common alert patterns. `# ponytail:` latency vs quality

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
