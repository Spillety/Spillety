# Model Integration — Structure

> Phase: S (Structure) | Slug: model-integration | Status: Draft

## 1. Project Structure

```
model_core/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── split.py
│   │   └── negatives.py
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── focal_loss.py
│   │   ├── contrastive_loss.py
│   │   ├── combined_loss.py
│   │   └── fsdp_wrapper.py
│   ├── feature_store/
│   │   ├── __init__.py
│   │   ├── redis_client.py
│   │   ├── drift_monitor.py
│   │   └── expectations.py
│   ├── causal_attention/
│   │   ├── __init__.py
│   │   ├── notears_dag.py
│   │   ├── causal_attention.py
│   │   ├── temporal_encoder.py
│   │   └── domain_prior.py
│   ├── hyperbolic/
│   │   ├── __init__.py
│   │   ├── lorentz_ops.py
│   │   ├── hyperbolic_linear.py
│   │   ├── message_passing.py
│   │   └── curvature.py
│   ├── hawkes/
│   │   ├── __init__.py
│   │   ├── power_law.py
│   │   ├── seasonal_mu.py
│   │   └── updater.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── bfs_subgraph.py
│   │   ├── micro_batch.py
│   │   ├── gpu_forward.py
│   │   └── subgraph_cache.py
│   └── explainability/
│       ├── __init__.py
│       ├── influence_functions.py
│       ├── path_ranking.py
│       ├── hnsw_lorentz.py
│       ├── embedding_store.py
│       ├── schema/
│       │   ├── __init__.py
│       │   ├── aml_schema.json
│       │   └── validator.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── llm_explainer.py
│       │   └── narrative.py
│       └── jurisdiction/
│           ├── __init__.py
│           ├── mapping.py
│           └── mapping.yaml
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_hawkes.py
│   ├── test_causal_attention.py
│   ├── test_hyperbolic.py
│   ├── test_inference.py
│   └── test_explainability.py
└── config/
    ├── model_config.yaml
    ├── split_config.json
    └── training_config.yaml
```

## 2. `pyproject.toml`

```toml
[project]
name = "model_core"
version = "1.0.0"
description = "K-BRAIN Model Core: Causal Attention, Hyperbolic MP, Training Pipeline"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "numpy>=1.26",
    "scipy>=1.12",
    "scikit-learn>=1.4",
    "pandas>=2.1",
    "pyyaml>=6.0",
    "redis>=5.0",
    "hnswlib>=0.8",
    "networkx>=3.2",
    "transformers>=4.40",
    "accelerate>=0.30",
    "torchmetrics>=1.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
]
fsdp = [
    "torch>=2.2.0",
]
```

## 3. Key Components

### 3.1 `dataset/`
- `loader.py`: Elliptic++ v2 parquet loader, 6-class mapping
- `split.py`: Temporal split with config-driven boundaries
- `negatives.py`: KNN hard negative sampling (sklearn for MVP)

### 3.2 `loss/`
- `focal_loss.py`: Focal loss γ=2.0, weight floor=0.1
- `contrastive_loss.py`: NT-Xent contrastive τ=0.1
- `combined_loss.py: α=1.0, β annealing 0.1→0.5
- `fsdp_wrapper.py`: FSDP SHARD_GRAD_OP for >10M nodes

### 3.3 `feature_store/`
- `redis_client.py`: Redis online, PostgreSQL offline (Feast-free MVP)
- `drift_monitor.py`: KS test + Wasserstein fallback
- `expectations.py`: 5 AML custom validators + schema/null/range/uniqueness

### 3.4 `causal_attention/`
- `notears_dag.py`: Differentiable DAG via adjacency matrix
- `causal_attention.py`: Backdoor-adjusted attention weights
- `temporal_encoder.py`: Pure PyTorch temporal convolution
- `domain_prior.py`: Expert graph encoding, edge type constraints

### 3.5 `hyperbolic/`
- `lorentz_ops.py`: Möbius addition, expmap, logmap
- `hyperbolic_linear.py`: Hyperbolic linear layer
- `message_passing.py`: Hyperbolic GNN message passing
- `curvature.py`: Learnable curvature nn.Parameter, sigmoid-constrained

### 3.6 `hawkes/`
- `power_law.py`: k(t) = τ^{-α}, TTL-truncated history
- `seasonal_mu.py`: Fourier seasonality (24h + 7d)
- `updater.py`: O(1) closed-form λ(t) increment

### 3.7 `inference/`
- `bfs_subgraph.py`: BFS depth 2 union from source + target
- `micro_batch.py`: Dynamic padding 32-64
- `gpu_forward.py`: A10G/T4 inference
- `subgraph_cache.py`: LRU cache 10k entries, 1h TTL

### 3.8 `explainability/`
- `influence_functions.py`: Koh & Liang approximation
- `path_ranking.py`: Top-3 causal paths via backdoor-adjusted effect
- `hnsw_lorentz.py`: HNSW index in Lorentz space
- `embedding_store.py`: Lorentz embeddings storage
- `schema/`: JSON Schema draft 2020-12, AML custom validator
- `llm/`: vLLM backend, LLaMA 3.1 8B Q4_K_M, Temperature=0
- `jurisdiction/`: Hardcoded dict, YAML source

## 4. Config Files

### `config/split_config.json`
```json
{"temporal_split": {"train_end": "2015-12-31", "val_end": "2016-06-30", "test_end": "2016-12-31"}, "hard_negative": {"knn_k": 10, "cosine_threshold": 0.85, "ratio_hard_easy": 0.5}, "labels": ["scam","ransomware","terrorist_financing","sanctions","mixer","legitimate"], "seed": 42, "label_confidence_threshold": 0.8}
```

### `config/training_config.yaml`
```yaml
model:
  num_classes: 6
  embedding_dim: 128
  causal_depth: 2
  hyperbolic_dim: 128

loss:
  alpha: 1.0
  beta_init: 0.1
  beta_final: 0.5
  annealing_epochs: 50
  focal_gamma: 2.0
  contrastive_tau: 0.1

fsdp:
  sharding_strategy: SHARD_GRAD_OP
  backend: nccl
  cpu_offloading: false

training:
  epochs: 100
  batch_size: 64
  lr: 1e-4
  optimizer: adamw
  warmup_epochs: 5
```

## 5. Tests

### `tests/test_dataset.py`
- Temporal split boundaries reproducible
- Hard negative ratio 1:1, cos_sim > 0.85
- Label confidence > 0.8, label=-1 filtered
- Address-level split enforced

### `tests/test_loss.py`
- Focal loss confident predictions → low loss
- Focal loss floor enforced (≥ 0.1)
- Contrastive low τ → higher loss than high τ
- FSDP initialization with SHARD_GRAD_OP
- Combined loss β annealing 0.1→0.5

### `tests/test_hawkes.py`
- Power-law kernel: α converges within 10k events
- λ(t) p99 latency < 1ms
- Seasonal μ(t) Fourier components
- O(1) update additive

### `tests/test_causal_attention.py`
- NOTEARS DAG reconstruction accuracy ≥ 0.85
- Domain prior masks edge types correctly
- Temporal encoder produces valid attention weights
- Numerical stability (gradient norms bounded)

### `tests/test_hyperbolic.py`
- Lorentz ops: expmap/logmap consistency
- Curvature stays within [-2.0, -0.1)
- No NaN/Inf in forward pass
- Gradient clipping max_norm=1.0

### `tests/test_inference.py`
- BFS depth 2 subgraph extraction
- LRU cache hit rate > 60% for hot addresses
- Micro-batch 32-64 events
- p99 inference latency < 200ms

### `tests/test_explainability.py`
- Influence functions: counterfactual fidelity ≥ 0.9
- HNSW recall@k ≥ 0.95
- JSON Schema validation passes
- Jurisdiction mapping returns correct regulation

## 6. Pushback: What Could Go Wrong

### [human] H1: Shared `pyproject.toml` dependency conflicts
> Разные подмодули могут нуждаться в разных версиях.

**Decision:** Один pyproject.toml с pinned versions. Мониторинг конфликтов. # ponytail: split pyproject, add when conflict detected.

### [agent] resolved
> Один pyproject.toml. Pinned versions.

### [human] H2: HNSW Lorentz не совместим с hnswlib по умолчанию
> hnswlib работает с евклидовым расстоянием, Lorentz — нет.

**Decision:** Кастомная метрика через `hnswlib` с `space='cosine'` и нормализацией в Lorentz. При нехватке — FAISS custom. # ponytail: hnswlib cosine, add when recall < 0.95.

### [agent] resolved
> hnswlib с cosine similarity + Lorentz normalization. FAISS fallback при нехватке.

### [human] H3: LLaMA 3.1 8B не помещается в A10G 24GB с vLLM
> vLLM overhead + KV cache может превысить VRAM.

**Decision:** GGUF Q4_K_M quantization. vLLM с `gpu_memory_utilization=0.90`. Async generation с кэшем narrative. # ponytail: quantization, add when VRAM > 90%.

### [agent] resolved
> Q4_K_M quantization. vLLM config. Async with cache.

## 7. Self-Review

- 0 🔴 Critical
- 0 ai-slops
- All `# ponytail:` markers present
- All AC from design docs checked
- English comments in code, Russian prose in docs
- Under 200 lines per section

## 8. Open Questions

Нет открытых вопросов. Переход к P (Plan) по согласованию.
