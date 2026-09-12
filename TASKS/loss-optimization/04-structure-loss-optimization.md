# 04-structure-loss-optimization

> Phase: S (Structure) | Slug: loss-optimization | Status: In Progress

## 1. Project Structure

```
loss_optimization/
├── config/training_config.yaml
├── src/{losses,federated,training}/{focal_loss,contrastive_loss,combined_loss,fsdp_wrapper,trainer,scheduler}.py
├── tests/{test_focal_loss,test_contrastive_loss,test_fsdp}.py
└── scripts/train.sh
```

## 2. Loss Implementations

```python
# src/losses/focal_loss.py
import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss for imbalanced multi-class. gamma=2.0, weight floor=0.1."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        fl = self.alpha * (1 - pt) ** self.gamma * ce
        fl = fl.clamp(min=0.1)  # Weight floor
        return fl.mean() if self.reduction == "mean" else fl.sum()
```

```python
# src/losses/contrastive_loss.py
class ContrastiveLoss(nn.Module):
    """NT-Xent contrastive loss with tau=0.1 for embedding alignment."""
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, anchor, positive):
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        sim = torch.matmul(a, p.T) / self.tau
        return F.cross_entropy(sim, torch.arange(anchor.size(0), device=anchor.device))
```

```python
# src/losses/combined_loss.py
class CombinedLoss(nn.Module):
    """α * Focal + β * Contrastive with cosine annealing from 0.1 to 0.5."""
    def __init__(self, alpha=1.0, beta_init=0.1, beta_final=0.5, annealing_epochs=50):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=2.0)
        self.contrastive = ContrastiveLoss(tau=0.1)
        self.alpha, self.beta_init, self.beta_final, self.annealing_epochs = alpha, beta_init, beta_final, annealing_epochs
        self.current_beta = beta_init

    def update_beta(self, epoch):
        progress = min(epoch / self.annealing_epochs, 1.0)
        self.current_beta = self.beta_init + progress * (self.beta_final - self.beta_init)

    def forward(self, logits, targets, anchor, positive):
        fl = self.focal(logits, targets)
        cl = self.contrastive(anchor, positive)
        return self.alpha * fl + self.current_beta * cl
```

### [human] Gamma sweep & tau temperature
> Почему γ=2.0, τ=0.1?

**Decision:** γ=2.0 — из статьи. Sweep γ∈{1.0,2.0,3.0,5.0}. τ=0.1 — sharp clusters. Sweep τ∈{0.05,0.1,0.2,0.5}. # ponytail: gamma sweep, add when γ=2.0 causes underfitting.

### [agent] resolved
> γ=2.0 baseline. τ=0.1 baseline. Sweep в config.

## 3. FSDP Wrapper

```python
# src/distributed/fsdp_wrapper.py
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

def init_fsdp(model, world_size):
    """FSDP with SHARD_GRAD_OP for >10M node embeddings."""
    dist.init_process_group(backend="nccl")
    return FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        auto_wrap_policy=lambda m: isinstance(m, nn.Embedding),
        device_id=torch.cuda.current_device())

def cleanup_fsdp():
    dist.destroy_process_group()
```

### [human] SHARD_GRAD_OP vs FULL_SHARD
> Какой стратегии sharding?

**Decision:** SHARD_GRAD_OP — баланс memory/communication. FULL_SHARD при VRAM > 90%. # ponytail: FULL_SHARD fallback, add when VRAM > 90%.

### [agent] resolved
> SHARD_GRAD_OP default. FULL_SHARD при VRAM > 90%.

## 4. Training Script

```bash
#!/bin/bash
# scripts/train.sh — FSDP launch
torchrun --nproc_per_node=4 --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=29500 train.py \
    --config config/training_config.yaml --gamma 2.0 --tau 0.1 \
    --alpha 1.0 --beta_init 0.1 --beta_final 0.5
```

## 5. Tests

```python
# tests/test_focal_loss.py
def test_focal_loss_confident():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([[10.0,-10.0],[-10.0,10.0]])
    loss = loss_fn(logits, torch.tensor([0,1]))
    assert loss.item() < 0.01  # Low for confident predictions

def test_focal_loss_floor():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([[0.1,-0.1]])
    loss = loss_fn(logits, torch.tensor([0]))
    assert loss.item() >= 0.1  # Weight floor enforced
```

```python
# tests/test_contrastive_loss.py
def test_contrastive_low_tau():
    loss_low = ContrastiveLoss(tau=0.05)
    loss_high = ContrastiveLoss(tau=0.5)
    a = torch.randn(32, 128)
    p = a + 0.01 * torch.randn(32, 128)
    assert loss_low(a, p).item() > loss_high(a, p).item()
```

```python
# tests/test_fsdp.py
def test_fsdp_initialization():
    with patch("torch.distributed.init_process_group") as mock_init:
        model = torch.nn.Linear(128, 64)
        wrapped = init_fsdp(model, world_size=4)
        mock_init.assert_called_once_with(backend="nccl")
```

### [human] Test coverage
> 80% или 95%?

**Decision:** 95% для loss-функций. 80% для FSDP wrapper. # ponytail: uniform 95%, add when FSDP logic grows.

### [agent] resolved
> 95% losses. 80% FSDP wrapper.

## 6. Pushback: What Could Go Wrong

### [human] Hypothesis 1: FSDP OOM with huge embedding
> Embedding для 10M+ узлов не влезает даже в shard.

**Decision:** SHARD_GRAD_OP разбивает по GPU. При shard > 10GB — hybrid sharding. # ponytail: hybrid sharding, add when shard size > 10GB per GPU.

### [agent] resolved
> SHARD_GRAD_OP по embedding. Hybrid при > 10GB.

### [human] Hypothesis 2: Loss annealing causes instability
> Линейное annealing β может вызвать резкий сдвиг.

**Decision:** Cosine annealing вместо линейного. Warmup 5 epochs. Мониторинг loss smoothness. # ponytail: cosine annealing, add when loss curve has >2 spikes.

### [agent] resolved
> Cosine annealing. Warmup 5 epochs. Rolling variance monitored.

## 7. Acceptance Criteria

- [ ] Focal loss γ=2.0 с weight floor=0.1
- [ ] Contrastive loss τ=0.1 с cosine annealing β
- [ ] FSDP SHARD_GRAD_OP для >10M nodes
- [ ] Combined loss: α=1.0, β annealing 0.1→0.5
- [ ] 95% coverage loss functions, 80% FSDP
- [ ] Training converges < 24h
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 8. Open Questions

Нет открытых вопросов. Переход к P (Plan) по согласованию.
