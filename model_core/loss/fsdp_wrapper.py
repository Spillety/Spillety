import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import torch.nn as nn


def init_fsdp(model: nn.Module) -> nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP requires GPU: no CUDA device available")
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_id=torch.cuda.current_device(),
    )


def cleanup_fsdp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
