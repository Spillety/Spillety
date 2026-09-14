import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import torch.nn as nn


def init_fsdp(model: nn.Module) -> FSDP:
    dist.init_process_group(backend="nccl")
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_id=torch.cuda.current_device(),
    )


def cleanup_fsdp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ponytail: FULL_SHARD fallback, add when VRAM > 90%.


def demo() -> None:
    import unittest.mock as mock
    with mock.patch("torch.distributed.init_process_group"), mock.patch("torch.cuda.current_device", return_value=0):
        model = nn.Linear(128, 64)
        wrapped = init_fsdp(model)
        assert isinstance(wrapped, FSDP)
        cleanup_fsdp()
    print("fsdp_wrapper demo passed")


if __name__ == "__main__":
    demo()
