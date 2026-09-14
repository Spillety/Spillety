import torch
from typing import Optional

# ponytail: batch size tuning.


class GPUInference:
    """A10G/T4 inference wrapper. Memory-bound at batch 64."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self._model = model.to(device)
        self._device = device
        self._model.eval()
        self._max_batch = 64

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch. Clamps to max_batch 64."""
        if batch.size(0) > self._max_batch:
            batch = batch[:self._max_batch]
        with torch.no_grad():
            output = self._model(batch.to(self._device))
        return output

    @torch.no_grad()
    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        """Public inference method with memory-bound batch size."""
        return self.forward(batch)


def demo() -> None:
    """Smoke test: GPUInference with dummy model and batch."""
    import torch
    model = torch.nn.Linear(10, 5)
    inference = GPUInference(model)
    batch = torch.randn(32, 10)
    output = inference.forward(batch)
    assert output.shape == (32, 5)
    large_batch = torch.randn(128, 10)
    out_large = inference.forward(large_batch)
    assert out_large.shape == (64, 5)
    print("GPUInference demo passed")


if __name__ == "__main__":
    demo()
