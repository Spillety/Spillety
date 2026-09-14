import torch
from typing import Optional

class GPUInference:
    """A10G/T4 inference wrapper. Memory-bound at batch 64."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self._model = model.to(device)
        self._device = device
        self._model.eval()
        self._max_batch = 64

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run inference in max_batch chunks; concatenates outputs, no truncation."""
        if batch.size(0) <= self._max_batch:
            with torch.no_grad():
                return self._model(batch.to(self._device))
        outs = []
        with torch.no_grad():
            for chunk in torch.split(batch, self._max_batch, dim=0):
                outs.append(self._model(chunk.to(self._device)))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        """Public inference method with memory-bound batch size."""
        return self.forward(batch)
