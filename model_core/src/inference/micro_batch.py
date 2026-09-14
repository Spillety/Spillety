from typing import Any

# ponytail: batch size tuning.


class MicroBatchBuilder:
    """Dynamic padding to 32-64 events. Flink buffer integration simulation."""

    def __init__(self, min_size: int = 32, max_size: int = 64):
        self.min_size = min_size
        self.max_size = max_size
        self._buffer: list[dict] = []

    def build(self, events: list[dict]) -> list[dict]:
        """Pack events into micro-batches with dynamic padding."""
        self._buffer.extend(events)
        batches = []
        while len(self._buffer) >= self.min_size:
            batch = self._buffer[:self.max_size]
            self._buffer = self._buffer[self.max_size:]
            padded = self._pad(batch)
            batches.append(padded)
        return batches

    def _pad(self, batch: list[dict]) -> list[dict]:
        """Pad batch to max_size with zero-tensors if needed."""
        target = self.max_size
        while len(batch) < target:
            batch.append({})
        return batch[:target]

    def flush(self) -> list[dict]:
        """Return remaining events as a final batch."""
        if self._buffer:
            batch = self._pad(self._buffer[:])
            self._buffer.clear()
            return [batch]
        return []


def demo() -> None:
    """Smoke test: micro-batch builder with 32-64 events."""
    import torch
    builder = MicroBatchBuilder(min_size=32, max_size=64)
    events = [{"data": torch.randn(10)} for _ in range(40)]
    batches = builder.build(events)
    assert len(batches) >= 1
    for batch in batches:
        assert len(batch) == 64
    print("MicroBatchBuilder demo passed")


if __name__ == "__main__":
    demo()
