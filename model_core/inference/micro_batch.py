from typing import Any

class MicroBatchBuilder:
    """Dynamic padding to 32-64 events. Flink buffer integration simulation."""

    def __init__(self, min_size: int = 32, max_size: int = 64):
        self.min_size = min_size
        self.max_size = max_size
        self._buffer: list[dict] = []

    def build(self, events: list[dict]) -> list[dict]:
        """Pack events into micro-batches without padding; partial tail stays unpadded."""
        self._buffer.extend(events)
        batches = []
        while len(self._buffer) >= self.min_size:
            take = min(len(self._buffer), self.max_size)
            batches.append(self._buffer[:take])
            self._buffer = self._buffer[take:]
        return batches

    def flush(self) -> list[dict]:
        """Return remaining events as a final unpadded batch."""
        if self._buffer:
            batch = self._buffer[:]
            self._buffer.clear()
            return [batch]
        return []
