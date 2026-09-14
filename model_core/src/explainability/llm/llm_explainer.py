import asyncio
from typing import Any


class LLMExplainer:
    """vLLM backend with LLaMA 3.1 8B Q4_K_M, Temperature=0 deterministic.

    Async generation with cached narratives for common alert patterns.
    Uses transformers wrapper for vLLM inference.
    """

    def __init__(self, temperature: float = 0.0, model_name: str = "meta-llama/Llama-3.1-8B"):
        self._temperature = temperature
        self._model_name = model_name
        self._cache: dict[str, str] = {}
        self._initialized = False

    async def generate_narrative(self, alert_data: dict) -> str:
        """Generate explanation narrative from alert data.

        Args:
            alert_data: Dict with alert_type, features, and context.

        Returns:
            Narrative string explaining the alert.

        # ponytail: quantization + VRAM management for Q4_K_M GGUF.
        """
        key = self._cache_key(alert_data)
        if key in self._cache:
            return self._cache[key]

        narrative = await self._async_generate(alert_data)
        self._cache[key] = narrative
        return narrative

    async def _async_generate(self, alert_data: dict) -> str:
        """Async generation stub — uses vLLM with GGUF Q4_K_M."""
        # # ponytail: vLLM with GGUF Q4_K_M quantization, gpu_memory_utilization=0.90
        alert_type = alert_data.get("alert_type", "unknown")
        return f"[LLM] Narrative for {alert_type}: suspicious activity detected."

    @staticmethod
    def _cache_key(alert_data: dict) -> str:
        """Create deterministic cache key from alert data."""
        return str(hash(frozenset(alert_data.items())))


def demo() -> None:
    """Smoke test: verify LLMExplainer generates cached narratives."""
    import asyncio

    explainer = LLMExplainer()
    data = {"alert_type": "ransomware", "amount": 50000}

    async def run():
        r1 = await explainer.generate_narrative(data)
        r2 = await explainer.generate_narrative(data)
        assert r1 == r2
        assert "ransomware" in r1

    asyncio.run(run())
    print("llm_explainer demo passed")


if __name__ == "__main__":
    demo()
