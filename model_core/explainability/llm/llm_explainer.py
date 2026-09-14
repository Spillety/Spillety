from collections.abc import Awaitable, Callable


class LLMExplainer:
    def __init__(
        self,
        temperature: float = 0.0,
        model_name: str = "meta-llama/Llama-3.1-8B",
        backend: Callable[[dict], Awaitable[str]] | None = None,
    ):
        if temperature != 0.0:
            raise ValueError("temperature must be 0 for deterministic explanations")
        self._temperature = temperature
        self._model_name = model_name
        if backend is None:
            raise RuntimeError("vLLM backend required: pass async backend explicitly")
        self._backend = backend
        self._cache: dict[str, str] = {}

    async def generate_narrative(self, alert_data: dict) -> str:
        key = self._cache_key(alert_data)
        if key in self._cache:
            return self._cache[key]
        narrative = await self._backend(dict(alert_data))
        self._cache[key] = narrative
        return narrative

    @staticmethod
    def _cache_key(alert_data: dict) -> str:
        return str(hash(frozenset(alert_data.items())))
