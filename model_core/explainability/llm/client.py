from typing import Any

from model_core.explainability.llm.config import LLMConfig


class LLMClient:
    SOURCE_VLLM = "vllm"

    def __init__(self, config: LLMConfig | None = None, engine: Any | None = None) -> None:
        self._config = config or LLMConfig()
        if self._config.temperature != 0.0:
            raise ValueError("temperature must be 0 for deterministic explanations")
        if engine is None:
            raise RuntimeError("vLLM engine required: pass engine explicitly (vLLM + LLaMA weights + GPU)")
        self._engine = engine

    @property
    def source(self) -> str:
        return self.SOURCE_VLLM

    def explain(self, alert: dict) -> dict:
        alert["alert_type"]
        self._serialize_path(alert["explanation"])
        self._detect_jurisdiction(alert["explanation"])
        text = self._vllm_generate(alert)
        return {"text": text, "source": self.SOURCE_VLLM, "model": self._config.model_name}

    def _vllm_generate(self, alert: dict) -> str:
        generate = getattr(self._engine, "generate", None)
        if callable(generate):
            return str(generate(alert))
        if callable(self._engine):
            return str(self._engine(alert))
        raise RuntimeError("vLLM engine must be callable or expose generate(alert)")

    @staticmethod
    def _serialize_path(explanation: dict) -> str:
        edges = explanation["causal_path"]
        return "->".join(str(e["edge"]) for e in edges)

    @staticmethod
    def _detect_jurisdiction(explanation: dict) -> str:
        refs = " ".join(explanation["regulatory_references"])
        upper = refs.upper()
        for code in ("US", "EU", "UK"):
            if code in upper:
                return code
        raise KeyError("unsupported jurisdiction: no US/EU/UK marker in regulatory_references")
