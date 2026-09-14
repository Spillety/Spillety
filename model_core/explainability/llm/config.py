from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:

    model_name: str = "meta-llama/Llama-3.1-8B"
    temperature: float = 0.0
    max_tokens: int = 512
    backend: str = "vllm"
