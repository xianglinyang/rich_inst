from typing import Literal

from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.openrouter_model import OpenRouterModel
from src.llm_zoo.cost_utils import CallResult

__all__ = ["load_model", "BaseLLM", "OpenRouterModel", "VLLMModel", "CallResult"]


def __getattr__(name: str):
    if name == "VLLMModel":
        from src.llm_zoo.vllm_model import VLLMModel
        return VLLMModel
    raise AttributeError(name)


def load_model(llm_name: str, mode: Literal["api", "vllm"] = "api") -> BaseLLM:
    """
    Load an LLM by name and backend mode.

    Parameters
    ----------
    llm_name : model identifier
        "api"  → OpenRouter model ID, e.g. "openai/gpt-4o-mini"
        "vllm" → HuggingFace model path, e.g. "meta-llama/Llama-3.1-8B-Instruct"
    mode : "api" | "vllm"
        "api"  → remote inference via OpenRouter (requires OPENROUTER_API_KEY)
        "vllm" → local GPU inference via vLLM
    """
    if mode == "api":
        return OpenRouterModel(llm_name)
    if mode == "vllm":
        from src.llm_zoo.vllm_model import VLLMModel
        return VLLMModel(llm_name)
    raise ValueError(f"Unknown mode {mode!r}. Choose 'api' or 'vllm'.")
