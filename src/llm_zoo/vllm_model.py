import logging
import time
from typing import List

from vllm import LLM, SamplingParams

from src.llm_zoo.base_model import BaseLLM

logger = logging.getLogger(__name__)


class VLLMModel(BaseLLM):
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        super().__init__(model_name_or_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Loading vLLM model: {model_name_or_path}")
        t0 = time.time()
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def _sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

    def _build_messages(self, prompt: str, system_prompt: str = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        outputs = self.llm.chat(
            messages=[self._build_messages(prompt, system_prompt)],
            sampling_params=self._sampling_params(),
        )
        return outputs[0].outputs[0].text.strip()

    def _sync_batch_invoke(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        all_messages = [self._build_messages(p, system_prompt) for p in prompts]
        outputs = self.llm.chat(
            messages=all_messages,
            sampling_params=self._sampling_params(),
        )
        return [o.outputs[0].text.strip() for o in outputs]

    async def batch_invoke(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        return self._sync_batch_invoke(prompts, system_prompt)
