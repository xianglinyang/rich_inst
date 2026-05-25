import asyncio
import json
import logging
import os
import time
from typing import List, Union

from openai import AsyncOpenAI, OpenAI

from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.cost_utils import CallResult, calculate_cost
from src.llm_zoo.rate_limiter import OPENROUTER_RATE_LIMIT, rate_limited_async_call

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.environ["OPENROUTER_API_KEY"]
        self.client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=_BASE_URL)

    def _build_messages(self, prompt: str, system_prompt: str = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        return_cost: bool = False,
        max_retries: int = 3,
    ) -> Union[str, CallResult]:
        messages = self._build_messages(prompt, system_prompt)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                if not response.choices:
                    raise ValueError(f"Empty choices from {self.model_name}: {response}")
                content = response.choices[0].message.content.strip()
                if not return_cost:
                    return content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                return CallResult(
                    response=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=calculate_cost(self.model_name, input_tokens, output_tokens),
                    model_name=self.model_name,
                )
            except (json.JSONDecodeError, ValueError, Exception) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"invoke attempt {attempt + 1}/{max_retries} failed "
                        f"[{self.model_name}]: {e!r} — retrying"
                    )
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"invoke failed after {max_retries} attempts [{self.model_name}]: {e!r}")
                    raise

    @rate_limited_async_call(OPENROUTER_RATE_LIMIT)
    async def _get_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        return_cost: bool = False,
    ) -> Union[str, CallResult, None]:
        messages = self._build_messages(prompt, system_prompt)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                if not response.choices:
                    raise ValueError(f"Empty choices from {self.model_name}: {response}")
                content = response.choices[0].message.content.strip()
                if not return_cost:
                    return content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                return CallResult(
                    response=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=calculate_cost(self.model_name, input_tokens, output_tokens),
                    model_name=self.model_name,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"async attempt {attempt + 1}/{max_retries} failed "
                        f"[{self.model_name}]: {e!r}"
                    )
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"async failed after {max_retries} attempts [{self.model_name}]: {e!r}")
                    return None

    async def batch_invoke(
        self,
        prompts: List[str],
        system_prompt: str = None,
        return_cost: bool = False,
    ) -> List[Union[str, CallResult, None]]:
        tasks = [self._get_completion(p, system_prompt, return_cost) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [None if isinstance(r, Exception) else r for r in results]
