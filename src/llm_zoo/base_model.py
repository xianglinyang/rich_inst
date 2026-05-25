from abc import ABC, abstractmethod
from typing import List


class BaseLLM(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        pass

    @abstractmethod
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        pass
