from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Interfaz común para clientes LLM asíncronos."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
    ) -> ModelResponse:
        """Genera una respuesta completa de forma asíncrona."""
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        """Genera una respuesta mediante streaming de tokens."""
        raise NotImplementedError