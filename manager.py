import os

from anthropic_client import AnthropicClient
from base_client import BaseLLMClient
from openai_client import OpenAIClient
from schemas import ModelConfig


class AsyncLLMManager:
    """Selecciona y construye el cliente LLM configurado."""

    def __init__(
        self,
        provider: str,
        config: ModelConfig,
    ) -> None:
        self.provider = provider.lower()
        self.config = config
        self.client = self._create_client()

    def _create_client(self) -> BaseLLMClient:
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY no está configurada."
                )

            return OpenAIClient(
                config=self.config,
                api_key=api_key,
            )

        if self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY no está configurada."
                )

            return AnthropicClient(
                config=self.config,
                api_key=api_key,
            )

        raise ValueError(
            f"Proveedor no soportado: {self.provider}. "
            "Usá 'openai' o 'anthropic'."
        )

    def get_client(self) -> BaseLLMClient:
        """Devuelve el cliente LLM configurado."""
        return self.client