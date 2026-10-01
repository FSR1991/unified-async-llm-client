from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    RateLimitError,
)

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    """Cliente asíncrono para OpenAI."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key)

    def _build_messages(
        self,
        messages: list[ChatMessage],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    async def generate(
        self,
        messages: list[ChatMessage],
    ) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self._build_messages(messages),
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content or ""

            return ModelResponse(
                content=content,
                model=self.config.model,
                provider="openai",
            )

        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI: se alcanzó el límite de solicitudes."
            ) from exc

        except APIConnectionError as exc:
            raise RuntimeError(
                "OpenAI: no fue posible conectarse con el servicio."
            ) from exc

        except APIError as exc:
            raise RuntimeError(
                f"OpenAI: error de API: {exc}"
            ) from exc

    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self._build_messages(messages),
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                token = chunk.choices[0].delta.content

                if token:
                    yield token

        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI: se alcanzó el límite de solicitudes durante el streaming."
            ) from exc

        except APIConnectionError as exc:
            raise RuntimeError(
                "OpenAI: no fue posible conectarse durante el streaming."
            ) from exc

        except APIError as exc:
            raise RuntimeError(
                f"OpenAI: error de API durante el streaming: {exc}"
            ) from exc