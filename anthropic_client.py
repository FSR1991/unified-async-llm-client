from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError,
    APIError,
    AsyncAnthropic,
    RateLimitError,
)

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    """Cliente asíncrono para Anthropic."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key)

    def _build_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[dict[str, str]]]:
        system_message = None
        conversation = []

        for message in messages:
            if message.role == "system":
                system_message = message.content
            else:
                conversation.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        return system_message, conversation

    async def generate(
        self,
        messages: list[ChatMessage],
    ) -> ModelResponse:
        try:
            system_message, conversation = self._build_messages(messages)

            kwargs = {
                "model": self.config.model,
                "messages": conversation,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            if system_message:
                kwargs["system"] = system_message

            response = await self.client.messages.create(**kwargs)

            content = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            return ModelResponse(
                content=content,
                model=self.config.model,
                provider="anthropic",
            )

        except RateLimitError as exc:
            raise RuntimeError(
                "Anthropic: se alcanzó el límite de solicitudes."
            ) from exc

        except APIConnectionError as exc:
            raise RuntimeError(
                "Anthropic: no fue posible conectarse con el servicio."
            ) from exc

        except APIError as exc:
            raise RuntimeError(
                f"Anthropic: error de API: {exc}"
            ) from exc

    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        try:
            system_message, conversation = self._build_messages(messages)

            kwargs = {
                "model": self.config.model,
                "messages": conversation,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            if system_message:
                kwargs["system"] = system_message

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except RateLimitError as exc:
            raise RuntimeError(
                "Anthropic: se alcanzó el límite de solicitudes durante el streaming."
            ) from exc

        except APIConnectionError as exc:
            raise RuntimeError(
                "Anthropic: no fue posible conectarse durante el streaming."
            ) from exc

        except APIError as exc:
            raise RuntimeError(
                f"Anthropic: error de API durante el streaming: {exc}"
            ) from exc