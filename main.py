import asyncio
import os

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig


def get_configuration() -> tuple[str, ModelConfig]:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "")
    elif provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "")
    else:
        raise ValueError(
            "LLM_PROVIDER debe ser 'openai' o 'anthropic'."
        )

    if not model:
        raise ValueError(
            f"No se configuró el modelo para el proveedor '{provider}'."
        )

    config = ModelConfig(
        model=model,
        temperature=0.7,
        max_tokens=1024,
    )

    return provider, config


async def run_normal(client, messages: list[ChatMessage]) -> None:
    print("\n=== RESPUESTA NORMAL ===")

    response = await client.generate(messages)

    print(response.content)


async def run_streaming(client, messages: list[ChatMessage]) -> None:
    print("\n=== STREAMING ===")

    async for token in client.stream(messages):
        print(token, end="", flush=True)

    print()


async def main() -> None:
    load_dotenv()

    provider, config = get_configuration()

    manager = AsyncLLMManager(
        provider=provider,
        config=config,
    )

    client = manager.get_client()

    messages = [
        ChatMessage(
            role="user",
            content="¿Qué es la entropía?",
        )
    ]

    await run_normal(client, messages)
    await run_streaming(client, messages)


if __name__ == "__main__":
    asyncio.run(main())