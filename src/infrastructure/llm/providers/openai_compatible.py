import json
import logging
from collections.abc import AsyncIterator

import httpx

from src.infrastructure.llm.messages import build_chat_messages

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    """OpenAI Chat Completions API и совместимые провайдеры (DeepSeek, Groq, Azure и др.)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 256,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        logger.info("Configured remote LLM provider (model=%s, base_url=%s)", model, self._base_url)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, object]:
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": stream,
        }

    async def generate(self, prompt: str, history: list[dict[str, str]]) -> str:
        messages = build_chat_messages(prompt, history)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post(
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, stream=False),
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])

    async def stream(self, prompt: str, history: list[dict[str, str]]) -> AsyncIterator[str]:
        messages = build_chat_messages(prompt, history)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, stream=True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line.removeprefix("data: ").strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield str(content)
