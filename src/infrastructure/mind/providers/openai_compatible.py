import json
import logging
from collections.abc import AsyncIterator

import httpx

from src.infrastructure.mind.tools import (
    PRIMARY_TOOLS_FOR_CHAT,
    format_say_message,
    format_think_message,
)

logger = logging.getLogger(__name__)

_DEFAULT_REASONING_EFFORT = "minimal"


def is_reasoning_model(model: str) -> bool:
    """GPT-5 / o-series models need max_completion_tokens and reject temperature."""
    name = model.lower().split("/", 1)[-1]
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAICompatibleMind:
    """OpenAI Chat Completions API and compatible providers."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 256,
        temperature: float = 0.7,
        timeout: float = 120.0,
        enable_say_tool: bool = True,
        reasoning_effort: str | None = None,
        proxy: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._enable_say_tool = enable_say_tool
        self._proxy = proxy.strip() if proxy and proxy.strip() else None
        if reasoning_effort is not None:
            self._reasoning_effort: str | None = reasoning_effort
        elif is_reasoning_model(model):
            self._reasoning_effort = _DEFAULT_REASONING_EFFORT
        else:
            self._reasoning_effort = None
        logger.info(
            "Configured remote mind (model=%s, base_url=%s, proxy=%s)",
            model,
            self._base_url,
            "yes" if self._proxy else "no",
        )

    def _client_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
        }
        if self._proxy is not None:
            kwargs["proxy"] = self._proxy
        return kwargs

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        if is_reasoning_model(self._model):
            payload["max_completion_tokens"] = self._max_tokens
            if self._reasoning_effort is not None:
                payload["reasoning_effort"] = self._reasoning_effort
        else:
            payload["max_tokens"] = self._max_tokens
            payload["temperature"] = self._temperature
        if self._enable_say_tool and not stream:
            payload["tools"] = PRIMARY_TOOLS_FOR_CHAT
            payload["tool_choice"] = "auto"
        return payload

    def _extract_reply(self, data: dict[str, object]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                raw_args = function.get("arguments", "{}")
                try:
                    args = json.loads(str(raw_args))
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if name == "say":
                    text = str(args.get("text", "")).strip()
                    if text:
                        return format_say_message(text)
                if name == "think":
                    topic = str(args.get("topic", "")).strip()
                    if topic:
                        return format_think_message(topic)

        content = message.get("content")
        return str(content or "").strip()

    async def think(self, history: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            response = await client.post(
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(history, stream=False),
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return ""
            return self._extract_reply(data)

    async def stream(self, history: list[dict[str, str]]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(history, stream=True),
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
