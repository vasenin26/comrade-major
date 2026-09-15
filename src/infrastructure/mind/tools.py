import json
import re
from typing import Any

SAY_TOOL = {
    "type": "function",
    "function": {
        "name": "say",
        "description": (
            "Send a short message to the user (speech). ONLY channel they receive. "
            "Use sparingly — real replies to what they said, or a thought you "
            "choose to share. Never send helpdesk filler "
            "('чем помочь', 'жду вопрос', 'дайте тему')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Exact words to send to the user "
                        "(Russian, short, 1–2 sentences)."
                    ),
                }
            },
            "required": ["text"],
        },
    },
}

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Ask the inner mind for a steering note. Blocks until finished. "
            "Result overwrites [INNER]. Use when stuck, helpdesk-looping, or "
            "unsure how to continue a private theme."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "What the inner mind should advise on.",
                }
            },
            "required": ["topic"],
        },
    },
}

# Schema list for HuggingFace chat templates (Qwen and similar).
PRIMARY_TOOLS_FOR_CHAT: list[dict[str, Any]] = [SAY_TOOL, THINK_TOOL]
# Backward-compatible alias
SAY_TOOLS_FOR_CHAT = PRIMARY_TOOLS_FOR_CHAT


def format_say_message(text: str) -> str:
    return f"say {text.strip()}"


def format_think_message(topic: str) -> str:
    return f"think {topic.strip()}"


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _tool_name_and_args(payload: object) -> tuple[str | None, dict[str, object]]:
    if not isinstance(payload, dict):
        return None, {}
    name = payload.get("name")
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        payload = function
    if not isinstance(name, str):
        return None, {}
    args: object = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return name, {"_raw": args.strip()}
        args = parsed
    if not isinstance(args, dict):
        return name, {}
    return name, args


def _say_text_from_payload(payload: object) -> str | None:
    name, args = _tool_name_and_args(payload)
    if name != "say":
        return None
    if "_raw" in args and len(args) == 1:
        return str(args["_raw"]) or None
    text = str(args.get("text", "")).strip()
    return text or None


def _think_topic_from_payload(payload: object) -> str | None:
    name, args = _tool_name_and_args(payload)
    if name != "think":
        return None
    if "_raw" in args and len(args) == 1:
        return str(args["_raw"]) or None
    topic = str(args.get("topic", "")).strip()
    return topic or None


def normalize_tool_reply(raw: str) -> str:
    """Map model output to PrimaryThinkingLoop contract.

    - ``say`` tool → ``say …``
    - ``think`` tool → ``think …``
    - Otherwise return stripped text (silent think), including legacy prefixes.
    """
    stripped = raw.strip()
    if not stripped:
        return ""

    for match in _TOOL_CALL_RE.finditer(stripped):
        body = match.group(1).strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        say_text = _say_text_from_payload(payload)
        if say_text:
            return format_say_message(say_text)
        think_topic = _think_topic_from_payload(payload)
        if think_topic:
            return format_think_message(think_topic)

    # Bare JSON tool call without tags
    if stripped.startswith("{") and '"name"' in stripped:
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            say_text = _say_text_from_payload(payload)
            if say_text:
                return format_say_message(say_text)
            think_topic = _think_topic_from_payload(payload)
            if think_topic:
                return format_think_message(think_topic)

    return stripped
