from unittest.mock import MagicMock, patch

from config.settings import Settings
from config.types import MindProvider, MindRole
from src.infrastructure.mind.factory import create_mind
from src.infrastructure.mind.local.transformers import (
    LocalTransformersMind,
    clear_local_mind_cache,
)
from src.infrastructure.mind.tools import (
    format_say_message,
    format_think_message,
    normalize_tool_reply,
)


def test_normalize_tool_reply_qwen_tool_call() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "say", "arguments": {"text": "Привет"}}\n'
        "</tool_call>"
    )
    assert normalize_tool_reply(raw) == "say Привет"


def test_normalize_tool_reply_think_tool_call() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "think", "arguments": {"topic": "как ответить"}}\n'
        "</tool_call>"
    )
    assert normalize_tool_reply(raw) == "think как ответить"


def test_normalize_tool_reply_arguments_as_string() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "say", "arguments": "{\\"text\\": \\"Ок\\"}"}\n'
        "</tool_call>"
    )
    assert normalize_tool_reply(raw) == "say Ок"


def test_normalize_tool_reply_silent_thought() -> None:
    assert normalize_tool_reply("Жду вопрос пользователя.") == "Жду вопрос пользователя."


def test_normalize_tool_reply_legacy_say_prefix() -> None:
    assert normalize_tool_reply("say Уже так") == "say Уже так"


def test_format_say_message() -> None:
    assert format_say_message("  hi  ") == "say hi"


def test_format_think_message() -> None:
    assert format_think_message("  topic  ") == "think topic"


def test_create_mind_local_primary_enables_say_tool() -> None:
    clear_local_mind_cache()
    tokenizer = MagicMock()
    tokenizer.pad_token = "pad"
    tokenizer.eos_token = "eos"
    model = MagicMock()
    settings = Settings(
        primary_mind_provider=MindProvider.LOCAL,
        primary_mind_model="fake/model",
        primary_mind_device="cpu",
        inner_voice_provider=MindProvider.LOCAL,
        inner_voice_model="fake/model",
        inner_voice_device="cpu",
        inner_voice_api_key=None,
        llm_provider=None,
        llm_model=None,
        llm_device=None,
        llm_api_key=None,
    )
    with (
        patch(
            "src.infrastructure.mind.local.transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "src.infrastructure.mind.local.transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
    ):
        primary = create_mind(settings, role=MindRole.PRIMARY)
        inner = create_mind(settings, role=MindRole.INNER_VOICE)
    assert isinstance(primary, LocalTransformersMind)
    assert isinstance(inner, LocalTransformersMind)
    assert primary._enable_say_tool is True
    assert inner._enable_say_tool is False
    clear_local_mind_cache()
