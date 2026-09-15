from collections.abc import AsyncIterator

from config.settings import Settings, get_settings
from config.types import MindProvider, MindRole
from src.infrastructure.mind.factory import create_mind
from src.infrastructure.mind.providers.openai_compatible import OpenAICompatibleMind


def test_settings_defaults() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=None,
        llm_model=None,
        llm_device=None,
        llm_max_new_tokens=None,
        llm_temperature=None,
        llm_api_key=None,
        llm_base_url=None,
        primary_mind_provider=MindProvider.LOCAL,
        memory_enabled=False,
    )
    assert settings.sample_rate == 16_000
    assert settings.vad_threshold == 0.5
    assert settings.whisper_model_size == "small"
    assert settings.primary_mind_provider == MindProvider.LOCAL
    assert settings.primary_mind_model == "Qwen/Qwen2.5-3B-Instruct"
    assert "Comrade Major" in settings.primary_mind_system_prompt
    assert "Russian" in settings.primary_mind_system_prompt
    assert "say" in settings.primary_mind_system_prompt
    assert "think" in settings.primary_mind_system_prompt
    assert "SEND TO USER" in settings.primary_mind_system_prompt
    assert "PRIVATE THINK" in settings.primary_mind_system_prompt
    assert "Hard bans" in settings.primary_mind_system_prompt
    assert "[THOUGHTS]" in settings.primary_mind_system_prompt
    assert "[MEMORY]" in settings.primary_mind_system_prompt
    assert "pulse" in settings.primary_mind_system_prompt.lower()
    assert "tool" in settings.primary_mind_system_prompt.lower()
    assert "never use 'say '" in settings.inner_voice_system_prompt
    assert "loop" in settings.inner_voice_system_prompt.lower()
    assert "helpdesk" in settings.inner_voice_system_prompt.lower()
    assert "theme" in settings.inner_voice_system_prompt.lower()
    assert "[THOUGHTS]" in settings.inner_voice_system_prompt
    assert settings.primary_think_pulse_seconds == 15.0
    assert settings.primary_thought_history == 8
    assert settings.inner_voice_interval_seconds == 60.0
    assert settings.tts_speaker == "aidar"
    assert settings.tts_device == "auto"
    assert settings.tts_synthesis_sample_rate == 24_000
    assert settings.tts_listen_hangover_ms == 400
    assert settings.resolved_tts_device() in ("cpu", "cuda")
    assert settings.ui_host == "127.0.0.1"
    assert settings.ui_port == 8765


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_factory_creates_openai_mind() -> None:
    settings = Settings(
        primary_mind_provider=MindProvider.OPENAI,
        primary_mind_api_key="test-key",
        primary_mind_model="gpt-4o-mini",
        llm_provider=None,
        llm_api_key=None,
    )
    service = create_mind(settings, role=MindRole.PRIMARY)
    assert isinstance(service, OpenAICompatibleMind)


def test_openai_provider_requires_api_key() -> None:
    try:
        Settings(
            primary_mind_provider=MindProvider.OPENAI,
            primary_mind_api_key=None,
            llm_provider=None,
            llm_api_key=None,
            memory_enabled=False,
        )
        raise AssertionError("Expected validation error")
    except ValueError as exc:
        assert "API key" in str(exc)


def test_memory_enabled_requires_api_key() -> None:
    try:
        Settings(
            _env_file=None,
            primary_mind_provider=MindProvider.LOCAL,
            primary_mind_api_key=None,
            inner_voice_api_key=None,
            llm_provider=None,
            llm_api_key=None,
            memory_enabled=True,
        )
        raise AssertionError("Expected validation error")
    except ValueError as exc:
        assert "MEMORY_ENABLED" in str(exc)


def test_legacy_llm_env_aliases() -> None:
    settings = Settings(
        llm_provider=MindProvider.OPENAI,
        llm_model="gpt-4o-mini",
        llm_api_key="legacy-key",
        primary_mind_provider=MindProvider.LOCAL,
        primary_mind_api_key=None,
    )
    assert settings.primary_mind_provider == MindProvider.OPENAI
    assert settings.primary_mind_model == "gpt-4o-mini"
    assert settings.primary_mind_api_key == "legacy-key"


def test_inner_voice_falls_back_to_primary() -> None:
    settings = Settings(
        primary_mind_model="primary-model",
        primary_mind_provider=MindProvider.LOCAL,
        memory_enabled=False,
        inner_voice_provider=None,
        inner_voice_model=None,
        inner_voice_device=None,
        inner_voice_max_new_tokens=None,
        inner_voice_temperature=None,
        inner_voice_api_key=None,
        inner_voice_base_url=None,
        inner_voice_proxy=None,
        llm_provider=None,
        llm_model=None,
    )
    assert settings.resolved_model(MindRole.INNER_VOICE) == "primary-model"
    assert settings.resolved_provider(MindRole.INNER_VOICE) == MindProvider.LOCAL


def test_factory_creates_openai_inner_with_local_primary() -> None:
    settings = Settings(
        primary_mind_provider=MindProvider.LOCAL,
        primary_mind_api_key=None,
        inner_voice_provider=MindProvider.OPENAI,
        inner_voice_model="gpt-5-mini",
        inner_voice_api_key="inner-key",
        llm_provider=None,
        llm_api_key=None,
        llm_model=None,
    )
    assert settings.resolved_provider(MindRole.PRIMARY) == MindProvider.LOCAL
    assert settings.resolved_provider(MindRole.INNER_VOICE) == MindProvider.OPENAI
    inner = create_mind(settings, role=MindRole.INNER_VOICE)
    assert isinstance(inner, OpenAICompatibleMind)
    assert inner._model == "gpt-5-mini"
    assert inner._reasoning_effort == "minimal"
    assert inner._enable_say_tool is False


def test_openai_payload_gpt5_uses_completion_tokens() -> None:
    mind = OpenAICompatibleMind(
        model="gpt-5-mini",
        api_key="k",
        max_tokens=128,
        temperature=0.5,
        enable_say_tool=False,
    )
    payload = mind._payload([{"role": "user", "content": "hi"}], stream=False)
    assert payload["max_completion_tokens"] == 128
    assert payload["reasoning_effort"] == "minimal"
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert "tools" not in payload


def test_openai_payload_classic_uses_max_tokens() -> None:
    mind = OpenAICompatibleMind(
        model="gpt-4o-mini",
        api_key="k",
        max_tokens=256,
        temperature=0.7,
        enable_say_tool=False,
    )
    payload = mind._payload([{"role": "user", "content": "hi"}], stream=False)
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.7
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload


def test_inner_voice_reasoning_effort_override() -> None:
    settings = Settings(
        primary_mind_provider=MindProvider.LOCAL,
        primary_mind_api_key=None,
        inner_voice_provider=MindProvider.OPENAI,
        inner_voice_model="gpt-5-mini",
        inner_voice_api_key="inner-key",
        inner_voice_reasoning_effort="low",
        llm_provider=None,
        llm_api_key=None,
    )
    inner = create_mind(settings, role=MindRole.INNER_VOICE)
    assert isinstance(inner, OpenAICompatibleMind)
    assert inner._reasoning_effort == "low"


def test_resolved_proxy_fallback_and_override() -> None:
    settings = Settings(
        primary_mind_provider=MindProvider.LOCAL,
        primary_mind_proxy="http://primary-proxy:8080",
        inner_voice_proxy=None,
        llm_provider=None,
        llm_api_key=None,
        memory_enabled=False,
    )
    assert settings.resolved_proxy(MindRole.PRIMARY) == "http://primary-proxy:8080"
    assert settings.resolved_proxy(MindRole.INNER_VOICE) == "http://primary-proxy:8080"

    settings_override = Settings(
        primary_mind_provider=MindProvider.LOCAL,
        primary_mind_proxy="http://primary-proxy:8080",
        inner_voice_proxy="http://127.0.0.1:7890",
        llm_provider=None,
        llm_api_key=None,
        memory_enabled=False,
    )
    assert (
        settings_override.resolved_proxy(MindRole.INNER_VOICE)
        == "http://127.0.0.1:7890"
    )


def test_factory_passes_proxy_to_openai_mind() -> None:
    settings = Settings(
        primary_mind_provider=MindProvider.LOCAL,
        primary_mind_api_key=None,
        primary_mind_proxy=None,
        inner_voice_provider=MindProvider.OPENAI,
        inner_voice_model="gpt-5-mini",
        inner_voice_api_key="inner-key",
        inner_voice_proxy="http://127.0.0.1:7890",
        llm_provider=None,
        llm_api_key=None,
    )
    inner = create_mind(settings, role=MindRole.INNER_VOICE)
    assert isinstance(inner, OpenAICompatibleMind)
    assert inner._proxy == "http://127.0.0.1:7890"
    assert "proxy" in inner._client_kwargs()
