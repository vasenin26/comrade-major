from collections.abc import AsyncIterator

from config.settings import Settings, get_settings
from config.types import MindProvider, MindRole
from src.infrastructure.mind.factory import create_mind
from src.infrastructure.mind.providers.openai_compatible import OpenAICompatibleMind


def test_settings_defaults() -> None:
    settings = Settings(
        llm_provider=None,
        llm_model=None,
        llm_device=None,
        llm_max_new_tokens=None,
        llm_temperature=None,
        llm_api_key=None,
        llm_base_url=None,
    )
    assert settings.sample_rate == 16_000
    assert settings.vad_threshold == 0.5
    assert settings.whisper_model_size == "small"
    assert settings.primary_mind_provider == MindProvider.LOCAL
    assert settings.primary_mind_model == "Qwen/Qwen2.5-0.5B-Instruct"


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
        )
        raise AssertionError("Expected validation error")
    except ValueError as exc:
        assert "API key" in str(exc)


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
        llm_provider=None,
        llm_model=None,
    )
    assert settings.resolved_model(MindRole.INNER_VOICE) == "primary-model"
    assert settings.resolved_provider(MindRole.INNER_VOICE) == MindProvider.LOCAL
