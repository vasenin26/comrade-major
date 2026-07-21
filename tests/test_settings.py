from config.settings import Settings, get_settings
from config.types import LLMProvider
from src.infrastructure.llm.factory import create_llm_service
from src.infrastructure.llm.providers.openai_compatible import OpenAICompatibleLLM


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.sample_rate == 16_000
    assert settings.vad_threshold == 0.5
    assert settings.whisper_model_size == "small"
    assert settings.llm_provider == LLMProvider.LOCAL
    assert settings.llm_model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_factory_creates_openai_provider() -> None:
    settings = Settings(
        llm_provider=LLMProvider.OPENAI,
        llm_api_key="test-key",
        llm_model="gpt-4o-mini",
    )
    service = create_llm_service(settings)
    assert isinstance(service, OpenAICompatibleLLM)


def test_openai_provider_requires_api_key() -> None:
    try:
        Settings(llm_provider=LLMProvider.OPENAI, llm_api_key=None)
        raise AssertionError("Expected validation error")
    except ValueError as exc:
        assert "LLM_API_KEY" in str(exc)
