from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.types import MindProvider, MindRole


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_min_silence_ms: int = Field(default=500, ge=0)

    whisper_model_size: str = Field(default="small")

    # Primary mind (env: PRIMARY_MIND_* with legacy LLM_* aliases via validator below)
    primary_mind_provider: MindProvider = Field(default=MindProvider.LOCAL)
    primary_mind_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    primary_mind_device: str = Field(default="auto")
    primary_mind_max_new_tokens: int = Field(default=256, ge=1, le=4096)
    primary_mind_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    primary_mind_api_key: str | None = Field(default=None)
    primary_mind_base_url: str = Field(default="https://api.openai.com/v1")
    primary_mind_system_prompt: str = Field(
        default=(
            "You are a voice agent that thinks continuously. "
            "To speak aloud, start the message with 'say ' followed by the text. "
            "Otherwise reply with silent thoughts only."
        )
    )

    # Inner voice (optional overrides; falls back to primary profile)
    inner_voice_provider: MindProvider | None = Field(default=None)
    inner_voice_model: str | None = Field(default=None)
    inner_voice_device: str | None = Field(default=None)
    inner_voice_max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    inner_voice_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    inner_voice_api_key: str | None = Field(default=None)
    inner_voice_base_url: str | None = Field(default=None)
    inner_voice_system_prompt: str = Field(
        default=(
            "You are the agent's inner voice. Briefly note priorities or corrections "
            "for the primary mind. Be concise."
        )
    )

    mind_context_trim_count: int = Field(default=2, ge=1, le=100)

    log_dir: str = Field(default="logs")

    # Legacy LLM_* env vars (still accepted)
    llm_provider: MindProvider | None = Field(default=None)
    llm_model: str | None = Field(default=None)
    llm_device: str | None = Field(default=None)
    llm_max_new_tokens: int | None = Field(default=None)
    llm_temperature: float | None = Field(default=None)
    llm_api_key: str | None = Field(default=None)
    llm_base_url: str | None = Field(default=None)

    @model_validator(mode="after")
    def apply_legacy_llm_aliases(self) -> Self:
        if self.llm_provider is not None:
            self.primary_mind_provider = self.llm_provider
        if self.llm_model is not None:
            self.primary_mind_model = self.llm_model
        if self.llm_device is not None:
            self.primary_mind_device = self.llm_device
        if self.llm_max_new_tokens is not None:
            self.primary_mind_max_new_tokens = self.llm_max_new_tokens
        if self.llm_temperature is not None:
            self.primary_mind_temperature = self.llm_temperature
        if self.llm_api_key is not None:
            self.primary_mind_api_key = self.llm_api_key
        if self.llm_base_url is not None:
            self.primary_mind_base_url = self.llm_base_url
        return self

    @model_validator(mode="after")
    def validate_mind_providers(self) -> Self:
        for role, provider, api_key in (
            (
                MindRole.PRIMARY,
                self.primary_mind_provider,
                self.primary_mind_api_key,
            ),
            (
                MindRole.INNER_VOICE,
                self.resolved_provider(MindRole.INNER_VOICE),
                self.resolved_api_key(MindRole.INNER_VOICE),
            ),
        ):
            if provider == MindProvider.OPENAI and not api_key:
                raise ValueError(
                    f"API key is required when {role.value} mind provider is openai "
                    "(PRIMARY_MIND_API_KEY / INNER_VOICE_API_KEY or LLM_API_KEY)"
                )
        return self

    def resolved_provider(self, role: MindRole) -> MindProvider:
        if role == MindRole.INNER_VOICE and self.inner_voice_provider is not None:
            return self.inner_voice_provider
        return self.primary_mind_provider

    def resolved_model(self, role: MindRole) -> str:
        if role == MindRole.INNER_VOICE and self.inner_voice_model is not None:
            return self.inner_voice_model
        return self.primary_mind_model

    def resolved_device(self, role: MindRole) -> str:
        if role == MindRole.INNER_VOICE and self.inner_voice_device is not None:
            return self.inner_voice_device
        return self.primary_mind_device

    def resolved_max_new_tokens(self, role: MindRole) -> int:
        if role == MindRole.INNER_VOICE and self.inner_voice_max_new_tokens is not None:
            return self.inner_voice_max_new_tokens
        return self.primary_mind_max_new_tokens

    def resolved_temperature(self, role: MindRole) -> float:
        if role == MindRole.INNER_VOICE and self.inner_voice_temperature is not None:
            return self.inner_voice_temperature
        return self.primary_mind_temperature

    def resolved_api_key(self, role: MindRole) -> str | None:
        if role == MindRole.INNER_VOICE and self.inner_voice_api_key is not None:
            return self.inner_voice_api_key
        return self.primary_mind_api_key

    def resolved_base_url(self, role: MindRole) -> str:
        if role == MindRole.INNER_VOICE and self.inner_voice_base_url is not None:
            return self.inner_voice_base_url
        return self.primary_mind_base_url

    def resolved_system_prompt(self, role: MindRole) -> str:
        if role == MindRole.INNER_VOICE:
            return self.inner_voice_system_prompt
        return self.primary_mind_system_prompt


@lru_cache
def get_settings() -> Settings:
    return Settings()
