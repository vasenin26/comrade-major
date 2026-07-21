from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.types import LLMProvider


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

    llm_provider: LLMProvider = Field(default=LLMProvider.LOCAL)
    llm_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    llm_device: str = Field(default="auto")
    llm_max_new_tokens: int = Field(default=256, ge=1, le=4096)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_api_key: str | None = Field(default=None)
    llm_base_url: str = Field(default="https://api.openai.com/v1")

    log_dir: str = Field(default="logs")

    @model_validator(mode="after")
    def validate_llm_provider(self) -> Self:
        if self.llm_provider == LLMProvider.OPENAI and not self.llm_api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
