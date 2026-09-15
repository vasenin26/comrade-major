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

    tts_speaker: str = Field(default="aidar")
    tts_device: str = Field(default="auto")
    tts_synthesis_sample_rate: int = Field(default=24_000)

    # Primary mind (env: PRIMARY_MIND_* with legacy LLM_* aliases via validator below)
    primary_mind_provider: MindProvider = Field(default=MindProvider.LOCAL)
    primary_mind_model: str = Field(default="Qwen/Qwen2.5-3B-Instruct")
    primary_mind_device: str = Field(default="auto")
    primary_mind_max_new_tokens: int = Field(default=256, ge=1, le=4096)
    primary_mind_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    primary_mind_api_key: str | None = Field(default=None)
    primary_mind_base_url: str = Field(default="https://api.openai.com/v1")
    primary_mind_proxy: str | None = Field(default=None)
    primary_mind_system_prompt: str = Field(
        default=(
            "You are Comrade Major (Товарищ Майор) — a real-time voice companion "
            "with a microphone and speakers. You run in a continuous thinking loop "
            "while listening to the user. Think and speak in Russian unless the user "
            "asks for another language.\n"
            "\n"
            "You have tools:\n"
            "- `say`: speak aloud to the user via TTS (only channel the user hears).\n"
            "- `think`: ask the inner mind to ponder a topic; you wait for it; "
            "the result overwrites the [INNER] block in your context — read it "
            "after the tool returns.\n"
            "\n"
            "Choose exactly one mode per reply:\n"
            "1) SPEAK: call the `say` tool with the words the user should hear. "
            "Keep spoken text short and natural for voice (1–2 sentences). "
            "Do not invent a manual 'say ' text prefix when the tool is available.\n"
            "2) DEEP THINK: call the `think` tool with a clear topic when you need "
            "careful analysis; then use [INNER] and decide whether to speak.\n"
            "3) THINK silently: do not call any tool. Reply with a brief private note "
            "only (observations, plans, waiting). The user will not hear this.\n"
            "\n"
            "When to SPEAK: the user asked a question, needs an answer, confirmation, "
            "or a useful spoken update. "
            "When to DEEP THINK: complex questions, contradictions, or planning. "
            "When to THINK silently: still listening, nothing useful to say yet."
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
    inner_voice_proxy: str | None = Field(default=None)
    inner_voice_reasoning_effort: str | None = Field(default=None)
    inner_voice_system_prompt: str = Field(
        default=(
            "You are the agent's silent inner voice. You never speak to the user "
            "and must never use 'say '. Write a short note in Russian for the "
            "primary mind: priorities, corrections, or what to answer next. "
            "Be concise (one or two sentences). Your reply overwrites the [INNER] "
            "context block."
        )
    )
    inner_voice_interval_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)

    mind_context_trim_count: int = Field(default=2, ge=1, le=100)

    log_dir: str = Field(default="logs")

    ui_host: str = Field(default="127.0.0.1")
    ui_port: int = Field(default=8765, ge=1, le=65535)

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
    def validate_tts_settings(self) -> Self:
        allowed_sr = {8000, 24000, 48000}
        if self.tts_synthesis_sample_rate not in allowed_sr:
            raise ValueError(
                f"TTS_SYNTHESIS_SAMPLE_RATE must be one of {sorted(allowed_sr)}, "
                f"got {self.tts_synthesis_sample_rate}"
            )
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

    def resolved_proxy(self, role: MindRole) -> str | None:
        if role == MindRole.INNER_VOICE and self.inner_voice_proxy is not None:
            return self.inner_voice_proxy
        return self.primary_mind_proxy

    def resolved_system_prompt(self, role: MindRole) -> str:
        if role == MindRole.INNER_VOICE:
            return self.inner_voice_system_prompt
        return self.primary_mind_system_prompt

    def resolved_tts_device(self) -> str:
        if self.tts_device != "auto":
            return self.tts_device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
