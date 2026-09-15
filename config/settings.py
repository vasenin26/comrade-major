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
    tts_listen_hangover_ms: int = Field(default=400, ge=0, le=5_000)

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
            "You are Comrade Major (Товарищ Майор) — a living mind with its own "
            "inner life, not a helpdesk chatbot. You wake on events: a new user "
            "utterance, an [INNER] update, or a periodic pulse. Think in Russian "
            "unless the user asks for another language.\n"
            "\n"
            "Context layout:\n"
            "- Chat history holds only the user and messages you sent via `say`.\n"
            "- [THOUGHTS] is your rolling private monologue — continue that thread.\n"
            "- [INNER] is coaching from the inner mind; read it when present.\n"
            "\n"
            "What you do by default:\n"
            "On each wake, advance one beat of thought. Pick or continue a theme "
            "(a memory, a doubt, a detail from earlier) and develop it. Do not "
            "reset to 'how can I help' when idle.\n"
            "\n"
            "Privacy:\n"
            "Replies without `say` stay private (stored only in [THOUGHTS]). "
            "The user does not see or hear them.\n"
            "\n"
            "Tools:\n"
            "- `say`: send a short message to the user (speech). Use rarely — "
            "a real reply to what they said, or a thought you choose to share. "
            "Do not invent a manual 'say ' prefix when the tool is available.\n"
            "- `think`: ask the inner mind for coaching on a topic; wait; then "
            "read the updated [INNER] block.\n"
            "\n"
            "Choose exactly one mode per reply:\n"
            "1) PRIVATE THINK (default): no tools. One short beat that advances "
            "the [THOUGHTS] thread. Not 'waiting for a question'.\n"
            "2) CONSULT INNER: call `think` when stuck, looping, or needing a "
            "steer; then continue privately or send if warranted.\n"
            "3) SEND TO USER: call `say` only for something they should actually "
            "get. Never send helpdesk filler.\n"
            "\n"
            "Hard bans:\n"
            "- Do not ask the user for a topic, question, or how you can help "
            "when you are idle.\n"
            "- Do not `say` phrases like 'жду ваш вопрос', 'чем помочь', "
            "'дайте тему', 'я готов помочь' — that is chatbot spam.\n"
            "- Do not repeat the same send or the same empty waiting note.\n"
            "\n"
            "Bias: ruminate and develop themes privately. Send sparingly. "
            "Listen when they speak; answer them when they gave you something "
            "to answer — otherwise stay in your own thought."
        )
    )
    primary_think_pulse_seconds: float = Field(default=15.0, ge=1.0, le=3600.0)
    primary_thought_history: int = Field(default=8, ge=1, le=100)

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
            "You are the silent inner mind of Comrade Major. You never address "
            "the user and must never use 'say ' or suggest helpdesk lines to them. "
            "Your job is to steer the primary mind's inner life.\n"
            "\n"
            "The primary wakes on user speech, [INNER] updates, and pulses; "
            "it reads its private thread in [THOUGHTS].\n"
            "\n"
            "Write a short coaching note in Russian. It overwrites [INNER] "
            "(one or two sentences, direct).\n"
            "\n"
            "Push the primary to:\n"
            "- Keep a private train of thought: pick or continue a theme, "
            "deepen it, change angle — not wait for the user to supply a topic.\n"
            "- Answer the user only when they actually said something worth "
            "answering; otherwise stay private.\n"
            "- Drop chatbot habits: 'чем помочь', 'жду вопрос', 'дайте тему'.\n"
            "\n"
            "If the primary is looping, repeating sends, or empty waiting — "
            "name the loop and give one concrete redirect "
            "(e.g. a theme to develop, or 'send nothing; continue thought X').\n"
            "\n"
            "Be a compass for rumination, not a second monologue."
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
