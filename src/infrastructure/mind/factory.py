from config.settings import Settings
from config.types import MindProvider, MindRole
from src.application.interfaces import Mind
from src.infrastructure.mind.local.transformers import LocalTransformersMind
from src.infrastructure.mind.providers.openai_compatible import OpenAICompatibleMind


def create_mind(settings: Settings, role: MindRole = MindRole.PRIMARY) -> Mind:
    provider = settings.resolved_provider(role)
    model = settings.resolved_model(role)
    match provider:
        case MindProvider.LOCAL:
            return LocalTransformersMind(
                model_id=model,
                device=settings.resolved_device(role),
                max_new_tokens=settings.resolved_max_new_tokens(role),
                temperature=settings.resolved_temperature(role),
                enable_say_tool=role == MindRole.PRIMARY,
            )
        case MindProvider.OPENAI:
            api_key = settings.resolved_api_key(role)
            if api_key is None:
                raise ValueError(f"API key required for openai mind role={role.value}")
            reasoning_effort: str | None = None
            if role == MindRole.INNER_VOICE:
                reasoning_effort = settings.inner_voice_reasoning_effort
            return OpenAICompatibleMind(
                model=model,
                api_key=api_key,
                base_url=settings.resolved_base_url(role),
                max_tokens=settings.resolved_max_new_tokens(role),
                temperature=settings.resolved_temperature(role),
                enable_say_tool=role == MindRole.PRIMARY,
                reasoning_effort=reasoning_effort,
                proxy=settings.resolved_proxy(role),
            )
