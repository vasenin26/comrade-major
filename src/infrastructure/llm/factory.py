from config.settings import Settings
from src.application.interfaces import LLMService
from src.infrastructure.llm.local.transformers import LocalTransformersLLM
from src.infrastructure.llm.providers.openai_compatible import OpenAICompatibleLLM
from config.types import LLMProvider


def create_llm_service(settings: Settings) -> LLMService:
    match settings.llm_provider:
        case LLMProvider.LOCAL:
            return LocalTransformersLLM(
                model_id=settings.llm_model,
                device=settings.llm_device,
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature,
            )
        case LLMProvider.OPENAI:
            if settings.llm_api_key is None:
                raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
            return OpenAICompatibleLLM(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                max_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature,
            )
