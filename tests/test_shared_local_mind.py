from unittest.mock import MagicMock, patch

from config.settings import Settings
from config.types import MindProvider, MindRole
from src.infrastructure.mind.factory import create_mind
from src.infrastructure.mind.local.transformers import (
    LocalTransformersMind,
    clear_local_mind_cache,
)


def test_local_minds_share_weights_for_same_model() -> None:
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
        memory_enabled=False,
    )

    with (
        patch(
            "src.infrastructure.mind.local.transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as tok_load,
        patch(
            "src.infrastructure.mind.local.transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as model_load,
    ):
        primary = create_mind(settings, role=MindRole.PRIMARY)
        inner = create_mind(settings, role=MindRole.INNER_VOICE)

    assert isinstance(primary, LocalTransformersMind)
    assert isinstance(inner, LocalTransformersMind)
    assert primary._model is inner._model
    assert primary._tokenizer is inner._tokenizer
    assert tok_load.call_count == 1
    assert model_load.call_count == 1
    clear_local_mind_cache()
