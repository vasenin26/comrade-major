from enum import StrEnum


class MindProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class MindRole(StrEnum):
    PRIMARY = "primary"
    INNER_VOICE = "inner_voice"


# Backward-compatible alias
LLMProvider = MindProvider
