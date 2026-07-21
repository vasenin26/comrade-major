def build_chat_messages(prompt: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [*history, {"role": "user", "content": prompt}]
