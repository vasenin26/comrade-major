from dataclasses import dataclass, field


@dataclass
class ChatSession:
    history: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self.history.clear()
