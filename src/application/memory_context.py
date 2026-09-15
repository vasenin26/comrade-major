from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole


def inject_memory_passages(
    history: list[dict[str, str]],
    passages: list[str],
) -> list[dict[str, str]]:
    """Insert a [MEMORY] system block after leading system extras."""
    cleaned = [p.strip() for p in passages if p.strip()]
    if not cleaned:
        return history
    block = {"role": "system", "content": "[MEMORY]\n" + "\n---\n".join(cleaned)}
    if not history:
        return [block]
    if history[0].get("role") != "system":
        return [block, *history]
    insert_at = 1
    while insert_at < len(history) and history[insert_at].get("role") == "system":
        insert_at += 1
    return [*history[:insert_at], block, *history[insert_at:]]


async def build_recall_query(
    store: ConversationStore,
    *,
    chat_tail: int = 6,
    thought_tail: int = 3,
) -> str:
    """Build a short retrieval query from recent chat + private thoughts."""
    messages = await store.snapshot()
    thoughts = await store.private_thoughts()
    parts: list[str] = []
    for message in messages[-chat_tail:]:
        if message.role == MessageRole.SYSTEM:
            continue
        parts.append(f"{message.role.value}: {message.content}")
    for thought in thoughts[-thought_tail:]:
        parts.append(f"thought: {thought}")
    return "\n".join(parts)
