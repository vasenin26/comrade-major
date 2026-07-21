SAY_TOOL = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak text aloud to the user via TTS.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak.",
                }
            },
            "required": ["text"],
        },
    },
}


def format_say_message(text: str) -> str:
    return f"say {text.strip()}"
