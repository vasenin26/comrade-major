import asyncio
import logging
from collections.abc import AsyncIterator
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.infrastructure.llm.messages import build_chat_messages

logger = logging.getLogger(__name__)


class LocalTransformersLLM:
    """Локальный LLM через HuggingFace transformers."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        logger.info("Loading local LLM %s (device=%s)", model_id, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_kwargs: dict[str, object] = {"torch_dtype": dtype}
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if device not in ("auto", "cpu"):
            self._model.to(device)

    def _prepare_input_ids(self, messages: list[dict[str, str]]) -> torch.Tensor:
        encoded = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded
        device = next(self._model.parameters()).device
        return input_ids.to(device)

    def _generate_sync(self, prompt: str, history: list[dict[str, str]]) -> str:
        messages = build_chat_messages(prompt, history)
        input_ids = self._prepare_input_ids(messages)
        outputs = self._model.generate(
            input_ids,
            max_new_tokens=self._max_new_tokens,
            do_sample=True,
            temperature=self._temperature,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        generated = outputs[0][input_ids.shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _stream_sync(self, prompt: str, history: list[dict[str, str]]) -> list[str]:
        messages = build_chat_messages(prompt, history)
        input_ids = self._prepare_input_ids(messages)
        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self._max_new_tokens,
            "do_sample": True,
            "temperature": self._temperature,
            "pad_token_id": self._tokenizer.pad_token_id,
            "streamer": streamer,
        }
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        chunks = list(streamer)
        thread.join()
        return chunks

    async def generate(self, prompt: str, history: list[dict[str, str]]) -> str:
        return await asyncio.to_thread(self._generate_sync, prompt, history)

    async def stream(self, prompt: str, history: list[dict[str, str]]) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def worker() -> None:
            try:
                for chunk in self._stream_sync(prompt, history):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        Thread(target=worker, daemon=True).start()
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
