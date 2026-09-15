import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.infrastructure.mind.tools import PRIMARY_TOOLS_FOR_CHAT, normalize_tool_reply

logger = logging.getLogger(__name__)

# One HF load per (model_id, device); primary + inner voice share weights.
_LOADED: dict[tuple[str, str], tuple[object, object]] = {}
_LOADED_LOCK = threading.Lock()


def _load_tokenizer_and_model(model_id: str, device: str) -> tuple[object, object]:
    key = (model_id, device)
    with _LOADED_LOCK:
        cached = _LOADED.get(key)
        if cached is not None:
            logger.info("Reusing local mind weights %s (device=%s)", model_id, device)
            return cached

        logger.info("Loading local mind %s (device=%s)", model_id, device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_kwargs: dict[str, object] = {"torch_dtype": dtype}
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if device not in ("auto", "cpu"):
            model.to(device)

        pair = (tokenizer, model)
        _LOADED[key] = pair
        return pair


class LocalTransformersMind:
    """Local Mind adapter via HuggingFace transformers.

    Weights are shared across instances with the same model_id+device.
    Inference is serialized with a process-wide lock so parallel loops
    (primary / inner voice) never call generate concurrently on one GPU model.
    """

    _infer_lock = threading.Lock()

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        enable_say_tool: bool = False,
    ) -> None:
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._enable_say_tool = enable_say_tool
        self._tokenizer, self._model = _load_tokenizer_and_model(model_id, device)

    def _prepare_input_ids(self, messages: list[dict[str, str]]) -> torch.Tensor:
        template_kwargs: dict[str, object] = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "tokenize": True,
        }
        if self._enable_say_tool:
            template_kwargs["tools"] = PRIMARY_TOOLS_FOR_CHAT

        try:
            encoded = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                messages,
                **template_kwargs,
            )
        except TypeError:
            # Older chat templates may not accept tools=
            template_kwargs.pop("tools", None)
            encoded = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                messages,
                **template_kwargs,
            )
        # transformers>=4.x may return BatchEncoding (Mapping), not a bare Tensor
        # and not a plain dict — isinstance(..., dict) is False for BatchEncoding.
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
        else:
            input_ids = encoded["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids)
        device = next(self._model.parameters()).device  # type: ignore[union-attr]
        return input_ids.to(device)

    def _think_sync(self, history: list[dict[str, str]]) -> str:
        with self._infer_lock:
            input_ids = self._prepare_input_ids(history)
            outputs = self._model.generate(  # type: ignore[union-attr]
                input_ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                pad_token_id=self._tokenizer.pad_token_id,  # type: ignore[union-attr]
            )
            generated = outputs[0][input_ids.shape[-1] :]
            # Keep special tokens so <tool_call>…</tool_call> survives decode.
            raw = self._tokenizer.decode(  # type: ignore[union-attr]
                generated, skip_special_tokens=False
            ).strip()
            # Strip leftover chat end markers if present
            for marker in ("<|im_end|>", "<|endoftext|>"):
                raw = raw.replace(marker, "")
            raw = raw.strip()
            if self._enable_say_tool:
                return normalize_tool_reply(raw)
            return normalize_tool_reply(raw) if "<tool_call>" in raw.lower() else raw

    def _stream_sync(self, history: list[dict[str, str]]) -> list[str]:
        with self._infer_lock:
            input_ids = self._prepare_input_ids(history)
            streamer = TextIteratorStreamer(
                self._tokenizer,  # type: ignore[arg-type]
                skip_special_tokens=not self._enable_say_tool,
            )
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": self._max_new_tokens,
                "do_sample": True,
                "temperature": self._temperature,
                "pad_token_id": self._tokenizer.pad_token_id,  # type: ignore[union-attr]
                "streamer": streamer,
            }
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)  # type: ignore[union-attr]
            thread.start()
            chunks = list(streamer)
            thread.join()
            return chunks

    async def think(self, history: list[dict[str, str]]) -> str:
        return await asyncio.to_thread(self._think_sync, history)

    async def stream(self, history: list[dict[str, str]]) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def worker() -> None:
            try:
                joined = "".join(self._stream_sync(history))
                if self._enable_say_tool or "<tool_call>" in joined.lower():
                    joined = normalize_tool_reply(joined)
                loop.call_soon_threadsafe(queue.put_nowait, joined)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        Thread(target=worker, daemon=True).start()
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


def clear_local_mind_cache() -> None:
    """Test helper: drop cached HF weights."""
    with _LOADED_LOCK:
        _LOADED.clear()
