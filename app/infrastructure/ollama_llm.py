"""Ollama implementation of the `LLM` port.

Ollama runs open models (Qwen, Llama, Mistral, ...) locally and serves them over
a small HTTP API, so no SDK is needed: plain httpx calls to /api/chat.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from app.answers.exceptions import LLMUnavailableError


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float,
        context_tokens: int,
        max_tokens: int = 300,
        transport: httpx.BaseTransport | None = None,  # tests inject a mock transport
    ):
        self.model = model
        self._context_tokens = context_tokens
        self._max_tokens = max_tokens
        # Small CPUs need a long read timeout; connecting should be quick.
        self._client = httpx.Client(
            base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=3.0), transport=transport
        )

    def generate(self, system: str, prompt: str) -> str:
        with _translate_errors(self.model):
            res = self._client.post("/api/chat", json=self._payload(system, prompt, stream=False))
            res.raise_for_status()
            return res.json()["message"]["content"].strip()

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        with _translate_errors(self.model):
            with self._client.stream("POST", "/api/chat", json=self._payload(system, prompt, stream=True)) as res:
                res.raise_for_status()
                # Ollama streams one JSON object per line.
                for line in res.iter_lines():
                    if line and (text := json.loads(line).get("message", {}).get("content")):
                        yield text

    def is_available(self) -> bool:
        try:
            res = self._client.get("/api/tags", timeout=2.0)
            res.raise_for_status()
        except httpx.HTTPError:
            return False
        installed = {m["name"] for m in res.json().get("models", [])}
        return self.model in installed or f"{self.model}:latest" in installed

    def _payload(self, system: str, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": stream,
            "keep_alive": "30m",  # keep the model in RAM between questions
            # num_predict caps the answer length: small models can otherwise ramble until the timeout.
            "options": {"temperature": 0.1, "num_ctx": self._context_tokens, "num_predict": self._max_tokens},
        }


@contextmanager
def _translate_errors(model: str) -> Iterator[None]:
    """Turn transport errors into the domain's LLMUnavailableError."""
    try:
        yield
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise LLMUnavailableError(f"Model '{model}' is not installed. Run: ollama pull {model}") from exc
        raise LLMUnavailableError(f"Ollama returned an error: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError("The language model took too long to answer.") from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailableError(f"Ollama is not reachable: {exc}") from exc
