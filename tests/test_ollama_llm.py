"""Ollama adapter tests against a mocked HTTP layer (no Ollama needed)."""

import json

import httpx
import pytest

from app.answers.exceptions import LLMUnavailableError
from app.infrastructure.ollama_llm import OllamaLLM


def llm_with(handler) -> OllamaLLM:
    return OllamaLLM("http://ollama", "qwen2.5:1.5b", 10, 2048, transport=httpx.MockTransport(handler))


def test_generate_sends_grounded_chat_request():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "  Two years [1].  "}})

    assert llm_with(handler).generate("system rules", "the prompt") == "Two years [1]."
    assert seen["model"] == "qwen2.5:1.5b"
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert seen["stream"] is False and seen["options"]["temperature"] <= 0.2
    assert seen["options"]["num_predict"] == 300  # answer length is capped


def test_stream_yields_tokens_from_ndjson():
    lines = [{"message": {"content": "Zwei"}}, {"message": {"content": " Jahre"}}, {"done": True}]
    body = "\n".join(json.dumps(line) for line in lines)
    llm = llm_with(lambda request: httpx.Response(200, text=body))
    assert list(llm.stream("s", "p")) == ["Zwei", " Jahre"]


def test_missing_model_gives_actionable_error():
    llm = llm_with(lambda request: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMUnavailableError, match="ollama pull qwen2.5:1.5b"):
        llm.generate("s", "p")


def test_timeout_has_its_own_message():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMUnavailableError, match="took too long"):
        llm_with(handler).generate("s", "p")


def test_connection_error_becomes_domain_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError, match="not reachable"):
        llm_with(handler).generate("s", "p")
    with pytest.raises(LLMUnavailableError):
        list(llm_with(handler).stream("s", "p"))


@pytest.mark.parametrize(
    ("models", "expected"),
    [(["qwen2.5:1.5b"], True), (["llama3.2:3b"], False), ([], False)],
)
def test_is_available_checks_model_is_installed(models, expected):
    llm = llm_with(lambda request: httpx.Response(200, json={"models": [{"name": m} for m in models]}))
    assert llm.is_available() is expected


def test_is_available_false_when_server_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert llm_with(handler).is_available() is False
