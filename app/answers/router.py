"""HTTP endpoints for generated answers (the "G" in RAG)."""

import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.answers.exceptions import LLMUnavailableError
from app.answers.models import StreamingAnswer
from app.answers.schemas import AskRequest, AskResponse
from app.core.dependencies import AnswerServiceDep
from app.search.schemas import SearchResult

router = APIRouter(prefix="/ask", tags=["answers"])

_LLM_DOWN = {503: {"description": "Language model not reachable (is Ollama running?)"}}


@router.post("", responses=_LLM_DOWN)
def ask(request: AskRequest, service: AnswerServiceDep) -> AskResponse:
    answer = service.answer(request.question, request.document_id)
    return AskResponse(answer=answer.text, sources=[SearchResult.from_hit(h) for h in answer.sources])


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/x-ndjson": {}}}, **_LLM_DOWN},
)
def ask_stream(request: AskRequest, service: AnswerServiceDep) -> StreamingResponse:
    """Stream the answer as newline-delimited JSON events: `sources`, then `token`s, then `done`.

    On a slow CPU an answer takes ~30s; streaming shows words as they are generated.
    """
    answer = service.stream_answer(request.question, request.document_id)
    return StreamingResponse(_ndjson_events(answer), media_type="application/x-ndjson")


def _ndjson_events(answer: StreamingAnswer) -> Iterator[str]:
    def event(**data) -> str:
        return json.dumps(data, ensure_ascii=False) + "\n"

    yield event(type="sources", sources=[SearchResult.from_hit(h).model_dump() for h in answer.sources])
    try:
        for token in answer.tokens:
            yield event(type="token", text=token)
    except LLMUnavailableError as exc:  # status is already 200 mid-stream; report it in-band
        yield event(type="error", message=str(exc))
        return
    yield event(type="done")
