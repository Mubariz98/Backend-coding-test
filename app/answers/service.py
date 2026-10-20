"""Use case: answer a question from the documents (retrieve -> prompt -> generate)."""

from app.answers.exceptions import LLMUnavailableError
from app.answers.models import Answer, StreamingAnswer
from app.answers.ports import LLM
from app.answers.prompt import SYSTEM_PROMPT, build_prompt
from app.search.service import SearchService
from app.shared.models import SearchHit

NO_CONTEXT_ANSWER = "I could not find relevant information about this in the uploaded documents."


class AnswerService:
    def __init__(self, search: SearchService, llm: LLM, top_k: int, min_score: float):
        self._search = search
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score

    def answer(self, question: str, document_id: str | None = None) -> Answer:
        sources = self._retrieve(question, document_id)
        if not sources:
            return Answer(NO_CONTEXT_ANSWER, [])
        return Answer(self._llm.generate(SYSTEM_PROMPT, build_prompt(question, sources)), sources)

    def stream_answer(self, question: str, document_id: str | None = None) -> StreamingAnswer:
        sources = self._retrieve(question, document_id)
        if not sources:
            return StreamingAnswer([], iter([NO_CONTEXT_ANSWER]))
        # Fail before the response starts; once streaming, the HTTP status can't change.
        if not self._llm.is_available():
            raise LLMUnavailableError(f"Language model '{self._llm.model}' is not reachable.")
        return StreamingAnswer(sources, self._llm.stream(SYSTEM_PROMPT, build_prompt(question, sources)))

    def _retrieve(self, question: str, document_id: str | None) -> list[SearchHit]:
        # No relevant passages -> skip the LLM entirely: faster, and nothing to hallucinate from.
        return self._search.search(question, self._top_k, document_id, self._min_score)
