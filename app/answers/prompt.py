"""Prompt construction for grounded answers.

Kept short on purpose: with the default 1.5B model, longer prompts with worked
examples gave slower and *less* accurate answers (measured on the sample PDF).
Sources are always returned separately, so answers stay traceable even when a
small model omits the [n] citations.
"""

from app.shared.models import SearchHit

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages from PDF documents.
Rules:
- Answer in the same language as the question.
- Use only facts stated in the context. Do not use outside knowledge.
- After each statement, cite the passage it comes from, like [1] or [2].
- If the context does not contain the answer, say that the documents do not contain this information.
- Be concise: at most 5 sentences."""


def build_prompt(question: str, sources: list[SearchHit]) -> str:
    """Number each passage so the model can cite it and the UI can link [n] to a page."""
    context = "\n\n".join(
        f"[{i}] ({hit.chunk.filename}, page {hit.chunk.page})\n{hit.chunk.text}"
        for i, hit in enumerate(sources, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
