"""Interface for a text-generation model. Any local or hosted LLM can implement it."""

from collections.abc import Iterator
from typing import Protocol


class LLM(Protocol):
    model: str

    def generate(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield the answer piece by piece as it is generated."""
        ...

    def is_available(self) -> bool:
        """Cheap check that the model can be reached (used before streaming and by /health)."""
        ...
