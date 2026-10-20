"""Domain errors for the answers module. Mapped to HTTP codes in core/error_handlers.py."""


class LLMUnavailableError(Exception):
    """The language model cannot be reached or is not installed."""
