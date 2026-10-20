"""Remove page headers/footers that repeat on most pages.

Repeated boilerplate (company name, dates, "page 6") ends up in every chunk
and makes all chunks look alike to the embedding model.
"""

import re
from collections import Counter


def _signature(line: str) -> str:
    """Normalise a line so headers that differ only by page number match."""
    return re.sub(r"\d+", "#", " ".join(line.split()))


def remove_repeated_lines(pages: list[str], min_ratio: float = 0.5) -> list[str]:
    """Drop lines that appear on at least `min_ratio` of pages."""
    if len(pages) < 3:  # too few pages to tell boilerplate from content
        return pages

    # Count each line once per page.
    counts = Counter(sig for page in pages for sig in {_signature(l) for l in page.splitlines()})
    threshold = len(pages) * min_ratio
    repeated = {sig for sig, n in counts.items() if n >= threshold and sig}

    return [
        "\n".join(l for l in page.splitlines() if _signature(l) not in repeated)
        for page in pages
    ]
