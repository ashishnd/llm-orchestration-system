"""arXiv corpus types and on-disk format.

The corpus is a JSONL file: one paper per line, each line a `Paper.model_dump_json()`.
Checking the corpus into the repo is intentional — it gives the grader a fixed,
reproducible knowledge base. Re-ingesting from arXiv on every run would mean
the eval scores depend on what papers happened to be available that day.

Format choice: JSONL over a single JSON array because we can stream-read it
and append-only-write it without rewriting the whole file when expanding the
corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class Paper(BaseModel):
    """One arXiv paper. Title + abstract is what we embed; the rest is metadata."""

    arxiv_id: str  # e.g. "2401.12345"
    title: str
    authors: list[str]
    abstract: str
    primary_category: str  # e.g. "cs.CL"
    published: datetime
    updated: datetime | None = None
    pdf_url: str | None = None

    @property
    def chunk_id(self) -> str:
        """Stable ID used as the Chroma chunk identifier.

        Using arxiv_id directly means re-ingesting the same paper produces
        the same vector store entry, not a duplicate. arXiv IDs are stable
        and unique across the whole corpus.
        """
        return f"arxiv:{self.arxiv_id}"

    @property
    def chunk_text(self) -> str:
        """The text we embed and retrieve against.

        Abstracts are short (~200-300 words), so chunking *within* a paper
        would fragment semantic units. We embed the whole abstract as one
        chunk, prepended with the title for context — titles often carry
        keywords (benchmark names, model architectures) that the abstract
        body doesn't repeat.
        """
        return f"{self.title}\n\n{self.abstract}"


def write_corpus(papers: list[Paper], path: Path) -> int:
    """Write papers to a JSONL file. Returns the number written.

    Idempotency: callers should de-dupe on arxiv_id before calling. We don't
    de-dupe here because the writer doesn't know about prior corpus state;
    the ingestion script does.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(paper.model_dump_json() + "\n")
    return len(papers)


def read_corpus(path: Path) -> Iterator[Paper]:
    """Stream papers from a JSONL file. Yields lazily so we don't load
    the whole corpus into memory."""
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Paper.model_validate_json(line)
            except Exception as e:
                raise ValueError(f"Corpus parse error at {path}:{line_num}: {e}") from e


def corpus_size(path: Path) -> int:
    """Count papers in the corpus without parsing them."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
