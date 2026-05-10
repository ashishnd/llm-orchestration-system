"""Tests for the RAG layer.

Three groups:
- Corpus IO: roundtrip JSONL, malformed input handling, idempotent merge
- Vector store: ingest/query/dedupe with a deterministic stub embedding fn
- Multi-hop retriever: hop ordering, cross-hop deduplication, result accessors

Embedding stub: we don't call OpenAI from tests. We use a tiny deterministic
embedding function (hash of the text bytes -> fixed-length float vector) so
tests are hermetic and reproducible. Quality is irrelevant; what we test is
mechanics (IDs, ordering, dedupe), not retrieval relevance.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.rag import (
    MultiHopRetriever,
    Paper,
    RetrievedChunk,
    VectorStore,
    corpus_size,
    make_ephemeral_store,
    read_corpus,
    write_corpus,
)

# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


def make_paper(arxiv_id: str, title: str, abstract: str = "stub abstract") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Author A", "Author B"],
        abstract=abstract,
        primary_category="cs.CL",
        published=datetime(2024, 1, 1, tzinfo=UTC),
    )


def stub_embedder(texts: list[str], dim: int = 16) -> list[list[float]]:
    """Deterministic 16-dim embedding from a hash of each input. NOT semantic.

    The tests here verify mechanics (IDs, ordering, dedupe) not retrieval
    quality, so a hash-based embedding is fine and removes the
    network/cost/model-download dependency. Tiny dim for speed.
    """
    out: list[list[float]] = []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(h[i % len(h)] / 127.5) - 1.0 for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


@pytest.fixture
def store() -> VectorStore:
    """Fresh isolated store per test.

    Chroma's EphemeralClient instances share an in-memory collection
    registry across the process, so creating a 'fresh' client doesn't
    actually isolate state. We use a unique collection name per test
    to get true isolation.
    """
    from uuid import uuid4

    return make_ephemeral_store(stub_embedder, collection_name=f"test_{uuid4().hex}")


# --------------------------------------------------------------------------- #
# Corpus IO
# --------------------------------------------------------------------------- #


class TestCorpusIO:
    def test_roundtrip_one_paper(self, tmp_path: Path):
        p = make_paper("2401.00001", "Title One")
        path = tmp_path / "corpus.jsonl"
        write_corpus([p], path)
        loaded = list(read_corpus(path))
        assert len(loaded) == 1
        assert loaded[0].arxiv_id == "2401.00001"
        assert loaded[0].title == "Title One"

    def test_roundtrip_many_papers_preserves_order(self, tmp_path: Path):
        papers = [make_paper(f"2401.{i:05d}", f"Title {i}") for i in range(10)]
        path = tmp_path / "corpus.jsonl"
        write_corpus(papers, path)
        loaded = list(read_corpus(path))
        assert [p.arxiv_id for p in loaded] == [p.arxiv_id for p in papers]

    def test_corpus_size_without_parsing(self, tmp_path: Path):
        papers = [make_paper(f"2401.{i:05d}", f"T{i}") for i in range(7)]
        path = tmp_path / "corpus.jsonl"
        write_corpus(papers, path)
        assert corpus_size(path) == 7

    def test_corpus_size_missing_file_returns_zero(self, tmp_path: Path):
        assert corpus_size(tmp_path / "does_not_exist.jsonl") == 0

    def test_blank_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            make_paper("2401.00001", "Real").model_dump_json() + "\n\n\n",
            encoding="utf-8",
        )
        loaded = list(read_corpus(path))
        assert len(loaded) == 1

    def test_malformed_line_raises_with_location(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        good = make_paper("2401.00001", "Real").model_dump_json()
        path.write_text(f"{good}\nNOT JSON AT ALL\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 2|:2:"):
            list(read_corpus(path))

    def test_paper_chunk_text_includes_title(self):
        p = make_paper("2401.00001", "My Title", abstract="The abstract.")
        assert "My Title" in p.chunk_text
        assert "The abstract." in p.chunk_text

    def test_paper_chunk_id_uses_arxiv_id(self):
        p = make_paper("2401.00001", "T")
        assert p.chunk_id == "arxiv:2401.00001"


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #


class TestVectorStore:
    def test_ingest_returns_count(self, store: VectorStore):
        papers = [make_paper(f"2401.{i:05d}", f"T{i}") for i in range(5)]
        n = store.ingest(papers)
        assert n == 5
        assert store.count() == 5

    def test_ingest_is_idempotent(self, store: VectorStore):
        """Re-ingesting the same papers should not produce duplicates (upsert)."""
        papers = [make_paper(f"2401.{i:05d}", f"T{i}") for i in range(3)]
        store.ingest(papers)
        store.ingest(papers)
        assert store.count() == 3

    def test_ingest_empty_list_is_safe(self, store: VectorStore):
        assert store.ingest([]) == 0
        assert store.count() == 0

    def test_query_returns_top_k(self, store: VectorStore):
        papers = [make_paper(f"2401.{i:05d}", f"Title {i}") for i in range(10)]
        store.ingest(papers)
        results = store.query("anything", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_query_results_carry_metadata(self, store: VectorStore):
        store.ingest([make_paper("2401.00001", "Some Title")])
        results = store.query("anything", top_k=1)
        assert results[0].metadata["arxiv_id"] == "2401.00001"
        assert results[0].metadata["title"] == "Some Title"
        assert results[0].metadata["primary_category"] == "cs.CL"

    def test_query_relevance_score_is_in_unit_interval(self, store: VectorStore):
        store.ingest([make_paper(f"2401.{i:05d}", f"T{i}") for i in range(5)])
        results = store.query("query text", top_k=5)
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0

    def test_query_exclude_ids_filters_results(self, store: VectorStore):
        papers = [make_paper(f"2401.{i:05d}", f"T{i}") for i in range(5)]
        store.ingest(papers)
        first_pass = store.query("q", top_k=5)
        first_id = first_pass[0].chunk_id

        second_pass = store.query("q", top_k=5, exclude_ids=[first_id])
        returned_ids = {r.chunk_id for r in second_pass}
        assert first_id not in returned_ids


# --------------------------------------------------------------------------- #
# Multi-hop retriever
# --------------------------------------------------------------------------- #


class TestMultiHopRetriever:
    """The retriever orchestrates hops; we mock the store to make assertions
    about the hop sequence and dedupe behavior independent of similarity
    quirks."""

    def _store_returning(self, hop_results: list[list[RetrievedChunk]]) -> MagicMock:
        """Build a mock store that returns the given chunks per call."""
        m = MagicMock(spec=VectorStore)
        m.query.side_effect = hop_results
        return m

    def _chunk(self, chunk_id: str, distance: float = 0.5) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=f"text for {chunk_id}",
            metadata={"arxiv_id": chunk_id.split(":", 1)[-1]},
            distance=distance,
        )

    def test_single_hop_records_query_and_chunks(self):
        store = self._store_returning([[self._chunk("arxiv:1"), self._chunk("arxiv:2")]])
        r = MultiHopRetriever(store=store, top_k_per_hop=2)

        hop = r.hop("first query")
        assert hop.hop_index == 0
        assert hop.query == "first query"
        assert [c.chunk_id for c in hop.chunks] == ["arxiv:1", "arxiv:2"]

    def test_subsequent_hops_exclude_prior_chunks(self):
        store = self._store_returning(
            [
                [self._chunk("arxiv:1"), self._chunk("arxiv:2")],
                [self._chunk("arxiv:3")],
            ]
        )
        r = MultiHopRetriever(store=store, top_k_per_hop=2)
        r.hop("q1")
        r.hop("q2")

        # The second call to store.query must have excluded chunks 1 and 2
        second_call_kwargs = store.query.call_args_list[1].kwargs
        excluded = set(second_call_kwargs["exclude_ids"])
        assert excluded == {"arxiv:1", "arxiv:2"}

    def test_first_hop_passes_no_exclusions(self):
        store = self._store_returning([[self._chunk("arxiv:1")]])
        r = MultiHopRetriever(store=store, top_k_per_hop=1)
        r.hop("q1")
        first_call_kwargs = store.query.call_args_list[0].kwargs
        assert first_call_kwargs["exclude_ids"] is None

    def test_all_chunks_deduplicates_across_hops(self):
        """Even if the store somehow returns a chunk we've seen, all_chunks
        deduplicates. This is defense-in-depth — exclude_ids should prevent
        it, but the property handles the case if it slips through."""
        store = self._store_returning(
            [
                [self._chunk("arxiv:1")],
                [self._chunk("arxiv:1"), self._chunk("arxiv:2")],
            ]
        )
        r = MultiHopRetriever(store=store, top_k_per_hop=2)
        r.hop("q1")
        r.hop("q2")
        result = r.result()
        ids = [c.chunk_id for c in result.all_chunks]
        assert ids == ["arxiv:1", "arxiv:2"]

    def test_chunk_to_hop_mapping(self):
        store = self._store_returning(
            [
                [self._chunk("arxiv:1")],
                [self._chunk("arxiv:2"), self._chunk("arxiv:3")],
            ]
        )
        r = MultiHopRetriever(store=store, top_k_per_hop=2)
        r.hop("q1")
        r.hop("q2")
        mapping = r.result().chunk_to_hop
        assert mapping == {"arxiv:1": 0, "arxiv:2": 1, "arxiv:3": 1}

    def test_result_includes_all_hops_in_order(self):
        store = self._store_returning([[self._chunk("arxiv:1")], [self._chunk("arxiv:2")]])
        r = MultiHopRetriever(store=store, top_k_per_hop=1)
        r.hop("q_a")
        r.hop("q_b")
        result = r.result()
        assert [h.query for h in result.hops] == ["q_a", "q_b"]
        assert [h.hop_index for h in result.hops] == [0, 1]
