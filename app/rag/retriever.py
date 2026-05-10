"""Multi-hop retrieval primitive.

This module is intentionally *not* the retrieval agent. The agent (in
app/agents/retrieval.py) decides what to query at each hop using an LLM;
this module just executes the hops with proper state management.

Why split:
- The agent is LLM-bound and slow. The primitive is fast and deterministic.
- The agent's LLM-driven query rewriting can be tested separately from the
  retrieval mechanics (deduplication, hop tracking, score aggregation).
- Tests for the agent can mock this primitive cleanly.

The brief requires `>= 2` retrieved chunks across hops before forming an
answer, with citations linking which chunk supported which part of the
answer. Deduplication across hops is essential — without it, every hop
retrieves the same most-similar chunk and "multi-hop" is a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.vector_store import RetrievedChunk, VectorStore


@dataclass
class HopResult:
    """One executed retrieval hop. Mirrors RetrievalHop in the schema but
    lives in the rag layer; the agent translates between them."""

    hop_index: int
    query: str
    chunks: list[RetrievedChunk]


@dataclass
class MultiHopResult:
    """Aggregate of all hops in one retrieval session."""

    hops: list[HopResult]

    @property
    def all_chunks(self) -> list[RetrievedChunk]:
        """Flat list of every chunk retrieved, in hop+rank order, deduplicated."""
        seen: set[str] = set()
        out: list[RetrievedChunk] = []
        for hop in self.hops:
            for chunk in hop.chunks:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    out.append(chunk)
        return out

    @property
    def chunk_to_hop(self) -> dict[str, int]:
        """Mapping from chunk_id to the hop_index where it was first seen.
        Useful for citation rendering."""
        out: dict[str, int] = {}
        for hop in self.hops:
            for chunk in hop.chunks:
                if chunk.chunk_id not in out:
                    out[chunk.chunk_id] = hop.hop_index
        return out


@dataclass
class MultiHopRetriever:
    """Executes a sequence of retrieval queries with cross-hop deduplication.

    Stateful within a single retrieval session — `seen_chunk_ids` accumulates
    so each hop excludes chunks from prior hops. This is the operationalization
    of "multi-hop": each hop sees genuinely different chunks.

    Use one instance per query session. Reset state by constructing a new one.
    """

    store: VectorStore
    top_k_per_hop: int = 3
    seen_chunk_ids: set[str] = field(default_factory=set)
    hops: list[HopResult] = field(default_factory=list)

    def hop(self, query: str) -> HopResult:
        """Execute one retrieval hop, excluding previously-seen chunks.

        Returns the HopResult and also appends it to self.hops so the caller
        can build a MultiHopResult from the accumulated state.
        """
        chunks = self.store.query(
            query,
            top_k=self.top_k_per_hop,
            exclude_ids=list(self.seen_chunk_ids) if self.seen_chunk_ids else None,
        )
        for c in chunks:
            self.seen_chunk_ids.add(c.chunk_id)
        result = HopResult(hop_index=len(self.hops), query=query, chunks=chunks)
        self.hops.append(result)
        return result

    def result(self) -> MultiHopResult:
        return MultiHopResult(hops=list(self.hops))
