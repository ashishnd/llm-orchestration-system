"""Vector store wrapper.

Wraps Chroma rather than exposing it directly so the rest of the system
depends on a small, stable interface. If we ever swap to FAISS or pgvector,
the agents and retriever don't change.

Embedding strategy:
We compute embeddings explicitly in this module via an `Embedder` callable
and pass `embeddings=...` to Chroma rather than registering an embedding
function inside Chroma's collection. Reasons:
1. Chroma 1.1.13+ persists embedding functions server-side and silently
   falls back to its default ONNX model if a custom EF doesn't conform to
   the persistence protocol — tests appeared to pass while actually using
   the wrong embeddings. Bypassing Chroma's EF system is more honest.
2. Decouples the embedder from Chroma. Swapping to a local
   sentence-transformers model or to a different vector store doesn't
   require re-shimming the embedding function.
3. Tests can inject a deterministic stub embedder without conforming to
   Chroma's evolving EF protocol.

Persistence:
- PersistentClient(path=...) for production
- EphemeralClient() for tests (in-memory, isolated per fixture)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.rag.corpus import Paper

# An embedder maps a list of texts to a list of vectors. Plain Callable so
# we don't impose a class hierarchy.
Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class RetrievedChunk:
    """One result from a similarity query.

    Exposed shape kept narrow so the rest of the system doesn't depend on
    Chroma's nested-list result format directly.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def relevance_score(self) -> float:
        """Convert cosine distance (lower=better) to relevance (higher=better),
        clamped to [0, 1]. Cosine distances live in [0, 2]."""
        return max(0.0, 1.0 - (self.distance / 2.0))


class EmbeddingFunction(Protocol):
    """Anything Chroma will accept as an embedding function (kept for compat
    with Chroma's built-in OpenAIEmbeddingFunction in production paths)."""

    def __call__(self, input: list[str]) -> list[list[float]]: ...


class VectorStore:
    """Thin wrapper over a single Chroma collection.

    Embeddings are computed by `embedder` (a Callable[[list[str]], list[list[float]]])
    and passed explicitly to Chroma. No embedding function is registered on
    the collection. This decouples our embedding choice from Chroma's
    server-side EF persistence machinery and makes testing straightforward.
    """

    DEFAULT_COLLECTION_NAME = "arxiv_papers"

    def __init__(
        self,
        client: ClientAPI,
        embedder: Embedder,
        collection_name: str | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        # No embedding_function: we always pass embeddings explicitly.
        self._collection: Collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, papers: list[Paper]) -> int:
        """Upsert papers into the collection. Returns count ingested."""
        if not papers:
            return 0
        ids = [p.chunk_id for p in papers]
        documents = [p.chunk_text for p in papers]
        embeddings = self._embedder(documents)
        metadatas = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "primary_category": p.primary_category,
                "authors": "; ".join(p.authors),
                "published": p.published.isoformat(),
            }
            for p in papers
        ]
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(papers)

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k similarity search.

        `exclude_ids` lets the multi-hop retriever skip chunks it has already
        seen on prior hops, which is critical — without it, every hop would
        re-retrieve the same most-similar chunk.
        """
        query_embedding = self._embedder([query_text])

        where: dict[str, Any] | None = None
        if exclude_ids:
            # Chroma's filter expects $and only when there are multiple clauses.
            arxiv_ids = [cid.split(":", 1)[1] for cid in exclude_ids]
            if len(arxiv_ids) == 1:
                where = {"arxiv_id": {"$ne": arxiv_ids[0]}}
            else:
                where = {"$and": [{"arxiv_id": {"$ne": aid}} for aid in arxiv_ids]}

        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )

        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        chunks = []
        for chunk_id, doc, meta, dist in zip(
            ids[0], documents[0], metadatas[0], distances[0], strict=True
        ):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc,
                    metadata=dict(meta) if meta else {},
                    distance=float(dist),
                )
            )
        return chunks

    def count(self) -> int:
        return self._collection.count()


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def make_openai_embedder(api_key: str) -> Embedder:
    """OpenAI text-embedding-3-small via direct API call.

    Returns a Callable suitable for passing to VectorStore. We use the
    OpenAI SDK directly instead of Chroma's OpenAIEmbeddingFunction
    so the embedding choice is independent of Chroma's lifecycle.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [d.embedding for d in resp.data]

    return embed


def make_persistent_store(persist_dir: str, api_key: str) -> VectorStore:
    client = chromadb.PersistentClient(path=persist_dir)
    return VectorStore(client, make_openai_embedder(api_key))


def make_ephemeral_store(embedder: Embedder, collection_name: str | None = None) -> VectorStore:
    """In-memory store. Embedder is injected so tests use a deterministic
    stub instead of calling OpenAI.

    NOTE: Chroma's EphemeralClient instances share an in-memory collection
    registry across the process — two `EphemeralClient()` calls reach the
    same collections. For test isolation, pass a unique `collection_name`
    per test (e.g. `f"test_{uuid4().hex}"`).
    """
    client = chromadb.EphemeralClient()
    return VectorStore(client, embedder, collection_name=collection_name)
