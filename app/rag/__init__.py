"""arXiv corpus ingestion, Chroma vector store, and multi-hop retriever."""

from app.rag.corpus import Paper, corpus_size, read_corpus, write_corpus
from app.rag.retriever import HopResult, MultiHopResult, MultiHopRetriever
from app.rag.vector_store import (
    Embedder,
    EmbeddingFunction,
    RetrievedChunk,
    VectorStore,
    make_ephemeral_store,
    make_openai_embedder,
    make_persistent_store,
)

__all__ = [
    "Embedder",
    "EmbeddingFunction",
    "HopResult",
    "MultiHopResult",
    "MultiHopRetriever",
    "Paper",
    "RetrievedChunk",
    "VectorStore",
    "corpus_size",
    "make_ephemeral_store",
    "make_openai_embedder",
    "make_persistent_store",
    "read_corpus",
    "write_corpus",
]
