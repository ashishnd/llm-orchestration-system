"""Dependency container for the API.

Holds the live LLM client, vector store, event bus, and a sessionmaker.
We keep them on a single object so tests can substitute fakes by setting
attributes directly, and so FastAPI doesn't need a deep dependency-injection
graph for what is essentially application-level singletons.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import LLMClient
from app.rag import VectorStore
from app.streaming import EventBus


@dataclass
class AppContainer:
    llm: LLMClient
    vector_store: VectorStore
    event_bus: EventBus
    sessionmaker: async_sessionmaker[AsyncSession]


_CONTAINER: AppContainer | None = None


def set_container(container: AppContainer) -> None:
    """Called once at FastAPI startup. Tests call this with their own fakes."""
    global _CONTAINER
    _CONTAINER = container


def get_container() -> AppContainer:
    if _CONTAINER is None:
        raise RuntimeError("AppContainer not initialized. Call set_container() at startup.")
    return _CONTAINER
