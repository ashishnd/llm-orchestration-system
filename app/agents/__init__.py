"""Pipeline agents.

Each agent inherits from BaseAgent (app.agents.base) and registers its
prompts in app.agents.prompts at import time. The orchestrator dispatches
agents by name; the meta-agent reads prompts by registry key.
"""

from app.agents.base import AgentExecutionError, BaseAgent
from app.agents.compression import CompressionAgent
from app.agents.critique import CritiqueAgent
from app.agents.decomposition import DecompositionAgent
from app.agents.prompts import (
    all_keys,
    get_baseline,
    get_prompt,
    register_prompt,
    reset_to_baseline,
    set_active_prompt,
    snapshot,
)
from app.agents.retrieval import RetrievalAgent
from app.agents.synthesis import SynthesisAgent

__all__ = [
    "AgentExecutionError",
    "BaseAgent",
    "CompressionAgent",
    "CritiqueAgent",
    "DecompositionAgent",
    "RetrievalAgent",
    "SynthesisAgent",
    "all_keys",
    "get_baseline",
    "get_prompt",
    "register_prompt",
    "reset_to_baseline",
    "set_active_prompt",
    "snapshot",
]
