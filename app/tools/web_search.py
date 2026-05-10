"""Web search stub.

The brief specifies a stub returning structured results with source URLs
and relevance scores — not a real web search. We return canned results
keyed off the query so eval cases can be deterministic.

Real production replacement would be a Tavily/Serper/Bing call here;
the contract (status, payload shape) stays the same.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.tools.base import Tool, ToolResult, ToolStatus

# Canned corpus for stub responses. Eval cases reference these.
_CANNED_RESULTS: dict[str, list[dict[str, Any]]] = {
    "default": [
        {
            "url": "https://arxiv.org/abs/1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "snippet": "We introduce a new language representation model called BERT...",
            "relevance_score": 0.92,
        },
        {
            "url": "https://arxiv.org/abs/2005.14165",
            "title": "Language Models are Few-Shot Learners (GPT-3)",
            "snippet": "We test GPT-3 on tasks ranging from translation to QA...",
            "relevance_score": 0.81,
        },
    ],
}


class WebSearchTool(Tool):
    name: ClassVar[str] = "web_search"
    default_timeout_s: ClassVar[float] = 5.0

    async def _execute(self, query: str | None = None, top_k: int = 3, **_: Any) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("web_search requires a non-empty `query` string")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("web_search `top_k` must be a positive int")

        # Lookup is case-insensitive prefix match against canned keys.
        key = next(
            (k for k in _CANNED_RESULTS if k != "default" and k in query.lower()),
            "default",
        )
        results = _CANNED_RESULTS[key][:top_k]

        if not results:
            return ToolResult(status=ToolStatus.EMPTY, payload={"results": []})
        return ToolResult(
            status=ToolStatus.OK,
            payload={"results": results, "query": query},
        )
