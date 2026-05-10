"""Structured data lookup tool.

The brief: queries a local database via natural language converted to
SQL by the agent. We run an in-memory SQLite database seeded with a
small fixed schema (papers + benchmarks tables) so eval cases are
deterministic.

The agent is expected to send SQL directly. We don't do NL->SQL
conversion inside the tool — that's the agent's job. Sending NL
without SQL gets MALFORMED_INPUT.

Safety: read-only by convention. We refuse anything that isn't a
SELECT to prevent agents from mutating the demo DB.
"""

from __future__ import annotations

import sqlite3
from typing import Any, ClassVar

from app.tools.base import Tool, ToolResult, ToolStatus

_SEED_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    primary_category TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmarks (
    name TEXT PRIMARY KEY,
    domain TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_benchmark_results (
    arxiv_id TEXT,
    benchmark_name TEXT,
    score REAL,
    PRIMARY KEY (arxiv_id, benchmark_name)
);

INSERT OR IGNORE INTO papers VALUES
    ('1810.04805', 'BERT', 2018, 'cs.CL'),
    ('2005.14165', 'GPT-3', 2020, 'cs.CL'),
    ('2106.09685', 'LoRA: Low-Rank Adaptation', 2021, 'cs.CL'),
    ('2305.18290', 'DPO: Direct Preference Optimization', 2023, 'cs.CL');

INSERT OR IGNORE INTO benchmarks VALUES
    ('GLUE', 'NLU'),
    ('SuperGLUE', 'NLU'),
    ('MMLU', 'reasoning');

INSERT OR IGNORE INTO paper_benchmark_results VALUES
    ('1810.04805', 'GLUE', 80.5),
    ('2005.14165', 'SuperGLUE', 71.8),
    ('2106.09685', 'GLUE', 84.3);
"""


class SQLLookupTool(Tool):
    name: ClassVar[str] = "sql_lookup"
    default_timeout_s: ClassVar[float] = 3.0

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SEED_SQL)
        self._conn.commit()

    async def _execute(self, sql: str | None = None, **_: Any) -> ToolResult:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql_lookup requires a non-empty `sql` string")

        # Read-only enforcement: first non-whitespace token must be SELECT.
        first_word = sql.lstrip().split(None, 1)[0].upper()
        if first_word != "SELECT":
            raise ValueError("sql_lookup is read-only; only SELECT queries are allowed")

        cur = self._conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return ToolResult(status=ToolStatus.EMPTY, payload={"rows": [], "sql": sql})
        return ToolResult(
            status=ToolStatus.OK,
            payload={"rows": rows, "row_count": len(rows), "sql": sql},
        )
