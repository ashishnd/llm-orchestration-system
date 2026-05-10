"""Run the full eval harness end-to-end.

    python -m scripts.run_eval                  # full 15-case run
    python -m scripts.run_eval --case-ids c1,c2 # subset

Persists EvalRun + EvalScore rows. Prints a summary at the end.

Pre-requisites:
- Postgres running (or any SQLAlchemy URL configured via env)
- Chroma corpus populated (run scripts/ingest_corpus.py first)
- OPENAI_API_KEY set
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.eval import run_eval
from app.llm import get_llm_client
from app.persistence import get_sessionmaker, init_db, repository
from app.rag import make_persistent_store
from app.settings import get_settings


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="Comma-separated subset of case IDs to run (default: all 15)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional notes attached to the EvalRun row",
    )
    args = parser.parse_args()

    case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()] if args.case_ids else None

    settings = get_settings()
    await init_db()
    Session = get_sessionmaker()
    llm = get_llm_client()
    store = make_persistent_store(settings.chroma_persist_dir, settings.openai_api_key)

    async with Session() as session:
        run_id = await run_eval(
            session=session,
            llm=llm,
            vector_store=store,
            case_ids=case_ids,
            notes=args.notes,
        )
        scores = await repository.get_eval_scores(session, run_id)

    # Print summary
    print(f"\nEval run {run_id} complete.")
    print(
        f"  {len(scores)} score rows across " f"{len({s.test_case_id for s in scores})} test cases."
    )
    by_dim = {}
    for s in scores:
        by_dim.setdefault(s.dimension, []).append(s.score)
    print("\nMean score per dimension:")
    for dim, vals in sorted(by_dim.items()):
        print(f"  {dim}: {sum(vals) / len(vals):.3f}  (n={len(vals)})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
