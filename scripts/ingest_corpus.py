"""Ingest a fixed-size arXiv corpus into a JSONL snapshot and a Chroma collection.

Run modes:
    python -m scripts.ingest_corpus              # fetch from arXiv + populate Chroma
    python -m scripts.ingest_corpus --from-snapshot   # only repopulate Chroma from JSONL

The first mode hits arXiv's API (no key required, but rate-limited by their
TOS). Once the JSONL exists, the second mode is offline and idempotent — this
is what `docker compose up` should call so reviewers don't need network access
to arXiv to run the system.

Reproducibility: the JSONL is committed. Re-running the first mode will *only*
add new papers, never replace the existing snapshot, so the corpus the grader
sees is the corpus we evaluated against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arxiv  # type: ignore[import-untyped]

from app.rag.corpus import Paper, corpus_size, read_corpus, write_corpus
from app.rag.vector_store import make_persistent_store
from app.settings import get_settings

CORPUS_FILENAME = "papers.jsonl"


def fetch_papers(category: str, max_papers: int) -> list[Paper]:
    """Fetch up to max_papers from arXiv in the given category, most recent first."""
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    out: list[Paper] = []
    for result in client.results(search):
        out.append(
            Paper(
                arxiv_id=result.entry_id.rsplit("/", 1)[-1].split("v", 1)[0],
                # entry_id looks like 'http://arxiv.org/abs/2401.12345v2'
                # split off the version suffix so re-publishes don't fragment IDs
                title=result.title.strip(),
                authors=[a.name for a in result.authors],
                abstract=result.summary.strip().replace("\n", " "),
                primary_category=result.primary_category,
                published=result.published,
                updated=result.updated,
                pdf_url=result.pdf_url,
            )
        )
    return out


def merge_into_snapshot(new_papers: list[Paper], snapshot_path: Path) -> tuple[list[Paper], int]:
    """Merge new papers into the existing snapshot. Returns (full_list, added_count).

    De-dupe on arxiv_id. Existing entries are preserved (snapshot is authoritative
    for what the eval was scored against)."""
    existing: dict[str, Paper] = {}
    if snapshot_path.exists():
        for p in read_corpus(snapshot_path):
            existing[p.arxiv_id] = p

    added = 0
    for p in new_papers:
        if p.arxiv_id not in existing:
            existing[p.arxiv_id] = p
            added += 1
    return list(existing.values()), added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-snapshot",
        action="store_true",
        help="Skip arXiv fetch; only ingest the existing JSONL into Chroma.",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Fetch and write JSONL but skip Chroma ingest.",
    )
    args = parser.parse_args()

    settings = get_settings()
    snapshot_path = Path(settings.arxiv_corpus_path) / CORPUS_FILENAME

    # ---- Phase 1: ensure JSONL snapshot exists ----
    if args.from_snapshot:
        existing_count = corpus_size(snapshot_path)
        if existing_count == 0:
            print(
                f"ERROR: --from-snapshot but no snapshot at {snapshot_path}",
                file=sys.stderr,
            )
            return 1
        print(f"Using existing snapshot: {existing_count} papers at {snapshot_path}")
    else:
        print(
            f"Fetching up to {settings.arxiv_max_papers} {settings.arxiv_category} "
            "papers from arXiv..."
        )
        new_papers = fetch_papers(settings.arxiv_category, settings.arxiv_max_papers)
        print(f"  Got {len(new_papers)} from arXiv")
        all_papers, added = merge_into_snapshot(new_papers, snapshot_path)
        write_corpus(all_papers, snapshot_path)
        print(f"  Snapshot now has {len(all_papers)} papers ({added} new) at {snapshot_path}")

    if args.snapshot_only:
        return 0

    # ---- Phase 2: populate Chroma ----
    print(f"Populating Chroma at {settings.chroma_persist_dir}...")
    store = make_persistent_store(settings.chroma_persist_dir, settings.openai_api_key)
    papers = list(read_corpus(snapshot_path))
    n = store.ingest(papers)
    print(f"  Ingested {n} papers; collection now has {store.count()} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
