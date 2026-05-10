# Exploratory data understanding: arXiv cs.CL RAG corpus

Schema- and pipeline-driven summary for the **“understand data before modelling”** rubric. The snapshot `data/arxiv_corpus/papers.jsonl` is optional until you run `python -m scripts.ingest_corpus` (see README).

---

## 1. Corpus choice and rationale

**arXiv `cs.CL`** clusters around shared benchmarks, model families, and methods—so **multi-hop retrieval** is meaningful (evidence often spans several abstracts). We cap at **`ARXIV_MAX_PAPERS` (default 50)** for variety without a huge index; the JSONL can be **committed** for reproducibility. Ingestion sorts by **submitted date descending**, so the snapshot skews **recent** while staying bounded.

---

## 2. What’s in a `Paper`

Defined in `app/rag/corpus.py` as JSONL rows: **`arxiv_id`**, **`title`**, **`authors`**, **`abstract`**, **`primary_category`**, **`published`**, **`updated`**, **`pdf_url`**. Typical cs.CL **abstracts are ~150–300 words**. **Titles** often carry keywords (benchmarks, architecture names) the abstract does not repeat—hence **`chunk_text = title + "\n\n" + abstract`** for embedding.

---

## 3. Chunking strategy: one chunk per abstract

Full PDFs would use sliding windows; **abstracts are one semantic unit**. Sub-chunking would split claims and hurt attribution. **`text-embedding-3-small`** has large context; one chunk per paper fits easily. **Multi-hop** is implemented **across papers** (`MultiHopRetriever`), not by fragmenting one abstract.

---

## 4. Embedding choice: `text-embedding-3-small`

**1536-dim**, OpenAI API, same stack as the chat model—**cheap** for ~50 abstracts and **no local model download** for reviewers. Embeddings are computed in `app/rag/vector_store.py` and passed explicitly to Chroma.

---

## 5. Distance metric: cosine

OpenAI embeddings pair naturally with **cosine** similarity. Chroma collection uses **`metadata={"hnsw:space": "cosine"}`**. `RetrievedChunk.relevance_score` maps distance to a bounded score for logging.

---

## 6. Multi-hop suitability

Questions that need **two or more chunks** (e.g. relate benchmark X to method Y, or compare lineages) are ill-served by single-hop. The retriever **excludes prior `chunk_id`s** on later hops so hops are not duplicates. **`RetrievalOutput.is_compliant()`** (`len(hops) >= 2`) is enforced at the **orchestrator** with a logged **`PolicyViolation`**, not by rejecting parse.

---

## 7. Reproducibility note

**Commit** `papers.jsonl` when populated. `scripts/ingest_corpus.py` **merges** by `arxiv_id` without wiping the snapshot. **`--from-snapshot`** refills Chroma **offline** (no arXiv fetch)—used in Docker when the JSONL exists.
