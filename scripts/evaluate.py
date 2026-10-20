"""Retrieval evaluation: how often does search find the page that holds the answer?

Runs the app's real pipeline (parser, cleaner, chunker, embedder, Qdrant) in a
temporary folder, once per chunk configuration, and scores it on questions
whose correct pages are known (eval/questions.json).

Metrics
  hit@k  share of questions where a correct page is among the top k results
  MRR    mean reciprocal rank of the first correct result (1st=1, 2nd=0.5, 3rd=0.33, none=0)

Usage
  python scripts/evaluate.py                         # defaults below
  python scripts/evaluate.py --configs 600:120 --k 5 --verbose
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.documents.service import DocumentService  # noqa: E402
from app.infrastructure.fastembed_embedder import FastEmbedEmbedder  # noqa: E402
from app.infrastructure.qdrant_store import QdrantVectorStore  # noqa: E402
from app.infrastructure.sqlite_document_repository import SqliteDocumentRepository  # noqa: E402
from app.search.service import SearchService  # noqa: E402


def evaluate(pdf: Path, questions: list[dict], chunk_size: int, overlap: int, k: int, embedder) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        store = QdrantVectorStore("eval", embedder.dimension, path=f"{tmp}/qdrant")
        docs = DocumentService(SqliteDocumentRepository(f"{tmp}/docs.db"), store, embedder, chunk_size, overlap)
        document = docs.ingest_pdf(pdf.read_bytes(), pdf.name)
        search = SearchService(store, embedder, default_top_k=k)

        rows = []
        for q in questions:
            pages = [h.chunk.page for h in search.search(q["question"], k)]
            rank = next((i for i, p in enumerate(pages, 1) if p in q["pages"]), None)
            rows.append({"question": q["question"], "found": pages, "rank": rank})

    n = len(rows)
    return {
        "config": f"{chunk_size}/{overlap}",
        "chunks": document.chunks,
        "hit": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=next((ROOT / "samples").glob("*.pdf"), None),
                        help="defaults to the PDF in samples/")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    parser.add_argument("--configs", default="400:80,600:120,1000:200", help="chunk_size:overlap,...")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", help="show every question")
    args = parser.parse_args()

    if not args.pdf or not args.pdf.exists():
        sys.exit(f"Sample PDF not found: {args.pdf}  (put it there or pass --pdf)")
    questions = json.loads(args.questions.read_text())["questions"]
    settings = Settings()
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)

    print(f"{len(questions)} questions · {args.pdf.name} · model {settings.embedding_model.split('/')[-1]}\n")
    print(f"| chunk size/overlap | chunks | hit@{args.k} | MRR@{args.k} |")
    print("|---|---|---|---|")
    results = []
    for config in args.configs.split(","):
        size, overlap = map(int, config.split(":"))
        r = evaluate(args.pdf, questions, size, overlap, args.k, embedder)
        results.append(r)
        print(f"| {r['config']} | {r['chunks']} | {r['hit']:.0%} | {r['mrr']:.2f} |", flush=True)

    for r in results if args.verbose else []:
        print(f"\n{r['config']}:")
        for row in r["rows"]:
            mark = f"✓ rank {row['rank']}" if row["rank"] else "✗"
            print(f"  {mark:<9} pages {row['found']}  {row['question']}")


if __name__ == "__main__":
    main()
