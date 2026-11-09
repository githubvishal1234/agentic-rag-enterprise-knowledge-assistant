"""Retrieval eval: does search put each case's evidence passage in the top k?

    python evaluation/run_retrieval_eval.py --variant baseline --method vector
    python evaluation/run_retrieval_eval.py --variant v1 --method hybrid

Indexes data/eval_corpus/ with the app's own ingestion and vector store,
runs every case in evaluation/cases.jsonl that has evidence, and writes
.claude/hillclimb/retrieval/<variant>/{results.jsonl,errors.jsonl,traces/}.
Retrieval is local and deterministic, so each case runs once (rep 0).
"""

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.ingestion import ingest_file  # noqa: E402
from app.services.vector_store import FaissVectorStore  # noqa: E402

CORPUS_DIR = ROOT / "data" / "eval_corpus"
CASES_FILE = ROOT / "evaluation" / "cases.jsonl"
FLOW_DIR = ROOT / ".claude" / "hillclimb" / "retrieval"


def normalize(text: str) -> str:
    return " ".join(text.split())


def evidence_ranks(case: dict, results: list[dict]) -> list[int | None]:
    """1-based rank of the first retrieved chunk holding each evidence item."""

    ranks = []

    for evidence in case["evidence"]:
        rank = next(
            (
                position
                for position, result in enumerate(results, start=1)
                if result["metadata"]["source"] == evidence["source"]
                and normalize(evidence["text"]) in normalize(result["text"])
            ),
            None,
        )
        ranks.append(rank)

    return ranks


def grade(ranks: list[int | None]) -> dict:
    found = [rank for rank in ranks if rank is not None]

    return {
        "hit": float(len(found) == len(ranks)),
        "recall": len(found) / len(ranks),
        # Standard MRR: reciprocal rank of the first relevant chunk.
        "mrr": 1 / min(found) if found else 0.0,
    }


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0

    p = passed / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return center - half, center + half


def build_store() -> FaissVectorStore:
    store = FaissVectorStore(EmbeddingService())

    for path in sorted(CORPUS_DIR.iterdir()):
        # Default chunking, exactly as POST /ingest does it.
        store.add_documents(ingest_file(str(path)))

    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="baseline, v1, v2, ...")
    parser.add_argument("--method", required=True, choices=["vector", "hybrid"])
    parser.add_argument("--k", type=int, default=4, help="results per query (API default: 4)")
    args = parser.parse_args()

    if not re.fullmatch(r"baseline|v\d+", args.variant):
        parser.error("--variant must be 'baseline' or v<N>, or the report skips it")

    cases = [
        json.loads(line)
        for line in CASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [case for case in cases if case["evidence"]]

    store = build_store()
    search = store.similarity_search if args.method == "vector" else store.hybrid_search

    # Deterministic and fast, so every run starts clean instead of resuming.
    # Only this runner's output is cleared; change.md / change.patch stay.
    variant_dir = FLOW_DIR / args.variant
    shutil.rmtree(variant_dir / "traces", ignore_errors=True)
    (variant_dir / "traces").mkdir(parents=True)

    grades = []

    with (variant_dir / "results.jsonl").open("w", encoding="utf-8") as results_file, \
            (variant_dir / "errors.jsonl").open("w", encoding="utf-8") as errors_file:
        for case in cases:
            try:
                started = time.perf_counter()
                results = search(case["question"], k=args.k)
                latency = time.perf_counter() - started

                ranks = evidence_ranks(case, results)
                case_grade = grade(ranks)
            except Exception as error:
                # A crash is a harness failure, never a zero score.
                errors_file.write(json.dumps({
                    "prompt_id": case["id"],
                    "rep": 0,
                    "failure_class": "harness_error",
                    "error": repr(error),
                }) + "\n")
                continue

            grades.append((case, case_grade))

            results_file.write(json.dumps({
                "prompt_id": case["id"],
                "rep": 0,
                "prompt": case["question"],
                "tags": case["tags"],
                "status": "ok",
                "grade": case_grade,
                "latency_s": round(latency, 4),
                "meta": {"method": args.method, "k": args.k, "evidence_ranks": ranks},
            }) + "\n")

            evidence_lines = "\n".join(
                f"- {evidence['source']}: \"{evidence['text']}\" -> "
                + (f"rank {rank}" if rank else f"NOT in top {args.k}")
                for evidence, rank in zip(case["evidence"], ranks)
            )
            trace = [
                {"role": "user", "content": case["question"]},
                {
                    "role": "tool_call",
                    "name": f"{args.method}_search",
                    "content": json.dumps({"query": case["question"], "k": args.k}, indent=2),
                },
                {
                    "role": "tool_result",
                    "name": f"{args.method}_search",
                    "content": json.dumps(
                        [
                            {
                                "rank": rank,
                                "source": result["metadata"]["source"],
                                "chunk_id": result["metadata"]["chunk_id"],
                                "score": round(result["score"], 4),
                                "text": result["text"],
                            }
                            for rank, result in enumerate(results, start=1)
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"Grade: {case_grade}\n\nExpected evidence:\n{evidence_lines}",
                },
            ]
            (variant_dir / "traces" / f"{case['id']}_rep0.json").write_text(
                json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    hits = sum(int(g["hit"]) for _, g in grades)
    low, high = wilson_interval(hits, len(grades))
    print(
        f"{args.variant} ({args.method}, k={args.k}): hit {hits}/{len(grades)} "
        f"= {hits / len(grades):.0%} [95% CI {low:.0%}-{high:.0%}], "
        f"recall {sum(g['recall'] for _, g in grades) / len(grades):.2f}, "
        f"MRR {sum(g['mrr'] for _, g in grades) / len(grades):.2f}, "
        f"errors {len(cases) - len(grades)}"
    )

    for category in dict.fromkeys(case["tags"][0] for case, _ in grades):
        in_category = [g for case, g in grades if case["tags"][0] == category]
        print(f"  {category:<12} hit {sum(int(g['hit']) for g in in_category)}/{len(in_category)}")


if __name__ == "__main__":
    main()
