"""
Evaluation of rag/win_qa.py against the golden dataset (eval/golden.jsonl).

Measures components separately (retrieval first, then end-to-end):

- Retrieval:  Recall@k and MRR. A hit is relevant if its page is in `expected_pages`.
- End-to-end: every case is run N times; the answer passes if it contains all
  `key_phrases` (positive cases) or the refusal text (negative cases).
  The pass rate per case shows unstable cases (passed once, failed once).

Usage:
    python eval/eval_rag.py              # retrieval + end-to-end (3 runs per case)
    python eval/eval_rag.py --retrieval  # retrieval only (embeddings only, cheap)

Needs OPEN_ROUTER_API_KEY. Results are printed as a table; traces of every
question land in traces/rag.jsonl.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
import win_qa  # noqa: E402

RUNS = 3
K = 5  # k for Recall@k (retrieval is measured on a wider list than the LLM sees)
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")


def load_golden() -> list[dict]:
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_index(program: str) -> list[dict]:
    doc = win_qa.DOCUMENTS[program]
    return win_qa.build_or_load_embeddings(doc["pdf"], doc["embeddings"], doc["label"])


def retrieval_metrics(case: dict, chunks: list[dict]) -> tuple[float, float]:
    """Return (recall@k, reciprocal rank) for one positive case."""
    hits = win_qa.retrieve(case["question"], chunks, top_k=K)
    relevant = set(case["expected_pages"])
    found = {h["page"] for h in hits} & relevant
    recall = len(found) / len(relevant)
    rank = next((i for i, h in enumerate(hits, start=1) if h["page"] in relevant), None)
    return recall, (1 / rank if rank else 0.0)


def passes(case: dict, answer: str) -> bool:
    if case["category"] == "Negativfall":
        return "keine information" in answer.lower()
    return all(p.lower() in answer.lower() for p in case["key_phrases"])


if __name__ == "__main__":
    retrieval_only = "--retrieval" in sys.argv
    cases = load_golden()
    indexes = {p: load_index(p) for p in {c["program"] for c in cases}}

    print(f"\n{'ID':6} {'Recall@%d' % K:9} {'RR':5} {'Pass':8} Frage")
    recalls, rrs, rates = [], [], []
    for case in cases:
        chunks = indexes[case["program"]]
        negative = case["category"] == "Negativfall"

        recall = rr = None
        if not negative:
            recall, rr = retrieval_metrics(case, chunks)
            recalls.append(recall)
            rrs.append(rr)

        rate = None
        if not retrieval_only:
            ok = sum(passes(case, win_qa.rag_query(case["question"], chunks, case["program"])) for _ in range(RUNS))
            rate = ok / RUNS
            rates.append(rate)

        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        passed = "-" if rate is None else f"{round(rate * RUNS)}/{RUNS}"
        print(f"{case['id']:6} {fmt(recall):9} {fmt(rr):5} {passed:8} {case['question'][:60]}")

    print(f"\nRecall@{K} = {sum(recalls) / len(recalls):.2f}   MRR = {sum(rrs) / len(rrs):.2f}")
    if rates:
        print(f"Pass rate (mean over cases and {RUNS} runs) = {sum(rates) / len(rates):.2f}")
        print("Report tendencies and error classes - with 15 cases, differences of a few percent are noise.")
