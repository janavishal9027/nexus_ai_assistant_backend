"""RAG retrieval evaluation harness.

Loads a labeled JSONL dataset and reports Recall@K, Precision@K, MRR, nDCG@K and
Hit Rate over a knowledge base's retrieval pipeline (the real ``retrieve()``, so
rewrite → hybrid → rerank → parent-expand is all exercised).
See docs/semantic-embedding/09-evaluation-observability.md.

Dataset — one JSON object per line:
    {"query": "How do I rotate a refresh token?", "relevant_chunk_ids": [42, 43]}
    {"query": "What is the login endpoint?",      "relevant_document_ids": [7]}

Usage:
    cd backend
    python -m scripts.rag_eval --kb-id 1 --owner-id 2 --dataset scripts/rag_eval.sample.jsonl --k 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path

# Allow both "python -m scripts.rag_eval" and "python scripts/rag_eval.py".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal            # noqa: E402
from app.models.rag_models import KnowledgeBase  # noqa: E402
from app.services.rag_retrieval import retrieve   # noqa: E402


def _load(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def _hit_vector(results: list[dict], rel_chunks: set, rel_docs: set) -> list[int]:
    return [
        1 if (r["chunk_id"] in rel_chunks or r["document_id"] in rel_docs) else 0
        for r in results
    ]


def _dcg(hits: list[int]) -> float:
    return sum(h / math.log2(i + 2) for i, h in enumerate(hits))


# ── LLM answer grading (reference-free) ─────────────────────────────────────

_GRADE_KEYS = ("groundedness", "citation_correctness", "relevance")

_JUDGE_SYSTEM = (
    "You are a strict RAG answer grader. Given a QUESTION, the numbered CONTEXT "
    "sources, and an ANSWER, score each 0.0-1.0: "
    "groundedness (is every claim supported by the context?), "
    "citation_correctness (do the bracketed [n] citations point to real, relevant "
    "sources?), relevance (does it actually answer the question?). "
    'Return ONLY compact JSON: {"groundedness":x,"citation_correctness":x,"relevance":x}'
)


def _parse_scores(text: str) -> dict:
    out = {k: 0.0 for k in _GRADE_KEYS}
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            for k in _GRADE_KEYS:
                v = data.get(k)
                if isinstance(v, (int, float)):
                    out[k] = max(0.0, min(1.0, float(v)))
        except Exception:
            pass
    return out


async def _grade_answer(db, query: str, sources: list[dict]) -> tuple[str, dict]:
    """Generate a grounded answer, then LLM-judge it. Needs a chat-model key."""
    from app.services.fallback_router import route_chat
    from app.services.rag_retrieval import build_grounded_messages
    from app.models.schemas import MessageDto

    answer = (await route_chat(
        db, build_grounded_messages(query, sources),
        temperature=0.2, max_tokens=400)).content or ""
    ctx = "\n\n".join(f"[{c['index']}] {c['text'][:600]}" for c in sources) or "(none)"
    judged = (await route_chat(db, [
        MessageDto(role="system", content=_JUDGE_SYSTEM),
        MessageDto(role="user",
                   content=f"QUESTION: {query}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nJSON:"),
    ], temperature=0.0, max_tokens=120)).content or ""
    return answer, _parse_scores(judged)


async def _run(kb_id: int, owner_id, dataset: str, k: int, grade: bool = False) -> None:
    db = SessionLocal()
    try:
        kb = db.get(KnowledgeBase, kb_id)
        if kb is None:
            print(f"Knowledge base {kb_id} not found.")
            return
        rows = _load(dataset)
        if not rows:
            print("Dataset is empty.")
            return

        agg = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "hit": 0.0}
        gagg = {gk: 0.0 for gk in _GRADE_KEYS}
        graded = 0
        for row in rows:
            query = row["query"]
            rel_ch = set(row.get("relevant_chunk_ids") or [])
            rel_dc = set(row.get("relevant_document_ids") or [])
            n_rel = len(rel_ch) + len(rel_dc)

            results = await retrieve(db, kb, query, owner_id)
            topk = results[:k]
            hits = _hit_vector(topk, rel_ch, rel_dc)
            n_hit = sum(hits)

            agg["recall"] += (n_hit / n_rel) if n_rel else 0.0
            agg["precision"] += (n_hit / len(topk)) if topk else 0.0
            agg["mrr"] += next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
            idcg = _dcg([1] * min(n_rel, k)) or 1.0
            agg["ndcg"] += _dcg(hits) / idcg
            agg["hit"] += 1.0 if n_hit else 0.0

            mark = "✓" if n_hit else "·"
            line = f"  {mark} hit={n_hit}/{min(n_rel, k) or '?'}  {query[:60]}"
            if grade and topk:
                try:
                    _, sc = await _grade_answer(db, query, topk)
                    for gk in _GRADE_KEYS:
                        gagg[gk] += sc[gk]
                    graded += 1
                    line += (f"  [g={sc['groundedness']:.2f} "
                             f"c={sc['citation_correctness']:.2f} r={sc['relevance']:.2f}]")
                except Exception as exc:
                    line += f"  [grade failed: {exc}]"
            print(line)

        n = len(rows)
        print("\n── RAG retrieval eval ──────────────────────────────")
        print(f"  Knowledge base : {kb.name} (#{kb_id})")
        print(f"  Queries        : {n}")
        print(f"  Recall@{k}      : {agg['recall'] / n:.3f}")
        print(f"  Precision@{k}   : {agg['precision'] / n:.3f}")
        print(f"  MRR            : {agg['mrr'] / n:.3f}")
        print(f"  nDCG@{k}        : {agg['ndcg'] / n:.3f}")
        print(f"  HitRate@{k}     : {agg['hit'] / n:.3f}")
        if graded:
            print("  ── answer grading (LLM judge) ──")
            print(f"  Groundedness   : {gagg['groundedness'] / graded:.3f}")
            print(f"  Citation acc.  : {gagg['citation_correctness'] / graded:.3f}")
            print(f"  Relevance      : {gagg['relevance'] / graded:.3f}")
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    ap.add_argument("--kb-id", type=int, required=True)
    ap.add_argument("--owner-id", type=int, default=None)
    ap.add_argument("--dataset", required=True, help="path to a JSONL label file")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--grade", action="store_true",
                    help="also generate + LLM-grade answers (groundedness/citation/relevance)")
    args = ap.parse_args()
    asyncio.run(_run(args.kb_id, args.owner_id, args.dataset, args.k, args.grade))


if __name__ == "__main__":
    main()
