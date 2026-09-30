"""The English questions of one dataset, shaped for the viewer: lists, one question, a summary.

Scope is English queries only (AGENTS.md). Every row comes from the pinned
`queries`, `qrels` and `documents_metadata` configs; nothing here is invented.
Hand checks of reference answers live in `data/audits/hand_checks.jsonl`.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import cache
from statistics import mean, median
from typing import Any

from vidore_loop.config import DATA_DIR
from vidore_loop.data import registry

LANGUAGE = "english"
HAND_CHECKS = DATA_DIR / "audits" / "hand_checks.jsonl"


def _rows(key: str, config: registry.Config) -> list[dict[str, Any]]:
    from vidore_loop.data.loader import load

    rows: list[dict[str, Any]] = load(key, config).to_list()
    return rows


@cache
def queries(key: str) -> list[dict[str, Any]]:
    """English queries with their relevant-page counts, in query id order."""
    rel = qrels(key)
    out = []
    for q in _rows(key, "queries"):
        if q["language"] != LANGUAGE:
            continue
        pages = rel.get(int(q["query_id"]), [])
        out.append(
            {
                "query_id": int(q["query_id"]),
                "query": q["query"],
                "query_types": list(q["query_types"] or []),
                "query_format": q["query_format"],
                "content_type": list(q["content_type"] or []),
                "query_generator": q["query_generator"],
                "query_generation_pipeline": q.get("query_generation_pipeline"),
                "query_type_for_generation": q.get("query_type_for_generation"),
                "source_type": q.get("source_type"),
                "answer": q["answer"],
                "raw_answers": list(q["raw_answers"] or []),
                "n_pages": len(pages),
                "n_full": sum(1 for p in pages if p["score"] == 2),
            }
        )
    return sorted(out, key=lambda r: r["query_id"])


@cache
def qrels(key: str) -> dict[int, list[dict[str, Any]]]:
    """query_id → relevant pages, full answers (grade 2) first, with the human boxes."""
    out: dict[int, list[dict[str, Any]]] = {}
    for r in _rows(key, "qrels"):
        out.setdefault(int(r["query_id"]), []).append(
            {
                "corpus_id": int(r["corpus_id"]),
                "score": int(r["score"]),
                "content_type": list(r["content_type"] or []),
                "boxes": [
                    {k: int(b[k]) for k in ("annotator", "x1", "y1", "x2", "y2")}
                    for b in (r["bounding_boxes"] or [])
                ],
            }
        )
    for pages in out.values():
        pages.sort(key=lambda p: (-p["score"], p["corpus_id"]))
    return out


@cache
def documents(key: str) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": d["doc_id"],
            "file_name": d["file_name"],
            "url": d["url"],
            "doc_type": d["doc_type"],
            "doc_language": d["doc_language"],
            "doc_year": str(d["doc_year"]),
            "visual_types": list(d["visual_types"] or []),
            "pages": int(d["page_number"]),
            "license": d["license"],
        }
        for d in _rows(key, "documents_metadata")
    ]


def hand_checks() -> dict[tuple[str, int], dict[str, Any]]:
    if not HAND_CHECKS.exists():
        return {}
    out = {}
    for line in HAND_CHECKS.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            out[(c["dataset"], int(c["query_id"]))] = c
    return out


def summary(key: str) -> dict[str, Any]:
    """Counts a person needs before opening a question: by type, content, format, author."""
    qs = queries(key)
    n_pages = [q["n_pages"] for q in qs] or [0]
    return {
        "n_queries": len(qs),
        "by_type": Counter(t for q in qs for t in q["query_types"]).most_common(),
        "by_content": Counter(c for q in qs for c in q["content_type"]).most_common(),
        "by_format": Counter(q["query_format"] for q in qs).most_common(),
        "by_generator": Counter(q["query_generator"] for q in qs).most_common(),
        "pages_per_query": {
            "mean": round(mean(n_pages), 2),
            "median": median(n_pages),
            "max": max(n_pages),
        },
        "hand_checked": sum(1 for (k, _), _c in hand_checks().items() if k == key),
    }


def question(key: str, query_id: int) -> dict[str, Any] | None:
    q = next((r for r in queries(key) if r["query_id"] == query_id), None)
    if q is None:
        return None
    return {
        **q,
        "pages": qrels(key).get(query_id, []),
        "hand_check": hand_checks().get((key, query_id)),
    }
