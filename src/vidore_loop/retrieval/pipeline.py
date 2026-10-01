"""The retrieval half of the RAG pipeline: text and visual channels → fusion → rerank.

`run` takes a question (a benchmark question by id, or free text) and returns every
stage's ranking, timing and, when the question has gold pages, the full metric set
from `eval.metrics.report`. A stage whose index or model is missing is reported as
unavailable rather than failing the others, so the tab works from the moment the
text index exists.
"""

from __future__ import annotations

import time
from typing import Any

from vidore_loop.data import catalog, pages
from vidore_loop.eval import metrics
from vidore_loop.retrieval import fuse, text

STAGES = ("text", "visual", "fused", "reranked")
CHANNEL_K = 100  # each channel's list, the pool fusion draws on
FUSED_K = 50  # what the reranker reads
SHOW = 10  # hits returned per stage


def status(key: str) -> dict[str, Any]:
    from vidore_loop.retrieval import visual

    mk = visual.active(key)
    v = visual.stats(key, mk) if mk else None
    return {
        "text": text.available(key),
        "visual": bool(v),
        "visual_complete": bool(v and v.get("complete")),
        "visual_stats": v,
        "reranker": _heavy_ok() and _reranker_downloaded(),
    }


def _reranker_downloaded() -> bool:
    from vidore_loop.retrieval import rerank

    return rerank.downloaded()


def _heavy_ok() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def run(
    key: str,
    query: str | None = None,
    query_id: int | None = None,
    stages: tuple[str, ...] = STAGES,
) -> dict[str, Any]:
    q = catalog.question(key, query_id) if query_id is not None else None
    if query_id is not None and q is None:
        raise KeyError(f"no English query {query_id} in {key}")
    query_text = (query or (q["query"] if q else "")).strip()
    if not query_text:
        raise ValueError("a question is needed: pass query or query_id")
    gold: dict[int, int] = {p["corpus_id"]: p["score"] for p in q["pages"]} if q else {}
    st = status(key)
    out: dict[str, dict[str, Any]] = {}
    lists: dict[str, list[tuple[int, float]]] = {}

    if "text" in stages or "fused" in stages or "reranked" in stages:
        if st["text"]:
            t0 = time.time()
            lists["text"] = text.search(key, query_text, CHANNEL_K)
            out["text"] = {"ms": _ms(t0)}
        else:
            out["text"] = {
                "unavailable": "no text index: run `make index DATASET=" + key + " STAGE=text`"
            }

    if "visual" in stages or "fused" in stages or "reranked" in stages:
        if not st["visual"]:
            out["visual"] = {
                "unavailable": "no visual index: run `make index DATASET=" + key + " STAGE=visual`"
            }
        elif not _heavy_ok():
            out["visual"] = {
                "unavailable": "the retrieval extra is not installed: `uv sync --extra retrieval`"
            }
        else:
            from vidore_loop.retrieval import visual

            t0 = time.time()
            lists["visual"] = visual.search(key, query_text, CHANNEL_K)
            out["visual"] = {"ms": _ms(t0)}
            if not st["visual_complete"]:
                out["visual"]["note"] = (
                    f"partial index: {st['visual_stats']['pages']} of {len(pages.page_index(key))} pages"
                )

    channels = [[cid for cid, _ in lists[c]] for c in ("text", "visual") if c in lists]
    if channels and ("fused" in stages or "reranked" in stages):
        t0 = time.time()
        lists["fused"] = fuse.rrf(channels, k=FUSED_K)
        out["fused"] = {"ms": _ms(t0), "inputs": [c for c in ("text", "visual") if c in lists]}

    if "reranked" in stages and "fused" in lists:
        if not st["reranker"]:
            out["reranked"] = {
                "unavailable": "the reranker is not ready: `uv sync --extra retrieval`, then "
                f"`make index DATASET={key} STAGE=reranker` downloads zerank-2 (≈8 GB)"
            }
        else:
            from vidore_loop.retrieval import rerank

            cands = []
            for cid, _ in lists["fused"]:
                meta = pages.page_meta(key, cid)
                cands.append((cid, text.page_text(meta) if meta else ""))
            t0 = time.time()
            ranked, _secs = rerank.rerank(query_text, cands)
            lists["reranked"] = ranked
            out["reranked"] = {"ms": _ms(t0), "model": rerank.MODEL_ID}

    index = pages.page_index(key)
    for name in STAGES:
        if name not in stages or name not in lists:
            continue
        ranking = [cid for cid, _ in lists[name]]
        hits = []
        for rank, (cid, score) in enumerate(lists[name][:SHOW], start=1):
            e = index.get(cid, {})
            hit: dict[str, Any] = {
                "corpus_id": cid,
                "rank": rank,
                "score": round(score, 4),
                "doc_id": e.get("doc_id"),
                "page": e.get("page"),
                "grade": gold.get(cid, 0) if gold else None,
                "ranks": {c: _rank_of(cid, lists[c]) for c in lists},
            }
            if name in ("text", "fused", "reranked"):
                meta = pages.page_meta(key, cid)
                hit["terms"] = text.matched_terms(query_text, text.page_text(meta)) if meta else []
            hits.append(hit)
        out[name]["hits"] = hits
        out[name]["n"] = len(ranking)
        if gold:
            out[name]["metrics"] = metrics.report(ranking, gold)
            out[name]["gold_ranks"] = {str(cid): _rank_of(cid, lists[name]) for cid in gold}

    deltas = {}
    if gold:
        for before, after in (("text", "fused"), ("visual", "fused"), ("fused", "reranked")):
            if before in lists and after in lists:
                deltas[f"{before}→{after}"] = metrics.stage_delta(
                    [c for c, _ in lists[before]], [c for c, _ in lists[after]], gold
                )

    return {
        "dataset": key,
        "query": query_text,
        "query_id": query_id,
        "gold": [
            {"corpus_id": cid, "grade": g}
            for cid, g in sorted(gold.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "stages": out,
        "deltas": deltas,
        "status": st,
    }


def _rank_of(cid: int, ranked: list[tuple[int, float]]) -> int | None:
    for i, (c, _) in enumerate(ranked, start=1):
        if c == cid:
            return i
    return None


def _ms(t0: float) -> int:
    return round((time.time() - t0) * 1000)
