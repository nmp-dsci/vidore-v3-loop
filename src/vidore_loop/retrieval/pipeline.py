"""The retrieval half of the RAG pipeline: text and visual channels → fusion → rerank.

`run` takes a question (a benchmark question by id, or free text) and returns every
stage's ranking, timing and, when the question has gold pages, the full metric set
from `eval.metrics.report`. A stage whose index or model is missing is reported as
unavailable rather than failing the others, so the tab works from the moment the
text index exists.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from vidore_loop.data import catalog, pages
from vidore_loop.eval import metrics
from vidore_loop.retrieval import fuse, text

STAGES = ("text", "visual", "fused", "reranked")
CHANNEL_K = 100  # each channel's list, the pool fusion draws on
FUSED_K = 50  # what the reranker reads
SHOW = 10  # hits returned per stage
# One request on the GPU at a time: the API serves requests from a thread pool, and two threads
# driving MPS at once abort the process (a Metal command-buffer assertion). Covers model loads too.
GPU = threading.Lock()


def status(key: str) -> dict[str, Any]:
    from vidore_loop.retrieval import visual

    mk = visual.active(key)
    v = visual.stats(key, mk) if mk else None
    return {
        "text": text.available(key),
        "visual": bool(v),
        "visual_complete": bool(v and v.get("complete")),
        "visual_building": visual.building(key),
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


def compute(
    key: str,
    query_text: str,
    stages: tuple[str, ...] = STAGES,
    rerank_depth: int = 20,
    weights: tuple[float, float] | None = None,
    st: dict[str, Any] | None = None,
    question_encoder: str | None = None,
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, dict[str, Any]]]:
    """Every requested stage's full ranking as (corpus_id, score), and each stage's timing,
    cost or the reason it is unavailable. The viewer's `run` and M3's scored runs share it,
    so a question scores the same in the RAG pipeline tab and in a run. A `question_encoder`
    (retrieval.tune) encodes the question for the visual channel; the pages stay as built."""
    st = st or status(key)
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
        elif st["visual_building"]:
            out["visual"] = {
                "unavailable": f"the visual index is being built ({st['visual_stats']['pages']} of "
                f"{len(pages.page_index(key))} pages so far); searching would load a second copy of "
                "the model, which does not fit in memory beside the build"
            }
        else:
            from vidore_loop.retrieval import visual

            with GPU:
                t0 = time.time()
                lists["visual"] = visual.search(
                    key, query_text, CHANNEL_K, question_encoder=question_encoder
                )
                out["visual"] = {"ms": _ms(t0)}
            if not st["visual_complete"]:
                out["visual"]["note"] = (
                    f"partial index: {st['visual_stats']['pages']} of {len(pages.page_index(key))} pages"
                )

    channels = [[cid for cid, _ in lists[c]] for c in ("text", "visual") if c in lists]
    if channels and ("fused" in stages or "reranked" in stages):
        t0 = time.time()
        inputs = [c for c in ("text", "visual") if c in lists]
        chosen = dict(zip(("text", "visual"), weights or fuse.WEIGHTS, strict=True))
        # the weights were chosen for both channels; with one channel it passes through as is
        used = {c: chosen[c] for c in inputs} if len(inputs) == 2 else {inputs[0]: 1.0}
        lists["fused"] = fuse.rrf(channels, k=FUSED_K, weights=list(used.values()))
        out["fused"] = {"ms": _ms(t0), "inputs": inputs, "weights": used}

    if "reranked" in stages and "fused" in lists:
        if not st["reranker"]:
            out["reranked"] = {
                "unavailable": "the reranker is not ready: `uv sync --extra retrieval`, then "
                f"`make index DATASET={key} STAGE=reranker` downloads zerank-2 (≈8 GB)"
            }
        else:
            from vidore_loop.retrieval import rerank

            depth = max(1, min(rerank_depth, len(lists["fused"])))
            cands = []
            for cid, _ in lists["fused"][:depth]:
                meta = pages.page_meta(key, cid)
                cands.append((cid, text.page_text(meta) if meta else ""))
            with GPU:
                t0 = time.time()
                ranked, cost = rerank.rerank(key, query_text, cands)
            # pages below the depth keep their fused order beneath the reranked ones
            tail = [(cid, float("-inf")) for cid, _ in lists["fused"][depth:]]
            lists["reranked"] = ranked + tail
            out["reranked"] = {"ms": _ms(t0), "model": rerank.MODEL_ID, "depth": depth, **cost}

    return lists, out


def run(
    key: str,
    query: str | None = None,
    query_id: int | None = None,
    stages: tuple[str, ...] = STAGES,
    rerank_depth: int = 20,
    weights: tuple[float, float] | None = None,
) -> dict[str, Any]:
    q = catalog.question(key, query_id) if query_id is not None else None
    if query_id is not None and q is None:
        raise KeyError(f"no English query {query_id} in {key}")
    query_text = (query or (q["query"] if q else "")).strip()
    if not query_text:
        raise ValueError("a question is needed: pass query or query_id")
    gold: dict[int, int] = {p["corpus_id"]: p["score"] for p in q["pages"]} if q else {}
    st = status(key)
    lists, out = compute(key, query_text, stages, rerank_depth, weights, st)

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
                "scores": {c: _score_of(cid, lists[c]) for c in lists},
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


def _score_of(cid: int, ranked: list[tuple[int, float]]) -> float | None:
    """The page's score in one stage's list (BM25, MaxSim, RRF or reranker); None if absent."""
    for c, score in ranked:
        if c == cid:
            return round(score, 4) if score != float("-inf") else None
    return None


def _ms(t0: float) -> int:
    return round((time.time() - t0) * 1000)
