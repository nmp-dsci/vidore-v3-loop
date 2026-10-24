"""M5: the agent. Claude searches, reads pages, then gives one structured reply.

One Agent SDK session per question, sealed like M4's call (`claude.agent`): no built-in
tools, only two in-process ones.

- `search(query)`: the champion ranking (`pipeline.compute`, the fused stage with the
  viewer's weights, exactly as scored runs use it); its top 10 as page id, document, page
  number and the opening of the page's text. At most `max_searches` calls.
- `view_page(page_id)`: the page image and its OCR markdown, as M4 gives them. At most
  `max_views` pages.

The final reply is structured output: up to 10 page ids (scored as a retrieval ranking),
the answer in the paper's wording (short, the answer only, so it is judged like M4's) and
the pages it cites. A short ranking is filled from reciprocal rank fusion over every search
the agent made (NeMo's fallback), so NDCG@10 always scores 10 pages. The tools record each
step, which is what the Agent tab draws.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vidore_loop.answer import claude
from vidore_loop.data import pages, registry
from vidore_loop.retrieval import fuse

SERVER = "vidore"
TOOLS = ("search", "view_page")
MAX_SEARCHES = 5
MAX_VIEWS = 10
SHOW = 10  # results a search returns
SNIPPET = 300  # characters of each result's text

SYSTEM = """You answer one question about a collection of {pages:,} pages from {documents} {domain} documents, using two tools.

- search(query) returns the 10 pages that best match a query: each page's id, its document, its page number and the start of its text. You may search {max_searches} times. Rewrite or split the question when the results miss the evidence.
- view_page(page_id) shows a page's image and its full text. You may view {max_views} pages. Read a page before relying on it: tables and charts are often only legible in the image.

When you have the evidence, give your final reply:
- ranking: up to 10 page ids, the pages that best answer the question, most relevant first. It is scored as a retrieval ranking, so include every page that holds part of the answer.
- answer: the answer to the question. Keep the response short when appropriate. Output the answer only.
- cited: the page ids your answer relies on."""

SEARCH_DOC = "Search the collection. Returns the 10 best-matching pages for the query."
VIEW_DOC = "View one page: its image and its full text."

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}, "maxItems": SHOW},
        "answer": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["ranking", "answer", "cited"],
    "additionalProperties": False,
}

SearchFn = Callable[[str, str], list[int]]


def prompt_version() -> str:
    raw = json.dumps([SYSTEM, SEARCH_DOC, VIEW_DOC, SCHEMA], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def system_prompt(key: str, max_searches: int, max_views: int) -> str:
    d = registry.get(key)
    return SYSTEM.format(
        pages=d.pages,
        documents=d.documents,
        domain=d.domain,
        max_searches=max_searches,
        max_views=max_views,
    )


def champion_search(key: str, query: str) -> list[int]:
    """The champion ranking for free text: the fused stage with the viewer's weights."""
    from vidore_loop.retrieval import pipeline

    lists, out = pipeline.compute(key, query, ("fused",), weights=fuse.WEIGHTS)
    if "fused" not in lists:
        raise RuntimeError(
            out.get("fused", {}).get("unavailable") or "the fused stage is unavailable"
        )
    return [cid for cid, _ in lists["fused"]]


@dataclass
class Trace:
    """What one attempt did: every step, each search's full ranking, the pages viewed."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    searches: list[list[int]] = field(default_factory=list)
    viewed: list[int] = field(default_factory=list)


def _text(s: str, error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"content": [{"type": "text", "text": s}]}
    if error:
        out["is_error"] = True
    return out


def _label(key: str, cid: int) -> str:
    m = pages.page_meta(key, cid) or {}
    return f"page {cid} · {m.get('doc_id', '?')} · p.{m.get('page', '?')}"


class Kit:
    """Builds fresh tools for each attempt (`claude.agent` calls it); keeps the latest
    attempt's trace and tool handlers."""

    def __init__(
        self,
        key: str,
        max_searches: int = MAX_SEARCHES,
        max_views: int = MAX_VIEWS,
        search_fn: SearchFn = champion_search,
    ) -> None:
        self.key, self.max_searches, self.max_views, self.search_fn = (
            key,
            max_searches,
            max_views,
            search_fn,
        )
        self.trace = Trace()
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def __call__(self) -> claude.ToolKit:
        from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool

        tr = self.trace = Trace()
        key = self.key

        async def search(args: dict[str, Any]) -> dict[str, Any]:
            return run_search(
                key, tr, str(args.get("query", "")), self.max_searches, self.search_fn
            )

        async def view_page(args: dict[str, Any]) -> dict[str, Any]:
            return run_view(key, tr, args.get("page_id"), self.max_views)

        tools = [
            tool(
                "search",
                SEARCH_DOC,
                {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "what to look for"}},
                    "required": ["query"],
                },
                annotations=ToolAnnotations(readOnlyHint=True),
            )(search),
            tool(
                "view_page",
                VIEW_DOC,
                {
                    "type": "object",
                    "properties": {"page_id": {"type": "integer", "description": "a page id"}},
                    "required": ["page_id"],
                },
                # a page's image and text stay inline, not spilled to a file
                annotations=ToolAnnotations(readOnlyHint=True, maxResultSizeChars=200_000),
            )(view_page),
        ]
        self.handlers = {t.name: t.handler for t in tools}
        server = create_sdk_mcp_server(SERVER, tools=tools)
        return {SERVER: server}, [f"mcp__{SERVER}__{t}" for t in TOOLS]


def run_search(key: str, tr: Trace, query: str, cap: int, search_fn: SearchFn) -> dict[str, Any]:
    query = query.strip()
    if len(tr.searches) >= cap:
        tr.steps.append({"tool": "search", "query": query, "refused": "cap"})
        return _text(f"Search limit reached ({cap}). Answer from the pages you have.")
    t0 = time.time()
    try:
        ranking = search_fn(key, query)
    except Exception as e:  # noqa: BLE001 - the agent sees the failure as a tool error
        tr.steps.append({"tool": "search", "query": query, "error": str(e)[:300]})
        return _text(f"Search failed: {e}", error=True)
    tr.searches.append(ranking)
    top = ranking[:SHOW]
    tr.steps.append(
        {"tool": "search", "query": query, "results": top, "ms": int((time.time() - t0) * 1000)}
    )
    lines = []
    for i, cid in enumerate(top, 1):
        m = pages.page_meta(key, cid) or {}
        snippet = " ".join(str(m.get("markdown", "")).split())[:SNIPPET]
        lines.append(f"{i}. {_label(key, cid)}\n   {snippet}")
    left = cap - len(tr.searches)
    return _text("\n".join(lines) + f"\n({left} searches left)")


def run_view(key: str, tr: Trace, page_id: Any, cap: int) -> dict[str, Any]:
    try:
        cid = int(page_id)
    except (TypeError, ValueError):
        return _text(f"{page_id!r} is not a page id.", error=True)
    if cid in tr.viewed:
        return _text(f"Page {cid} is already shown above.")
    if len(tr.viewed) >= cap:
        tr.steps.append({"tool": "view_page", "page": cid, "refused": "cap"})
        return _text(f"Page view limit reached ({cap}). Answer from the pages you have.")
    meta = pages.page_meta(key, cid)
    img = pages.image_path(key, cid)
    if meta is None or img is None:
        return _text(f"There is no page {cid}.", error=True)
    tr.viewed.append(cid)
    tr.steps.append({"tool": "view_page", "page": cid})
    left = cap - len(tr.viewed)
    return {
        "content": [
            {"type": "text", "text": _label(key, cid)},
            {
                "type": "image",
                "data": base64.b64encode(img.read_bytes()).decode(),
                "mimeType": "image/jpeg",
            },
            {
                "type": "text",
                "text": f"Text of page {cid}:\n{meta['markdown']}\n({left} page views left)",
            },
        ]
    }


def _ids(xs: Any, known: set[int]) -> list[int]:
    out: list[int] = []
    for x in xs if isinstance(xs, list) else []:
        try:
            c = int(x)
        except (TypeError, ValueError):
            continue
        if c in known and c not in out:
            out.append(c)
    return out


def fill(ranking: list[int], searches: list[list[int]], k: int = SHOW) -> list[int]:
    """The agent's ranking, topped up to k from RRF over every search it made."""
    out = list(ranking[:k])
    if len(out) < k and searches:
        for cid, _ in fuse.rrf(searches, k=fuse_depth(searches)):
            if cid not in out:
                out.append(cid)
            if len(out) >= k:
                break
    return out


def fuse_depth(searches: list[list[int]]) -> int:
    return max((len(s) for s in searches), default=0)


def answer(
    key: str,
    query: str,
    model: str = "sonnet",
    effort: str = "medium",
    max_searches: int = MAX_SEARCHES,
    max_views: int = MAX_VIEWS,
    search_fn: SearchFn = champion_search,
) -> dict[str, Any]:
    """One question through the agent: its ranking, answer, cited pages, steps and usage."""
    kit = Kit(key, max_searches, max_views, search_fn)
    rep = claude.agent(
        system_prompt(key, max_searches, max_views),
        [claude.text_block(f"Question: {query}")],
        kit,
        model=model,
        effort=effort,
        schema=SCHEMA,
        max_turns=max_searches + max_views + 6,
    )
    s = rep.structured if isinstance(rep.structured, dict) else None
    if not s or not str(s.get("answer", "")).strip():
        raise claude.ClaudeError(f"the agent gave no structured reply: {rep.text[:200]!r}")
    tr = kit.trace
    known = set(pages.page_index(key))
    ranked = _ids(s.get("ranking"), known)[:SHOW]
    return {
        "ranking": fill(ranked, tr.searches),
        "agent_ranking": ranked,
        "answer": str(s["answer"]).strip(),
        "cited": _ids(s.get("cited"), known),
        "steps": tr.steps,
        "searches": len(tr.searches),
        "views": len(tr.viewed),
        "usage": rep.usage(),
    }
