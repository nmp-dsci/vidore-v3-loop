"""Claude answers a question from pages, with the paper's prompts (arXiv 2601.08620 v2, App. L).

Two settings, as in the paper's Table 3:

- `retrieved`: the top k pages of a retrieval run (k = 5, the paper's choice);
- `oracle`: every human-marked page (grade 1 or 2), the ceiling for the generator.

(The paper's no-document setting was dropped on the owner's call: it measures model
memory, not the system.) Each page is given as its image and its OCR markdown (the
plan's "image + markdown"; the paper gives one modality or both). Figure 25's wording
is kept, so our accuracy sits beside the paper's.
"""

from __future__ import annotations

import hashlib
from typing import Any

from vidore_loop.answer import claude
from vidore_loop.data import pages

MODES = ("retrieved", "oracle")
K = 5
MODALITIES = ("image+text", "image", "text")

# Figure 25, split around the pages
WITH_PAGES_HEAD = "You are an expert at answering query based on documents.\nHere is a list of relevant documents:"
WITH_PAGES_TAIL = (
    "Based on the above documents, answer the following query: {query}\n"
    "Keep the response short when appropriate. Output the answer only."
)


def prompt_version() -> str:
    """A short hash of the prompt text, recorded on every run."""
    raw = "\n".join((WITH_PAGES_HEAD, WITH_PAGES_TAIL))
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def page_blocks(key: str, corpus_ids: list[int], modality: str) -> list[claude.Block]:
    blocks: list[claude.Block] = []
    for i, cid in enumerate(corpus_ids, 1):
        meta = pages.page_meta(key, cid)
        img = pages.image_path(key, cid)
        if meta is None or img is None:
            continue
        blocks.append(claude.text_block(f"Document {i}: {meta['doc_id']}, page {meta['page']}"))
        if "image" in modality:
            blocks.append(claude.image_block(img))
        if "text" in modality and meta["markdown"].strip():
            blocks.append(claude.text_block(f"Text of document {i}:\n{meta['markdown']}"))
    return blocks


def build(key: str, query: str, corpus_ids: list[int], modality: str) -> list[claude.Block]:
    """The user message: Figure 25 around the pages."""
    return [
        claude.text_block(WITH_PAGES_HEAD),
        *page_blocks(key, corpus_ids, modality),
        claude.text_block(WITH_PAGES_TAIL.format(query=query)),
    ]


def answer(
    key: str,
    query: str,
    corpus_ids: list[int],
    model: str = "sonnet",
    effort: str = "medium",
    modality: str = "image+text",
) -> dict[str, Any]:
    """One answer: the text, the pages it was given, and the call's usage."""
    rep = claude.ask("", build(key, query, corpus_ids, modality), model=model, effort=effort)
    return {"answer": rep.text, "pages": corpus_ids, "usage": rep.usage()}
