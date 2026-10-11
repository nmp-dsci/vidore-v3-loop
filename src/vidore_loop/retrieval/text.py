"""Channel A, text: BM25S over each page's OCR markdown, with its document's name prepended.

The markdown is the dataset's own OCR (the same text every text retriever on the
board reads); nothing is re-OCR'd here. Built once per dataset into
`data/cache/index/<key>/text/` in seconds; no model weights.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from vidore_loop.config import settings
from vidore_loop.data import pages

STOPWORDS = "en"
_WORD = re.compile(r"(?u)\b\w\w+\b")


def index_dir(key: str) -> Path:
    return settings().hf_home / "index" / key / "text"


def page_text(meta: dict[str, Any]) -> str:
    """What the text channel reads for a page: document name, then the OCR markdown."""
    title = str(meta["doc_id"]).replace("_", " ").replace("-", " ")
    return f"{title}\n{meta['markdown']}"


@cache
def _stemmer() -> Any:
    import Stemmer

    return Stemmer.Stemmer("english")


def build(key: str) -> int:
    """Index every page of a dataset (fetching any page not yet cached); returns pages indexed."""
    import bm25s

    ids = sorted(pages.page_index(key))
    texts = []
    for cid in ids:
        meta = pages.page_meta(key, cid)
        texts.append(page_text(meta) if meta else "")
    tokens = bm25s.tokenize(texts, stopwords=STOPWORDS, stemmer=_stemmer(), show_progress=False)
    model = bm25s.BM25()
    model.index(tokens, show_progress=False)
    out = index_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out), show_progress=False)
    (out / "ids.json").write_text(json.dumps(ids))
    _load.cache_clear()
    return len(ids)


def available(key: str) -> bool:
    return (index_dir(key) / "ids.json").exists()


@cache
def _load(key: str) -> tuple[Any, list[int]]:
    import bm25s

    d = index_dir(key)
    return bm25s.BM25.load(str(d)), json.loads((d / "ids.json").read_text())


def search(key: str, query: str, k: int = 100) -> list[tuple[int, float]]:
    """Top k pages as (corpus_id, BM25 score), best first; pages scoring 0 are dropped."""
    import bm25s

    model, ids = _load(key)
    q = bm25s.tokenize([query], stopwords=STOPWORDS, stemmer=_stemmer(), show_progress=False)
    if not q.ids or not q.ids[0]:
        return []
    docs, scores = model.retrieve(q, k=min(k, len(ids)), show_progress=False)
    return [(ids[int(i)], float(s)) for i, s in zip(docs[0], scores[0], strict=True) if s > 0]


def matched_terms(query: str, text: str) -> list[str]:
    """The query words BM25 actually scored on (its own stopwords and stems) that occur in the page."""
    import bm25s

    stem = _stemmer().stemWord
    scored = set(
        bm25s.tokenize(
            [query], stopwords=STOPWORDS, stemmer=_stemmer(), return_ids=False, show_progress=False
        )[0]
    )
    page = {stem(w) for w in _WORD.findall(text.lower())}
    out: list[str] = []
    for w in _WORD.findall(query.lower()):
        s = stem(w)
        if s in scored and s in page and w not in out:
            out.append(w)
    return out
