"""Stage D: zerank-2 rereads the top of the fused list and reorders it.

zerank-2 (ZeroEntropy, Apache-2.0) is a 4B cross-encoder: it reads the question and
one page's OCR markdown together and returns a relevance logit. Its hosted API closed
in September 2026, so it runs here. On the M1 Pro (MPS) it costs ≈3.2 s a page at
1,024 tokens (≈1.7 s at 512) and batching does not help, so the depth is a choice
(default 20) and every (question, page) score is cached on disk: a question is
reranked once, then instant.
"""

from __future__ import annotations

import sqlite3
import time
from functools import cache
from typing import Any

from vidore_loop.config import settings
from vidore_loop.retrieval.visual import device

MODEL_ID = "zeroentropy/zerank-2-reranker"
MAX_TOKENS = 1024
DEPTHS = (10, 20, 50)
DEFAULT_DEPTH = 20
SECONDS_PER_PAGE = 3.2  # measured on the M1 Pro, MAX_TOKENS = 1024


@cache
def model() -> Any:
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        MODEL_ID,
        device=device(),
        max_length=MAX_TOKENS,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )


def downloaded(model_id: str = MODEL_ID) -> bool:
    """The weights are on disk, so a request never waits on a multi-GB download."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
    except Exception:
        return False
    return True


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(settings().hf_home / "rerank_cache.sqlite")
    con.execute(
        "CREATE TABLE IF NOT EXISTS score (model TEXT, max_tokens INT, dataset TEXT, query TEXT,"
        " corpus_id INT, score REAL, PRIMARY KEY (model, max_tokens, dataset, query, corpus_id))"
    )
    return con


def rerank(
    dataset: str, query: str, candidates: list[tuple[int, str]]
) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    """(corpus_id, page text) pairs → reordered (corpus_id, logit), and what it cost."""
    t0 = time.time()
    with _db() as con:
        rows = con.execute(
            "SELECT corpus_id, score FROM score"
            " WHERE model=? AND max_tokens=? AND dataset=? AND query=?",
            (MODEL_ID, MAX_TOKENS, dataset, query),
        ).fetchall()
        scores = {int(c): float(s) for c, s in rows}
        todo = [(cid, txt) for cid, txt in candidates if cid not in scores]
        if todo:
            fresh = model().predict(
                [(query, txt) for _, txt in todo], batch_size=1, show_progress_bar=False
            )
            for (cid, _), s in zip(todo, fresh, strict=True):
                scores[cid] = float(s)
            con.executemany(
                "INSERT OR REPLACE INTO score VALUES (?, ?, ?, ?, ?, ?)",
                [(MODEL_ID, MAX_TOKENS, dataset, query, cid, scores[cid]) for cid, _ in todo],
            )
    ranked = sorted(((cid, scores[cid]) for cid, _ in candidates), key=lambda kv: (-kv[1], kv[0]))
    return ranked, {
        "scored": len(todo),
        "cached": len(candidates) - len(todo),
        "seconds": round(time.time() - t0, 1),
    }
