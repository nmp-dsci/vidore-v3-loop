"""Stage D: zerank-2 rereads the fused top 50 and reorders them.

zerank-2 (ZeroEntropy, Apache-2.0) is a 4B cross-encoder: it reads the question and
one page's OCR markdown together and returns a relevance logit. Its hosted API closed
in September 2026, so it runs here, on the same device as the visual model. Pages are
cut to `MAX_TOKENS` tokens; finance pages run to ≈1,500.
"""

from __future__ import annotations

import time
from functools import cache
from typing import Any

from vidore_loop.retrieval.visual import device

MODEL_ID = "zeroentropy/zerank-2-reranker"
MAX_TOKENS = 1024


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


def loaded() -> bool:
    return model.cache_info().currsize > 0


def downloaded(model_id: str = MODEL_ID) -> bool:
    """The weights are on disk, so a request never waits on a multi-GB download."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
    except Exception:
        return False
    return True


def rerank(
    query: str, candidates: list[tuple[int, str]], batch_size: int = 4
) -> tuple[list[tuple[int, float]], float]:
    """(corpus_id, page text) pairs → reordered (corpus_id, logit), and the seconds it took."""
    t0 = time.time()
    scores = model().predict(
        [(query, text) for _, text in candidates], batch_size=batch_size, show_progress_bar=False
    )
    ranked = sorted(
        ((cid, float(s)) for (cid, _), s in zip(candidates, scores, strict=True)),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return ranked, time.time() - t0
