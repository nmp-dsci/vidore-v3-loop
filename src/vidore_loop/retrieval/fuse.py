"""Stage C: reciprocal rank fusion of the channels' rankings.

A page scores Σ 1 / (K + rank) over the lists it appears in, so a page ranked high
by either channel rises and a page both channels like rises most. K = 60 is the
value from the RRF paper (Cormack et al., 2009) and the common default.
"""

from __future__ import annotations

from collections.abc import Sequence

K = 60


def rrf(rankings: Sequence[Sequence[int]], k: int = 50, rrf_k: int = K) -> list[tuple[int, float]]:
    """Fused top k as (corpus_id, RRF score), best first; ties broken by corpus id for determinism."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(dict.fromkeys(ranking), start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
