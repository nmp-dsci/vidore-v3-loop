"""Stage C: reciprocal rank fusion of the channels' rankings.

A page scores Σ 1 / (K + rank) over the lists it appears in, so a page ranked high
by either channel rises and a page both channels like rises most. K = 60 is the
value from the RRF paper (Cormack et al., 2009) and the common default.

Each channel can carry a weight (w / (K + rank)). `WEIGHTS` is what the viewer uses;
M3 chose it on the train split, never on test: on finance_en train every text weight
above 0 lowered NDCG@10 (1.0 → 67.3, 0.25 → 72.7, 0 → 75.8), so fusion is visual alone
until a better text channel earns a weight back.
"""

from __future__ import annotations

from collections.abc import Sequence

K = 60
WEIGHTS: tuple[float, float] = (0.0, 1.0)  # (text, visual); chosen on train in M3


def rrf(
    rankings: Sequence[Sequence[int]],
    k: int = 50,
    rrf_k: int = K,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fused top k as (corpus_id, RRF score), best first; ties broken by corpus id for determinism."""
    ws = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(ws) != len(rankings):
        raise ValueError(f"{len(ws)} weights for {len(rankings)} rankings")
    score: dict[int, float] = {}
    for w, ranking in zip(ws, rankings, strict=True):
        if w == 0:
            continue
        for rank, cid in enumerate(dict.fromkeys(ranking), start=1):
            score[cid] = score.get(cid, 0.0) + w / (rrf_k + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
