"""Retrieval metrics on graded qrels, computed the way trec_eval (and so MTEB) does.

ViDoRe V3 grades a page 2 (fully relevant: holds the whole answer) or 1
(critically relevant: holds a required part). NDCG uses the grade as the gain
(linear, as `pytrec_eval`'s `ndcg_cut`), so our numbers are comparable with the
MTEB leaderboard's NDCG@10. Recall counts any grade > 0 as relevant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[int, int]  # corpus_id -> grade, for one query


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10) -> float:
    """NDCG@k of one ranked list of corpus ids; 0.0 when the query has no relevant page."""
    ideal = dcg(sorted((g for g in qrels.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([qrels.get(cid, 0) for cid in _dedupe(ranking)[:k]]) / ideal


def recall_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10) -> float:
    relevant = {cid for cid, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(_dedupe(ranking)[:k])) / len(relevant)


def mean_over_queries(
    rankings: Mapping[int, Sequence[int]],
    qrels: Mapping[int, Qrels],
    metric: str = "ndcg",
    k: int = 10,
) -> float:
    """Mean of a metric over every query in `qrels`; a query with no ranking scores 0."""
    fn = {"ndcg": ndcg_at_k, "recall": recall_at_k}[metric]
    if not qrels:
        return 0.0
    return sum(fn(rankings.get(q, ()), rel, k) for q, rel in qrels.items()) / len(qrels)


def _dedupe(ranking: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(ranking))
