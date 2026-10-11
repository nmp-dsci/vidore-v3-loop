"""Retrieval metrics on graded qrels, computed the way trec_eval (and so MTEB) does.

ViDoRe V3 grades a page 2 (fully relevant: holds the whole answer) or 1
(critically relevant: holds a required part). NDCG uses the grade as the gain
(linear, as `pytrec_eval`'s `ndcg_cut`), so our numbers are comparable with the
MTEB leaderboard's NDCG@10. Every other metric counts any grade > 0 as relevant,
except the full-answer ones, which count grade 2 only. `tests/test_metrics.py`
checks each one against `pytrec_eval` on random rankings.

One function, `report`, scores one ranking for the RAG pipeline tab (one question)
and for a run (averaged over questions), so the two can never disagree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

Qrels = Mapping[int, int]  # corpus_id -> grade, for one query
FULL = 2  # grade of a page that holds the whole answer
RECALL_KS = (1, 5, 10, 50, 100)


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10) -> float:
    """NDCG@k of one ranked list of corpus ids; 0.0 when the query has no relevant page."""
    ideal = dcg(sorted((g for g in qrels.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([qrels.get(cid, 0) for cid in _dedupe(ranking)[:k]]) / ideal


def _relevant(qrels: Qrels, min_grade: int = 1) -> set[int]:
    return {cid for cid, g in qrels.items() if g >= min_grade}


def recall_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10) -> float:
    """Coverage: the share of relevant pages found in the top k (trec_eval `recall_k`)."""
    relevant = _relevant(qrels)
    if not relevant:
        return 0.0
    return len(relevant & set(_dedupe(ranking)[:k])) / len(relevant)


def precision_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10) -> float:
    """The share of the top k that is relevant, over k even when fewer were returned (trec_eval `P_k`)."""
    return len(_relevant(qrels) & set(_dedupe(ranking)[:k])) / k


def complete_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10) -> bool:
    """Every relevant page is in the top k: what a multi-hop or list question needs."""
    relevant = _relevant(qrels)
    return bool(relevant) and relevant <= set(_dedupe(ranking)[:k])


def hit_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10, min_grade: int = FULL) -> bool:
    """At least one page of `min_grade` or better in the top k (default: a full-answer page)."""
    return bool(_relevant(qrels, min_grade) & set(_dedupe(ranking)[:k]))


def first_rank(ranking: Sequence[int], qrels: Qrels, min_grade: int = 1) -> int | None:
    """1-based rank of the first page of `min_grade` or better; None if it was not retrieved."""
    relevant = _relevant(qrels, min_grade)
    for i, cid in enumerate(_dedupe(ranking)):
        if cid in relevant:
            return i + 1
    return None


def reciprocal_rank(ranking: Sequence[int], qrels: Qrels) -> float:
    """1 / rank of the first relevant page (trec_eval `recip_rank`)."""
    r = first_rank(ranking, qrels)
    return 0.0 if r is None else 1.0 / r


def average_precision_at_k(ranking: Sequence[int], qrels: Qrels, k: int = 10) -> float:
    """AP cut at k, divided by all relevant pages (trec_eval `map_cut_k`)."""
    relevant = _relevant(qrels)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, cid in enumerate(_dedupe(ranking)[:k]):
        if cid in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / len(relevant)


def report(ranking: Sequence[int], qrels: Qrels) -> dict[str, Any]:
    """Every metric the RAG pipeline tab and the Runs tab show, for one ranking of one question."""
    out: dict[str, Any] = {
        "ndcg_10": ndcg_at_k(ranking, qrels, 10),
        "map_10": average_precision_at_k(ranking, qrels, 10),
        "mrr": reciprocal_rank(ranking, qrels),
        "first_rank": first_rank(ranking, qrels),
        "first_full_rank": first_rank(ranking, qrels, FULL),
        "precision_5": precision_at_k(ranking, qrels, 5),
        "precision_10": precision_at_k(ranking, qrels, 10),
        "hit_full_5": hit_at_k(ranking, qrels, 5),
        "hit_full_10": hit_at_k(ranking, qrels, 10),
        "complete_10": complete_at_k(ranking, qrels, 10),
        "complete_50": complete_at_k(ranking, qrels, 50),
        "n_relevant": len(_relevant(qrels)),
        "n_full": len(_relevant(qrels, FULL)),
    }
    for k in RECALL_KS:
        out[f"recall_{k}"] = recall_at_k(ranking, qrels, k)
    return out


def stage_delta(
    before: Sequence[int], after: Sequence[int], qrels: Qrels, k: int = 10
) -> dict[str, list[int]]:
    """Relevant pages a stage brought into the top k, and the ones it pushed out."""
    relevant = _relevant(qrels)
    b, a = set(_dedupe(before)[:k]) & relevant, set(_dedupe(after)[:k]) & relevant
    return {"gained": sorted(a - b), "lost": sorted(b - a)}


def mean_over_queries(
    rankings: Mapping[int, Sequence[int]],
    qrels: Mapping[int, Qrels],
    metric: str = "ndcg",
    k: int = 10,
) -> float:
    """Mean of a metric over every query in `qrels`; a query with no ranking scores 0."""
    fn = {"ndcg": ndcg_at_k, "recall": recall_at_k, "precision": precision_at_k}[metric]
    if not qrels:
        return 0.0
    return sum(fn(rankings.get(q, ()), rel, k) for q, rel in qrels.items()) / len(qrels)


def _dedupe(ranking: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(ranking))
