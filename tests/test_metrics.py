import math

import pytest

from vidore_loop.cli import SMOKE_QRELS, SMOKE_RANKING
from vidore_loop.eval.metrics import mean_over_queries, ndcg_at_k, recall_at_k


def test_ndcg_graded_linear_gain() -> None:
    # DCG = 2/log2(2) + 1/log2(4) = 2.5; IDCG = 2 + 1/log2(3) + 1/log2(4)
    ideal = 2 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(SMOKE_RANKING, SMOKE_QRELS) == pytest.approx(2.5 / ideal)


def test_ndcg_perfect_and_empty() -> None:
    assert ndcg_at_k([11, 12, 13], SMOKE_QRELS) == pytest.approx(1.0)
    assert ndcg_at_k([], SMOKE_QRELS) == 0.0
    assert ndcg_at_k([1, 2], {}) == 0.0


def test_ndcg_cuts_at_k_and_ignores_duplicates() -> None:
    assert ndcg_at_k([99, 11], {11: 1}, k=1) == 0.0
    assert ndcg_at_k([11, 11, 12], {11: 1, 12: 1}, k=2) == pytest.approx(1.0)


def test_recall() -> None:
    assert recall_at_k(SMOKE_RANKING, SMOKE_QRELS) == pytest.approx(2 / 3)
    assert recall_at_k(SMOKE_RANKING, SMOKE_QRELS, k=1) == pytest.approx(1 / 3)


def test_mean_over_queries_scores_missing_ranking_zero() -> None:
    qrels = {1: {10: 2}, 2: {20: 1}}
    assert mean_over_queries({1: [10]}, qrels) == pytest.approx(0.5)
    assert mean_over_queries({1: [10], 2: [20]}, qrels, metric="recall") == pytest.approx(1.0)


def test_report_and_stage_delta_on_the_smoke_ranking() -> None:
    from vidore_loop.eval.metrics import report, stage_delta

    r = report(SMOKE_RANKING, SMOKE_QRELS)
    assert r["first_rank"] == 1 and r["first_full_rank"] == 1
    assert r["hit_full_5"] and not r["complete_10"]  # page 13 is never retrieved
    assert r["precision_5"] == pytest.approx(2 / 5)
    assert r["recall_100"] == pytest.approx(2 / 3)
    assert stage_delta([99, 98], SMOKE_RANKING, SMOKE_QRELS) == {"gained": [11, 12], "lost": []}


def test_every_metric_matches_pytrec_eval() -> None:
    """Our implementations against trec_eval itself, on random graded qrels and rankings."""
    import random

    pytrec_eval = pytest.importorskip("pytrec_eval")
    from vidore_loop.eval import metrics as m

    rng = random.Random(300)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    ours: dict[str, dict[str, float]] = {}
    for q in range(40):
        pool = rng.sample(range(500), 120)
        rel = {cid: rng.choice([1, 1, 2]) for cid in rng.sample(pool, rng.randint(1, 8))}
        ranking = rng.sample(pool, rng.randint(5, 110))
        qrels[str(q)] = {str(c): g for c, g in rel.items()}
        run[str(q)] = {str(c): float(len(ranking) - i) for i, c in enumerate(ranking)}  # no ties
        ours[str(q)] = {
            "ndcg_cut_10": m.ndcg_at_k(ranking, rel, 10),
            "recall_5": m.recall_at_k(ranking, rel, 5),
            "recall_100": m.recall_at_k(ranking, rel, 100),
            "P_10": m.precision_at_k(ranking, rel, 10),
            "recip_rank": m.reciprocal_rank(ranking, rel),
            "map_cut_10": m.average_precision_at_k(ranking, rel, 10),
        }
    measures = {"ndcg_cut_10", "recall_5", "recall_100", "P_10", "recip_rank", "map_cut_10"}
    theirs = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    for q, vals in ours.items():
        for name, v in vals.items():
            assert v == pytest.approx(theirs[q][name], abs=1e-4), (q, name)
